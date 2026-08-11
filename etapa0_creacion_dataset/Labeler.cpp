/*
Labeler: sesion interactiva para renombrar videos crudos segun la
convencion Ciudad_Platillo_Angulo_Autor_Calidad. Muestra el primer
frame de cada video como referencia visual mientras el usuario escribe
el nuevo nombre.

Cambios respecto a tu version original:
  - VideoMetadata ya no vive aqui: se movio a scripts_genericos, porque
    era identica a la de VideoToImage.cpp y el riesgo de que ambas
    copias se desincronizaran con el tiempo era real.
  - validateMetadata() tenia un "return true;" al inicio que desactivaba
    TODA la validacion (angulo, autor, calidad) sin decirlo en ningun
    log. Lo quite y dejo la validacion real activa via
    evaluador::validarMetadata(). Si la desactivaste a proposito porque
    aun estabas probando con nombres fuera de formato, dimelo y lo
    regreso pero con un log de WARN explicito en vez de silencioso.
  - Quite la funcion preview() (el thread standalone): interactiveSession
    ya reimplementaba la misma logica de imshow/waitKey en linea, asi
    que preview() estaba declarada y nunca se usaba. Si la necesitas
    para otro flujo (ej. previsualizar un solo video sin sesion
    interactiva completa), la regreso.
*/

#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <algorithm>

namespace fs = std::filesystem;
using namespace evaluador;

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

struct Video {
    cv::Mat first_frame;               // Primer frame del video, usado como preview
    std::string current_filename;      // Nombre actual del archivo
    std::string file_path;             // Ruta completa del archivo
    VideoMetadata metadata;            // Metadatos estructurados (modulo compartido)
    int frame_width;
    int frame_height;
    cv::VideoCapture capture;
    double fps;
    int64_t total_frames;

    Video(const std::string& path)
        : file_path(path), current_filename(fs::path(path).filename()), fps(0), total_frames(0)
    {
        first_frame = cv::Mat();
        if (!capture.open(path)) {
            throw std::runtime_error("No se pudo abrir el archivo: " + path);
        }

        frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = capture.get(cv::CAP_PROP_FPS);
        total_frames = static_cast<int64_t>(capture.get(cv::CAP_PROP_FRAME_COUNT));

        // Solo se lee el primer frame (modo batch): no queremos tener
        // todos los videos completos abiertos en memoria a la vez.
        if (!capture.read(first_frame)) {
            first_frame = cv::Mat();
        }

        if (capture.isOpened()) {
            capture.release();
        }
    }

    ~Video() {
        if (capture.isOpened()) {
            capture.release();
        }
    }

    double getDuration() const {
        if (fps == 0) return 0;
        return total_frames / fps;
    }
};

// ============================================================================
// DECLARACIONES
// ============================================================================

std::vector<std::unique_ptr<Video>> batchLoad(const std::string& directory);
bool renameVideo(Video* video, const VideoMetadata& new_metadata);
bool renameVideoByName(Video* video, const std::string& new_basename);
void interactiveSession(std::vector<std::unique_ptr<Video>>& videos);
std::string getExtension(const std::string& filename);

// ============================================================================
// IMPLEMENTACIONES
// ============================================================================

std::string getExtension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) return "";
    return filename.substr(dot_pos);
}

// Carga el primer frame de cada video dado un directorio.
std::vector<std::unique_ptr<Video>> batchLoad(const std::string& directory) {
    std::vector<std::unique_ptr<Video>> videos;

    if (!fs::exists(directory)) {
        Logger::instance().error("Labeler", "El directorio no existe: " + directory);
        return videos;
    }
    if (!fs::is_directory(directory)) {
        Logger::instance().error("Labeler", "La ruta no es un directorio: " + directory);
        return videos;
    }

    const std::vector<std::string> supported_extensions = {".mp4", ".mov", ".avi", ".mkv"};

    try {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            bool soportado = std::any_of(
                supported_extensions.begin(), supported_extensions.end(),
                [&ext](const std::string& s) { return ext == s; }
            );
            if (!soportado) continue;

            try {
                auto video = std::make_unique<Video>(entry.path().string());
                Logger::instance().debug("Labeler", "Cargado: " + entry.path().filename().string());
                videos.push_back(std::move(video));
            } catch (const std::exception& e) {
                Logger::instance().warn("Labeler",
                    "No se pudo cargar " + entry.path().filename().string() + ": " + e.what());
            }
        }
    } catch (const std::exception& e) {
        Logger::instance().error("Labeler", "Error al recorrer directorio: " + std::string(e.what()));
    }

    Logger::instance().info("Labeler", std::to_string(videos.size()) + " video(s) cargados.");
    return videos;
}

// Renombra el archivo del video si los metadatos son validos.
bool renameVideo(Video* video, const VideoMetadata& new_metadata) {
    if (!video) {
        Logger::instance().error("Labeler", "Puntero de video nulo");
        return false;
    }

    auto error = validarMetadata(new_metadata);
    if (error.has_value()) {
        Logger::instance().error("Labeler", "Metadatos invalidos: " + error.value());
        return false;
    }

    std::string extension = getExtension(video->current_filename);
    std::string base_name = new_metadata.formatear();
    std::string new_filename = base_name + extension;

    fs::path old_path = video->file_path;
    fs::path new_path = old_path.parent_path() / new_filename;

    // Si el nombre ya existe, agrega un sufijo numerico incremental en
    // vez de sobreescribir.
    int counter = 1;
    while (fs::exists(new_path)) {
        new_filename = base_name + std::to_string(counter) + extension;
        new_path = old_path.parent_path() / new_filename;
        counter++;
    }

    try {
        video->capture.release();
        fs::rename(old_path, new_path);

        Logger::instance().info("Labeler",
            "Renombrado: " + video->current_filename + " -> " + new_filename);

        video->current_filename = new_filename;
        video->file_path = new_path.string();
        video->metadata = new_metadata;

        if (!video->capture.open(video->file_path)) {
            Logger::instance().warn("Labeler", "No se pudo reabrir el video despues del renombrado");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        Logger::instance().error("Labeler", "Error durante renombrado: " + std::string(e.what()));
        video->capture.open(video->file_path);
        return false;
    }
}

// Renombra usando el nombre base proporcionado (sin extension).
bool renameVideoByName(Video* video, const std::string& new_basename) {
    if (!video) return false;

    std::string base = new_basename;
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base = base.substr(0, dot);
    }

    std::string extension = getExtension(video->current_filename);
    std::string candidate = base + extension;

    VideoMetadata meta = parsearNombreArchivo(candidate);

    auto error = validarMetadata(meta);
    if (error.has_value()) {
        Logger::instance().error("Labeler",
            "Metadatos invalidos para '" + base + "': " + error.value());
        return false;
    }

    return renameVideo(video, meta);
}

// Sesion interactiva para etiquetar videos.
void interactiveSession(std::vector<std::unique_ptr<Video>>& videos) {
    if (videos.empty()) {
        Logger::instance().info("Labeler", "No hay videos para procesar");
        return;
    }

    cv::namedWindow("Preview", cv::WINDOW_NORMAL);

    for (size_t i = 0; i < videos.size(); ++i) {
        Video* current_video = videos[i].get();

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Video " << (i + 1) << " de " << videos.size() << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Nombre actual: " << current_video->current_filename << std::endl;

        const int maxW = 480;
        const int maxH = 320;
        int fw = current_video->first_frame.cols;
        int fh = current_video->first_frame.rows;

        double scale = 1.0;
        if (fw > maxW || fh > maxH) {
            double sx = static_cast<double>(maxW) / fw;
            double sy = static_cast<double>(maxH) / fh;
            scale = std::min(sx, sy);
        }

        cv::Mat display;
        if (scale != 1.0) {
            cv::resize(current_video->first_frame, display, cv::Size(), scale, scale, cv::INTER_AREA);
        } else {
            display = current_video->first_frame;
        }

        std::cout << "\nIngresa nuevo nombre (formato Ciudad_Platillo_Angulo_Autor[_Calidad])" << std::endl;
        std::cout << "Dejar vacio y presionar ENTER conserva el nombre actual." << std::endl;
        std::cout << "Nuevo nombre: ";

        // Lectura de consola en hilo secundario para que std::getline no
        // congele la ventana de preview (imshow/waitKey deben correr en
        // el hilo principal).
        std::string input;
        std::atomic<bool> input_ready(false);

        std::thread input_thread([&input, &input_ready]() {
            std::getline(std::cin, input);
            input_ready.store(true);
        });

        while (!input_ready.load()) {
            cv::imshow("Preview", display);
            cv::waitKey(30);
        }

        input_thread.join();

        if (input.empty()) {
            Logger::instance().info("Labeler", "Conservando nombre: " + current_video->current_filename);
            continue;
        }

        bool renamed = renameVideoByName(current_video, input);
        if (!renamed) {
            std::cout << "Fallo al renombrar con el nombre proporcionado. Se conserva el nombre actual." << std::endl;
        }
    }

    cv::destroyWindow("Preview");
    Logger::instance().info("Labeler", "Proceso completado.");
}

// ============================================================================
// FUNCION PRINCIPAL
// ============================================================================

int main(int argc, char* argv[]) {
    // PathManager reemplaza la ruta hardcodeada anterior. Si necesitas
    // apuntar a otra carpeta puntual, pasala como argumento.
    PathManager rutas(".");
    std::string video_directory = rutas.videosCrudos().string();
    if (argc >= 2) {
        video_directory = argv[1];
    }

    Logger::instance().info("Labeler", "Sistema de Reindexacion Manual de Videos");
    Logger::instance().info("Labeler", "Directorio: " + video_directory);

    try {
        std::vector<std::unique_ptr<Video>> videos = batchLoad(video_directory);

        if (videos.empty()) {
            Logger::instance().error("Labeler", "No se encontraron videos en el directorio");
            return 1;
        }

        interactiveSession(videos);
        // Los unique_ptr se limpian automaticamente al salir del scope.

    } catch (const std::exception& e) {
        Logger::instance().error("Labeler", "Error fatal: " + std::string(e.what()));
        return 1;
    }

    return 0;
}

// Compilacion (ejemplo):
// g++ -std=c++17 -I../scripts_genericos/include \
//   ../scripts_genericos/src/Logger.cpp ../scripts_genericos/src/PathManager.cpp \
//   ../scripts_genericos/src/VideoMetadata.cpp \
//   Labeler.cpp -o labeler `pkg-config --cflags --libs opencv4`
//
// Ejemplo de ejecucion:
// ./labeler [directorio opcional]
