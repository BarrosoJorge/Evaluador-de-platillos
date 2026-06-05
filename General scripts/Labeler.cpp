#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <regex>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>

namespace fs = std::filesystem;

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

struct VideoMetadata 
{
    std::string city;           // Ciudad (Carpeta origen)
    std::string dish;           // Platillo
    std::string angle;          // Superior o Lateral
    std::string author;         // Chef o Estudiante
    std::string quality;        // B, R, M (solo para Estudiante)
    
    // Retorna el nombre formateado según: Ciudad_Platillo_Angulo_Autor_Clase
    std::string format() const
    {
        std::string result = city + "_" + dish + "_" + angle + "_" + author;
        if (!quality.empty()) {
            result += "_" + quality;
        }
        return result;
    }
};

struct Video 
{
    cv::Mat first_frame;               // Primer frame del video
    std::string current_filename;      // Nombre actual del archivo
    std::string file_path;             // Ruta completa del archivo
    VideoMetadata metadata;            // Metadatos estructurados
    int frame_width;                   // Ancho del frame
    int frame_height;                  // Alto del frame
    cv::VideoCapture capture;         // Capture (usado temporalmente)
    double fps;                       // fps del video
    int64_t total_frames;             // cantidad total de frames
    
    // Constructor que abre el video
    Video(const std::string& path) 
        : file_path(path), current_filename(fs::path(path).filename()), fps(0), total_frames(0)
    {
        first_frame = cv::Mat();
        if (!capture.open(path)) {
            throw std::runtime_error("No se pudo abrir el archivo: " + path);
        }

        // Extraer propiedades del video
        frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = capture.get(cv::CAP_PROP_FPS);
        total_frames = static_cast<int64_t>(capture.get(cv::CAP_PROP_FRAME_COUNT));

        // Leer solo el primer frame (batch mode)
        if (!capture.read(first_frame)) {
            first_frame = cv::Mat();
        }

        // Liberar la captura: ya tenemos el primer frame en memoria
        if (capture.isOpened()) {
            capture.release();
        }
    }
    
    // Destructor (libera automáticamente con unique_ptr)
    ~Video() 
    {
        if (capture.isOpened()) {
            capture.release();
        }
    }
    
    // Método para obtener duration en segundos
    double getDuration() const
    {
        if (fps == 0) return 0;
        return total_frames / fps;
    }
};

// ============================================================================
// DECLARACIONES DE FUNCIONES
// ============================================================================

std::vector<std::unique_ptr<Video>> batchLoad(const std::string& directory);
std::thread preview(Video* video, std::atomic<bool>& stopFlag, int frame_interval = 30);
bool renameVideo(Video* video, const VideoMetadata& new_metadata);
bool renameVideoByName(Video* video, const std::string& new_basename);
VideoMetadata parseFilename(const std::string& filename);
bool validateMetadata(const VideoMetadata& metadata);
void interactiveSession(std::vector<std::unique_ptr<Video>>& videos);
std::string getExtension(const std::string& filename);

// ============================================================================
// IMPLEMENTACIONES
// ============================================================================

// Obtiene la extensión de un archivo
std::string getExtension(const std::string& filename)
{
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return "";
    }
    return filename.substr(dot_pos);
}

// Valida que los metadatos cumplan con el formato especificado
bool validateMetadata(const VideoMetadata& metadata)
{

    //! temporal
    return true;

    
    // Validar que city no esté vacío y no contenga caracteres especiales
    if (metadata.city.empty() || !std::regex_match(metadata.city, std::regex("[a-zA-Z0-9_]+"))) {
        std::cerr << "Error: Ciudad inválida: " << metadata.city << std::endl;
        return false;
    }
    
    // Validar dish
    if (metadata.dish.empty() || !std::regex_match(metadata.dish, std::regex("[a-zA-Z0-9_]+"))) {
        std::cerr << "Error: Platillo inválido: " << metadata.dish << std::endl;
        return false;
    }
    
    // Validar angle (debe ser Superior o Lateral)
    if (metadata.angle != "Superior" && metadata.angle != "Lateral") {
        std::cerr << "Error: Ángulo inválido: " << metadata.angle 
                  << " (debe ser 'Superior' o 'Lateral')" << std::endl;
        return false;
    }
    
    // Validar author
    if (metadata.author != "Chef" && metadata.author != "Estudiante") {
        std::cerr << "Error: Autor inválido: " << metadata.author 
                  << " (debe ser 'Chef' o 'Estudiante')" << std::endl;
        return false;
    }
    
    // Validar quality (solo para Estudiante, debe ser B, R o M)
    if (metadata.author == "Estudiante") {
        if (metadata.quality.empty() || 
            (metadata.quality != "B" && metadata.quality != "R" && metadata.quality != "M")) {
            std::cerr << "Error: Calidad inválida: " << metadata.quality 
                      << " (debe ser 'B', 'R' o 'M')" << std::endl;
            return false;
        }
    } else if (!metadata.quality.empty()) {
        std::cerr << "Error: Chef no debe tener calidad especificada" << std::endl;
        return false;
    }
    
    return true;
}

// Intenta parsear un nombre de archivo en metadatos
VideoMetadata parseFilename(const std::string& filename)
{
    VideoMetadata metadata;
    
    // Remover extensión
    std::string name = filename.substr(0, filename.find_last_of('.'));
    
    // Dividir por underscore
    std::vector<std::string> parts;
    size_t start = 0, end = 0;
    
    while ((end = name.find('_', start)) != std::string::npos) {
        parts.push_back(name.substr(start, end - start));
        start = end + 1;
    }
    parts.push_back(name.substr(start));
    
    // Asignar partes según la cantidad
    if (parts.size() >= 4) {
        metadata.city = parts[0];
        metadata.dish = parts[1];
        metadata.angle = parts[2];
        metadata.author = parts[3];
        if (parts.size() >= 5) {
            metadata.quality = parts[4];
        }
    }
    
    return metadata;
}

// Carga el primer frame de cada video dado un directorio
// Carga el primer frame de cada video dado un directorio
std::vector<std::unique_ptr<Video>> batchLoad(const std::string& directory)
{
    std::vector<std::unique_ptr<Video>> videos;
    
    if (!fs::exists(directory)) {
        std::cerr << "Error: El directorio no existe: " << directory << std::endl;
        return videos;
    }
    
    if (!fs::is_directory(directory)) {
        std::cerr << "Error: La ruta no es un directorio: " << directory << std::endl;
        return videos;
    }
    
    // Extensiones soportadas
    const std::vector<std::string> supported_extensions = {".mp4", ".mov", ".avi", ".mkv"};
    
    try {
        // CAMBIO AQUÍ: Usar recursive_directory_iterator en lugar de directory_iterator
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            
            // Si es una carpeta, el iterador recursivo entra automáticamente, 
            // pero nosotros solo queremos procesar los archivos finales.
            if (!entry.is_regular_file()) {
                continue;
            }
            
            std::string ext = entry.path().extension().string();
            
            // Convertir a minúsculas para comparación
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            // Verificar si la extensión es soportada
            bool is_supported = std::any_of(
                supported_extensions.begin(),
                supported_extensions.end(),
                [&ext](const std::string& s) { return ext == s; }
            );
            
            if (!is_supported) {
                continue;
            }
            
            try {
                // Usar unique_ptr para gestión automática de memoria
                std::unique_ptr<Video> video = std::make_unique<Video>(entry.path().string());
                std:: cout << "Cargado : " << entry.path().filename() << std::endl;
                videos.push_back(std::move(video));
                
            } catch (const std::exception& e) {
                std::cerr << "Advertencia al cargar " << entry.path().filename() 
                          << ": " << e.what() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error al recorrer directorio: " << e.what() << std::endl;
    }
    
    std:: cout << "FIN DE FUNCION BATCHLOAD: " << videos.size() << " videos cargados." << std::endl;
    return videos;
}
// Muestra una vista previa del video (una imagen estatica) en un hilo
std::thread preview(Video* video, std::atomic<bool>& stopFlag, int frame_interval)
{
    if (!video) {
        std::cerr << "Error: Video nulo" << std::endl;
        return std::thread();
    }

    if (video->first_frame.empty()) {
        std::cerr << "Error: No hay primer frame disponible para " << video->current_filename << std::endl;
        return std::thread();
    }

    // Ventana pequeña: escalar la imagen para que quepa en 480x320 como máximo
    const int maxW = 480;
    const int maxH = 320;
    int fw = video->first_frame.cols;
    int fh = video->first_frame.rows;

    double scale = 1.0;
    if (fw > maxW || fh > maxH) {
        double sx = static_cast<double>(maxW) / fw;
        double sy = static_cast<double>(maxH) / fh;
        scale = std::min(sx, sy);
    }

    cv::Mat display;
    if (scale != 1.0) {
        cv::resize(video->first_frame, display, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        display = video->first_frame;
    }

    std::string win_name = std::string("Preview - ") + video->current_filename;

    // Lanzar hilo que mantenga la GUI activa hasta que stopFlag sea true
    return std::thread([display, win_name, &stopFlag, frame_interval]() mutable {
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        while (!stopFlag.load()) {
            cv::imshow(win_name, display);
            cv::waitKey(frame_interval);
        }
        cv::destroyWindow(win_name);
    });
}

// Renombra el archivo del video si los metadatos son válidos
// Renombra el archivo del video si los metadatos son válidos
bool renameVideo(Video* video, const VideoMetadata& new_metadata)
{
    if (!video) {
        std::cerr << "Error: Puntero de video nulo" << std::endl;
        return false;
    }
    
    // Validar nuevos metadatos
    if (!validateMetadata(new_metadata)) {
        std::cerr << "Error: Metadatos inválidos para renombrado" << std::endl;
        return false;
    }
    
    // Construir nuevo nombre base
    std::string extension = getExtension(video->current_filename);
    std::string base_name = new_metadata.format();
    std::string new_filename = base_name + extension;
    
    fs::path old_path = video->file_path;
    fs::path new_path = old_path.parent_path() / new_filename;
    
    // --- LÓGICA DE SECUENCIA NUMÉRICA ---
    // Si el archivo ya existe, le agregamos 1, 2, 3... al final del base_name
    int counter = 1;
    while (fs::exists(new_path)) {
        new_filename = base_name + std::to_string(counter) + extension;
        new_path = old_path.parent_path() / new_filename;
        counter++;
    }
    // ------------------------------------
    
    try {
        // Cerrar la captura antes de renombrar
        video->capture.release();
        
        // Renombrar archivo
        fs::rename(old_path, new_path);
        
        std::cout << "Renombrado: " << video->current_filename << " -> " << new_filename << std::endl;
        
        // Actualizar datos del video
        video->current_filename = new_filename;
        video->file_path = new_path.string();
        video->metadata = new_metadata;
        
        // Reabrir video
        if (!video->capture.open(video->file_path)) {
            std::cerr << "Advertencia: No se pudo reabrir el video después del renombrado" << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error durante renombrado: " << e.what() << std::endl;
        // Intentar reabrir el video original
        video->capture.open(video->file_path);
        return false;
    }
}
// Renombra usando el nombre base proporcionado (sin extensión)
bool renameVideoByName(Video* video, const std::string& new_basename)
{
    if (!video) return false;

    // Quitar posible extension del new_basename
    std::string base = new_basename;
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base = base.substr(0, dot);
    }

    // Construir nombre temporal con extension para parseo
    std::string extension = getExtension(video->current_filename);
    std::string candidate = base + extension;

    VideoMetadata meta = parseFilename(candidate);

    if (!validateMetadata(meta)) {
        std::cerr << "Metadatos inválidos para el nombre proporcionado: " << base << std::endl;
        return false;
    }

    return renameVideo(video, meta);
}

// Sesión interactiva para etiquetar videos
// Sesión interactiva para etiquetar videos
void interactiveSession(std::vector<std::unique_ptr<Video>>& videos)
{
    if (videos.empty()) {
        std::cout << "No hay videos para procesar" << std::endl;
        return;
    }

    // CREAMOS LA VENTANA UNA SOLA VEZ AL INICIO (Hilo Principal)
    cv::namedWindow("Preview", cv::WINDOW_NORMAL);

    for (size_t i = 0; i < videos.size(); ++i) {
        Video* current_video = videos[i].get();

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Video " << (i + 1) << " de " << videos.size() << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Nombre actual: " << current_video->current_filename << std::endl;

        // Escalar la imagen si es necesario para la preview
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
        std::cout << "Dejar vacío y presionar ENTER conserva el nombre actual." << std::endl;
        std::cout << "Nuevo nombre: ";

        // LECTURA DE CONSOLA EN HILO SECUNDARIO
        // Así std::getline no congela la interfaz gráfica
        std::string input;
        std::atomic<bool> input_ready(false);
        
        std::thread input_thread([&input, &input_ready]() {
            std::getline(std::cin, input);
            input_ready.store(true);
        });

        // BUCLE GUI (IMSHOW / WAITKEY) SE MANTIENE EN EL HILO PRINCIPAL
        while (!input_ready.load()) {
            cv::imshow("Preview", display);
            cv::waitKey(30); // Refresca la ventana y mantiene WSLg vivo
        }

        // Esperar a que el usuario haya dado ENTER para limpiar el hilo
        input_thread.join();

        if (input.empty()) {
            std::cout << "Conservando nombre: " << current_video->current_filename << std::endl;
            continue;
        }

        // Intentar renombrar
        bool renamed = renameVideoByName(current_video, input);
        if (!renamed) {
            std::cout << "Fallo al renombrar con el nombre proporcionado. Se conserva el nombre actual." << std::endl;
        }
    }

    // Cerramos la ventana hasta que terminamos toda la sesión
    cv::destroyWindow("Preview");
    std::cout << "\nProceso completado." << std::endl;
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main()
{
    // Usar el directorio de windows  D:\Data\Raw\Videos
    std::string video_directory = "/mnt/d/Data/Raw/Videos";
    
    std::cout << "Sistema de Reindexación Manual de Videos" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Directorio: " << video_directory << std::endl << std::endl;
    
    try {
        // Cargar todos los videos del directorio
        std::vector<std::unique_ptr<Video>> videos = batchLoad(video_directory);
        
        if (videos.empty()) {
            std::cerr << "No se encontraron videos en el directorio" << std::endl;
            return 1;
        }
        
        std::cout << "\nVideos cargados: " << videos.size() << std::endl;
        
        // Iniciar sesión interactiva
        interactiveSession(videos);
        
        // Los unique_ptr se limpian automáticamente al salir del scope
        
    } catch (const std::exception& e) {
        std::cerr << "Error fatal: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

//Compilación (ejemplo):
// g++ -std=c++17 -o labeler Labeler.cpp `pkg-config --cflags --libs opencv4` 

//Ejemplo de ejecución:
// ./labeler