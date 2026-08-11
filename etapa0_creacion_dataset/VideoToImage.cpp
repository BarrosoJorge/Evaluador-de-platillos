/*
VideoToImage: convierte videos ya etiquetados (formato
Ciudad_Platillo_Angulo_Autor_Calidad) en imagenes organizadas por
metadato, y extrae hasta N imagenes por video.

Cambios en esta version:

1. RECURSIVO (correccion de bug): antes usaba fs::directory_iterator
   (un solo nivel). Con tu estructura real, los videos estan anidados
   varios niveles (Data/Raw/Videos/CDMX/Decoración Limón/*.MP4,
   Queretaro/Huevo/Huevo/*.mov, etc.) — con el iterador de un solo
   nivel esto encontraba CERO videos. Ahora usa
   fs::recursive_directory_iterator.

2. FILTRO DE ANGULO: nuevo parametro opcional (4to argumento) para
   procesar solo un angulo especifico (ej. "Superior", ignorando
   "Lateral"). Por defecto es "Superior" porque fue lo que pediste;
   pasa "" (string vacio) como 4to argumento si en algun momento
   quieres procesar todos los angulos sin filtrar.

Conflicto de jerarquia de carpetas (de la version anterior, sigue igual):
  Tu comentario original describia la salida como
    Data/Raw/Imagenes/Ciudad/Platillo/Angulo/Autor[/Calidad]
  pero el PathManager que ya definimos en scripts_genericos usa un
  arbol PLANO:
    Data/Imagenes/Crudas   (sin subcarpetas por metadato)

  Se resolvio usando PathManager::imagenesCrudas() y
  PathManager::videosCrudos() como RAIZ, y anidando por metadato
  DEBAJO de esa raiz — conserva tu logica de organizacion pero ya no
  vive en una ruta hardcodeada de Windows/WSL.

Tambien: VideoMetadata (struct + parseFilename + format) ya no vive
aqui, se movio a scripts_genericos porque era identica a la de
Labeler.cpp.
*/

#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace evaluador;

// Construye la carpeta de salida anidada por metadato debajo de la raiz
// que le pasen (normalmente PathManager::imagenesCrudas()).
bool ensureOutputDir(const VideoMetadata& m, const fs::path& base, fs::path& outDir) {
    if (m.vacio()) return false;
    outDir = base / m.ciudad / m.platillo / m.angulo / m.autor;
    if (!m.calidad.empty()) outDir /= m.calidad;
    std::error_code ec;
    fs::create_directories(outDir, ec);
    return ec.value() == 0 || fs::exists(outDir);
}

// Extrae hasta nImages por video, distribuidos con paso fijo sobre el
// total de frames. Si nImages<=0 extrae todos los frames.
//
// Nota: a diferencia de ImageIO::extraerFramesUniformes (que hace seek
// directo por indice via CAP_PROP_POS_FRAMES), esta version lee
// secuencialmente con cap.read() y decide con un modulo cual frame
// guardar. Es mas lento pero mas confiable con codecs donde el seek
// por indice no es exacto — lo dejo como estaba porque es una eleccion
// valida, no un bug. Si algun dia el volumen de video crece mucho y la
// velocidad importa mas que la precision de codec, ImageIO ya tiene la
// alternativa por seek lista para usarse.
//
// anguloFiltro: si no esta vacio, se omite el video completo (sin
// siquiera abrirlo con VideoCapture) cuando su angulo no coincide.
// Se compara tal como quedo parseado — sensible a mayusculas/minusculas,
// asi que "superior" (minuscula) NO coincidiria con "Superior".
void processVideo(const fs::path& videoPath, const fs::path& outBase, int nImages,
                   const std::string& anguloFiltro) {
    std::string filename = videoPath.filename().string();
    VideoMetadata meta = parsearNombreArchivo(filename);

    if (!anguloFiltro.empty() && meta.angulo != anguloFiltro) {
        Logger::instance().debug("VideoToImage",
            "Omitido (angulo '" + meta.angulo + "' != '" + anguloFiltro + "'): " + filename);
        return;
    }

    fs::path outDir;
    if (!ensureOutputDir(meta, outBase, outDir)) {
        outDir = outBase / "unknown";
        fs::create_directories(outDir);
        Logger::instance().warn("VideoToImage",
            "Metadatos incompletos para " + filename + ", guardando en carpeta 'unknown'");
    }

    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        Logger::instance().error("VideoToImage", "No se pudo abrir video: " + videoPath.string());
        return;
    }

    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (total <= 0) total = 0;

    int saved = 0;
    int step = 1;
    if (nImages > 0 && total > 0) {
        step = std::max(1, total / nImages);
    }

    int idx = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        std::string outName = meta.formatear();
        if (outName.empty()) outName = filename.substr(0, filename.find_last_of('.'));

        if (nImages > 0) {
            if (idx % step == 0) {
                fs::path outPath = outDir / (outName + "_" + std::to_string(saved) + ".jpg");
                cv::imwrite(outPath.string(), frame);
                ++saved;
                if (saved >= nImages) break;
            }
        } else {
            fs::path outPath = outDir / (outName + "_" + std::to_string(idx) + ".jpg");
            cv::imwrite(outPath.string(), frame);
            ++saved;
        }

        ++idx;
    }

    cap.release();
    Logger::instance().info("VideoToImage",
        "Procesado: " + filename + " -> " + std::to_string(saved) + " imagenes en " + outDir.string());
}

int main(int argc, char** argv) {
    PathManager rutas(".");
    fs::path videoDir = rutas.videosCrudos();
    fs::path outBase = rutas.imagenesCrudas();
    int nImages = 10; // por defecto
    std::string anguloFiltro = "Superior"; // por defecto: solo Superior, pasa "" para desactivar

    if (argc >= 2) videoDir = argv[1];
    if (argc >= 3) outBase = argv[2];
    if (argc >= 4) nImages = std::stoi(argv[3]);
    if (argc >= 5) anguloFiltro = argv[4];

    if (!fs::exists(videoDir) || !fs::is_directory(videoDir)) {
        Logger::instance().error("VideoToImage", "Directorio de videos no existe: " + videoDir.string());
        return 1;
    }

    if (!anguloFiltro.empty()) {
        Logger::instance().info("VideoToImage", "Filtro de angulo activo: solo '" + anguloFiltro + "'");
    } else {
        Logger::instance().info("VideoToImage", "Sin filtro de angulo — se procesan todos.");
    }

    std::vector<std::string> exts = {".mp4", ".mov", ".avi", ".mkv"};
    int videosEncontrados = 0;

    for (const auto& entry : fs::recursive_directory_iterator(videoDir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;

        processVideo(entry.path(), outBase, nImages, anguloFiltro);
        ++videosEncontrados;
    }

    if (videosEncontrados == 0) {
        Logger::instance().warn("VideoToImage", "No se encontraron videos con extension soportada en " + videoDir.string());
    }

    Logger::instance().info("VideoToImage", "Proceso finalizado. Videos encontrados: " + std::to_string(videosEncontrados));
    return 0;
}

// Compilacion (ejemplo):
// g++ -std=c++17 -I../scripts_genericos/include \
//   ../scripts_genericos/src/Logger.cpp ../scripts_genericos/src/PathManager.cpp \
//   ../scripts_genericos/src/VideoMetadata.cpp \
//   VideoToImage.cpp -o VideoToImage `pkg-config --cflags --libs opencv4`
//
// Ejemplo de ejecucion:
// ./VideoToImage [ruta_videos] [ruta_salida] [n_imagenes_por_video] [angulo_filtro]
//   ./VideoToImage                     -> defaults, filtro "Superior"
//   ./VideoToImage . . 10 ""           -> sin filtro, procesa todos los angulos
