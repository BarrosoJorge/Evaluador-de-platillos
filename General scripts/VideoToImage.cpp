/*
script para generacion de imagenes dado un video

Recibe un video (ya etiquetado) y devuelve nImagenes de ese video (tambien etiqeutadas)


1. Primero crea una copia de las carpetas, es decir, crea una copia de la jerarquia de las carpetas, pero vacias


2. Una vez creada las carpetas lee los videos y separa el video por frames y los guarda en una carpeta que corresponda

quedando la jerarquia final como 

Data
    |____Raw
            |__Videos (Aqui estan todos los videos)
            |__Imagenes
                    |____Ciudad
                            |___Platillo
                                    |______Superior
                                    |______Lateral
                                                |_____Chef
                                                |_____Estudiante
    |____Processed
*/


// Programa: convierte videos etiquetados a imágenes organizadas
// - Crea la jerarquía bajo Data/Raw/Imagenes
// - Extrae hasta N imágenes por video y las guarda en la carpeta correspondiente

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct VideoMetadata {
        std::string city;
        std::string dish;
        std::string angle;
        std::string author;
        std::string quality;

        std::string format() const {
                std::string result = city + "_" + dish + "_" + angle + "_" + author;
                if (!quality.empty()) result += "_" + quality;
                return result;
        }
};

VideoMetadata parseFilename(const std::string &filename) {
        VideoMetadata meta;
        std::string name = filename;
        // remover extension
        size_t dot = name.find_last_of('.');
        if (dot != std::string::npos) name = name.substr(0, dot);

        std::vector<std::string> parts;
        size_t start = 0, end = 0;
        while ((end = name.find('_', start)) != std::string::npos) {
                parts.push_back(name.substr(start, end - start));
                start = end + 1;
        }
        parts.push_back(name.substr(start));

        if (parts.size() >= 4) {
                meta.city = parts[0];
                meta.dish = parts[1];
                meta.angle = parts[2];
                meta.author = parts[3];
                if (parts.size() >= 5) meta.quality = parts[4];
        }
        return meta;
}

bool ensureOutputDir(const VideoMetadata &m, const fs::path &base, fs::path &outDir) {
        // Construir ruta: base / city / dish / angle / author [ / quality ]
        if (m.city.empty() || m.dish.empty() || m.angle.empty() || m.author.empty()) return false;
        outDir = base / m.city / m.dish / m.angle / m.author;
        if (!m.quality.empty()) outDir /= m.quality;
        std::error_code ec;
        fs::create_directories(outDir, ec);
        return ec.value() == 0 || fs::exists(outDir);
}

// Extrae hasta nImages por video. Si nImages<=0 extrae todos los frames.
void processVideo(const fs::path &videoPath, const fs::path &outBase, int nImages) {
        std::string filename = videoPath.filename().string();
        VideoMetadata meta = parseFilename(filename);

        fs::path outDir;
        if (!ensureOutputDir(meta, outBase, outDir)) {
                // Fallback: guardar en carpeta "unknown"
                outDir = outBase / "unknown";
                fs::create_directories(outDir);
        }

        cv::VideoCapture cap(videoPath.string());
        if (!cap.isOpened()) {
                std::cerr << "No se pudo abrir video: " << videoPath << std::endl;
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

                if (nImages > 0) {
                        if (idx % step == 0) {
                                std::string outName = meta.format();
                                if (outName.empty()) outName = filename.substr(0, filename.find_last_of('.'));
                                outName += "_" + std::to_string(saved) + ".jpg";
                                fs::path outPath = outDir / outName;
                                cv::imwrite(outPath.string(), frame);
                                ++saved;
                                if (saved >= nImages) break;
                        }
                } else {
                        // guardar todos los frames
                        std::string outName = meta.format();
                        if (outName.empty()) outName = filename.substr(0, filename.find_last_of('.'));
                        outName += "_" + std::to_string(idx) + ".jpg";
                        fs::path outPath = outDir / outName;
                        cv::imwrite(outPath.string(), frame);
                }

                ++idx;
        }

        cap.release();
        std::cout << "Procesado: " << filename << " -> " << saved << " imágenes en " << outDir << std::endl;
}

int main(int argc, char** argv) {
        // Base de videos y salida
        fs::path videoDir = "/home/jorge/Evaluador de platillos/Data/Raw/Videos";
        fs::path outBase = "/home/jorge/Evaluador de platillos/Data/Raw/Imagenes";
        int nImages = 10; // por defecto

        if (argc >= 2) videoDir = argv[1];
        if (argc >= 3) outBase = argv[2];
        if (argc >= 4) nImages = std::stoi(argv[3]);

        if (!fs::exists(videoDir) || !fs::is_directory(videoDir)) {
                std::cerr << "Directorio de videos no existe: " << videoDir << std::endl;
                return 1;
        }

        std::vector<std::string> exts = {".mp4",".mov",".avi",".mkv"};

        for (const auto &entry : fs::directory_iterator(videoDir)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;
                processVideo(entry.path(), outBase, nImages);
        }

        std::cout << "Proceso finalizado." << std::endl;
        return 0;
}


//Compilación: g++ -std=c++17 -o VideoToImage VideoToImage.cpp `pkg-config --cflags --libs opencv4`
//Ejemplo de ejecución: ./VideoToImage [ruta_videos] [ruta_salida] [n_imagenes_por_video]

