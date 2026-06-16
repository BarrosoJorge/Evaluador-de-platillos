#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

// Extensiones de video soportadas
const std::vector<std::string> SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"};

bool isSupportedExtension(const std::string& ext)
{
    std::string lower = ext;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& s : SUPPORTED_EXTENSIONS) {
        if (lower == s) return true;
    }
    return false;
}

// Dado el nombre base (sin extensión), quita el sufijo _X si existe
// Retorna el nuevo nombre base, o vacío si no tiene etiqueta
std::string removeLabel(const std::string& stem)
{
    // Busca el último '_' y verifica que lo que sigue sea la etiqueta (1+ chars)
    size_t pos = stem.find_last_of('_');
    if (pos == std::string::npos || pos == 0) return "";

    std::string label = stem.substr(pos + 1);
    if (label.empty()) return "";

    return stem.substr(0, pos);
}

int main(int argc, char* argv[])
{
    std::string directory = "/mnt/d/Data/Raw/Videos";
    if (argc >= 2) {
        directory = argv[1];
    }

    std::cout << "Eliminador de etiquetas de archivos de video" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Directorio: " << directory << std::endl << std::endl;

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Error: directorio no existe o no es válido: " << directory << std::endl;
        return 1;
    }

    std::vector<std::pair<fs::path, fs::path>> pending; // (old, new)

    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        if (!isSupportedExtension(ext)) continue;

        std::string stem = entry.path().stem().string();
        std::string new_stem = removeLabel(stem);

        if (new_stem.empty()) {
            std::cout << "[SIN ETIQUETA] " << entry.path().filename().string() << std::endl;
            continue;
        }

        fs::path new_path = entry.path().parent_path() / (new_stem + ext);
        pending.push_back({entry.path(), new_path});
        std::cout << "[DETECTADO]   " << entry.path().filename().string()
                  << "  ->  " << new_path.filename().string() << std::endl;
    }

    if (pending.empty()) {
        std::cout << "\nNo se encontraron archivos con etiqueta." << std::endl;
        return 0;
    }

    std::cout << "\n" << pending.size() << " archivo(s) serán renombrados." << std::endl;
    std::cout << "¿Continuar? (s/N): ";
    std::string confirm;
    std::getline(std::cin, confirm);

    if (confirm != "s" && confirm != "S") {
        std::cout << "Cancelado." << std::endl;
        return 0;
    }

    int ok = 0, fail = 0;
    for (const auto& [old_path, new_path] : pending) {
        if (fs::exists(new_path)) {
            std::cerr << "[ERROR] Ya existe: " << new_path.filename().string()
                      << " — se omite " << old_path.filename().string() << std::endl;
            fail++;
            continue;
        }
        try {
            fs::rename(old_path, new_path);
            std::cout << "[OK]   " << old_path.filename().string()
                      << "  ->  " << new_path.filename().string() << std::endl;
            ok++;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] " << old_path.filename().string() << ": " << e.what() << std::endl;
            fail++;
        }
    }

    std::cout << "\nCompletado: " << ok << " renombrados, " << fail << " fallidos." << std::endl;
    return (fail > 0) ? 1 : 0;
}

// Compilación:
// g++ -std=c++17 -o delabeler Delabeler.cpp

// Uso:
// ./delabeler [directorio]
// ./delabeler /mnt/d/Data/Raw/Videos
