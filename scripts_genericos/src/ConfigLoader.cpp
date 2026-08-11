#include "ConfigLoader.hpp"
#include "Logger.hpp"

#include <fstream>

namespace evaluador {

ConfigLoader::ConfigLoader(const std::filesystem::path& rutaArchivo) {
    parsear(rutaArchivo);
}

void ConfigLoader::parsear(const std::filesystem::path& rutaArchivo) {
    std::ifstream archivo(rutaArchivo);
    if (!archivo.is_open()) {
        Logger::instance().error("ConfigLoader",
            "No se pudo abrir el archivo de configuracion: " + rutaArchivo.string());
        cargadoExitosamente_ = false;
        return;
    }

    std::string linea;
    while (std::getline(archivo, linea)) {
        std::string limpia = trim(linea);
        if (limpia.empty() || limpia.front() == '#') {
            continue;
        }

        auto posIgual = limpia.find('=');
        if (posIgual == std::string::npos) {
            Logger::instance().warn("ConfigLoader", "Linea ignorada (sin '='): " + limpia);
            continue;
        }

        std::string clave = trim(limpia.substr(0, posIgual));
        std::string valor = trim(limpia.substr(posIgual + 1));

        if (clave.empty()) {
            continue;
        }

        valores_[clave] = valor;
    }

    cargadoExitosamente_ = true;
    Logger::instance().info("ConfigLoader",
        "Configuracion cargada (" + std::to_string(valores_.size()) + " claves) desde " +
        rutaArchivo.string());
}

std::optional<std::string> ConfigLoader::getString(const std::string& clave) const {
    auto it = valores_.find(clave);
    if (it == valores_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::string ConfigLoader::trim(const std::string& s) {
    const char* espacios = " \t\r\n";
    auto inicio = s.find_first_not_of(espacios);
    if (inicio == std::string::npos) return "";
    auto fin = s.find_last_not_of(espacios);
    return s.substr(inicio, fin - inicio + 1);
}

} // namespace evaluador
