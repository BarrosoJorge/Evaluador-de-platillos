#include "RegionTemplate.hpp"
#include "Logger.hpp"

#include <fstream>
#include <sstream>

namespace evaluador {

cv::Rect RegionInfo::aRectanguloPixeles(const cv::Size& tamanoImagen) const {
    int x = static_cast<int>(xRel * tamanoImagen.width);
    int y = static_cast<int>(yRel * tamanoImagen.height);
    int w = static_cast<int>(wRel * tamanoImagen.width);
    int h = static_cast<int>(hRel * tamanoImagen.height);

    cv::Rect r(x, y, w, h);
    // Recortar al area valida de la imagen, por si el redondeo se pasa
    // del borde.
    return r & cv::Rect(0, 0, tamanoImagen.width, tamanoImagen.height);
}

bool PlantillaRegiones::guardar(const std::filesystem::path& ruta) const {
    std::error_code ec;
    std::filesystem::create_directories(ruta.parent_path(), ec);

    std::ofstream archivo(ruta);
    if (!archivo.is_open()) {
        Logger::instance().error("RegionTemplate", "No se pudo escribir: " + ruta.string());
        return false;
    }

    archivo << "# plantilla_regiones\n";
    archivo << "dish=" << claveDish << "\n";
    archivo << "# nombre;x_rel;y_rel;w_rel;h_rel;b;g;r\n";

    for (const auto& region : regiones) {
        archivo << region.nombre << ";"
                << region.xRel << ";" << region.yRel << ";"
                << region.wRel << ";" << region.hRel << ";"
                << region.colorPromedioBGR[0] << ";"
                << region.colorPromedioBGR[1] << ";"
                << region.colorPromedioBGR[2] << "\n";
    }

    Logger::instance().info("RegionTemplate",
        "Plantilla guardada (" + std::to_string(regiones.size()) + " regiones): " + ruta.string());
    return true;
}

std::optional<PlantillaRegiones> PlantillaRegiones::cargar(const std::filesystem::path& ruta) {
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        Logger::instance().warn("RegionTemplate", "No existe la plantilla: " + ruta.string());
        return std::nullopt;
    }

    PlantillaRegiones plantilla;
    std::string linea;

    while (std::getline(archivo, linea)) {
        if (linea.empty() || linea.front() == '#') continue;

        if (linea.rfind("dish=", 0) == 0) {
            plantilla.claveDish = linea.substr(5);
            continue;
        }

        std::vector<std::string> campos;
        std::istringstream iss(linea);
        std::string campo;
        while (std::getline(iss, campo, ';')) {
            campos.push_back(campo);
        }

        if (campos.size() != 8) {
            Logger::instance().warn("RegionTemplate", "Linea invalida ignorada: " + linea);
            continue;
        }

        RegionInfo region;
        try {
            region.nombre = campos[0];
            region.xRel = std::stod(campos[1]);
            region.yRel = std::stod(campos[2]);
            region.wRel = std::stod(campos[3]);
            region.hRel = std::stod(campos[4]);
            region.colorPromedioBGR = cv::Vec3d(
                std::stod(campos[5]), std::stod(campos[6]), std::stod(campos[7]));
        } catch (const std::exception& e) {
            Logger::instance().warn("RegionTemplate", "Error parseando linea '" + linea + "': " + e.what());
            continue;
        }

        plantilla.regiones.push_back(region);
    }

    if (plantilla.regiones.empty()) {
        Logger::instance().error("RegionTemplate", "Plantilla sin regiones validas: " + ruta.string());
        return std::nullopt;
    }

    Logger::instance().info("RegionTemplate",
        "Plantilla cargada (" + std::to_string(plantilla.regiones.size()) + " regiones): " + ruta.string());
    return plantilla;
}

} // namespace evaluador
