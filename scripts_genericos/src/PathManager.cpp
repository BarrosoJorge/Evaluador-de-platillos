#include "PathManager.hpp"
#include "Logger.hpp"
#include <vector>

namespace evaluador {

PathManager::PathManager(std::filesystem::path raiz) : raiz_(std::move(raiz)) {}

void PathManager::asegurarEstructura() const {
    const std::vector<std::filesystem::path> carpetas = {
        videosCrudos(),
        imagenesCrudas(),
        preprocesadas(),
        segmentadas(),
        featuresRaw(),
        featuresEstadisticosPorRegion(),
        featuresEstadisticosGlobal(),
        featuresVectorFinal(),
        comparacionCovarianzas(),
        comparacionDistancias(),
        fuzzyPercentiles(),
        fuzzyPertenencias(),
        scoresElementales(),
        rubricaReportes(),
        sesiones()
    };

    for (const auto& carpeta : carpetas) {
        std::error_code ec;
        std::filesystem::create_directories(carpeta, ec);
        if (ec) {
            Logger::instance().error("PathManager",
                "No se pudo crear la carpeta " + carpeta.string() + ": " + ec.message());
        }
    }

    Logger::instance().info("PathManager",
        "Estructura de carpetas asegurada bajo " + raiz_.string());
}

std::filesystem::path PathManager::videosCrudos() const { return raiz_ / "Data" / "Raw" / "Videos"; }
std::filesystem::path PathManager::imagenesCrudas() const { return raiz_ / "Data" / "Raw" / "Imagenes"; }
std::filesystem::path PathManager::preprocesadas() const { return raiz_ / "Data" / "Preprocesadas"; }
std::filesystem::path PathManager::segmentadas() const { return raiz_ / "Data" / "Segmentadas"; }

std::filesystem::path PathManager::featuresRaw() const { return raiz_ / "Data" / "Features" / "Raw"; }
std::filesystem::path PathManager::featuresEstadisticosPorRegion() const { return raiz_ / "Data" / "Features" / "Estadisticos" / "PorRegion"; }
std::filesystem::path PathManager::featuresEstadisticosGlobal() const { return raiz_ / "Data" / "Features" / "Estadisticos" / "Global"; }

std::filesystem::path PathManager::featuresVectorFinal() const {
    return raiz_ / "Data" / "Features" / "Estadisticos";
}

std::filesystem::path PathManager::comparacionCovarianzas() const { return raiz_ / "Data" / "Comparacion" / "Covarianzas"; }
std::filesystem::path PathManager::comparacionDistancias() const { return raiz_ / "Data" / "Comparacion" / "Distancias"; }
std::filesystem::path PathManager::fuzzyPercentiles() const { return raiz_ / "Data" / "Fuzzy" / "Percentiles"; }
std::filesystem::path PathManager::fuzzyPertenencias() const { return raiz_ / "Data" / "Fuzzy" / "Pertenencias"; }
std::filesystem::path PathManager::scoresElementales() const { return raiz_ / "Data" / "SistemaDifuso" / "ScoresElementales"; }
std::filesystem::path PathManager::rubricaReportes() const { return raiz_ / "Data" / "Rubrica" / "Reportes"; }
std::filesystem::path PathManager::sesiones() const { return raiz_ / "Data" / "Sesiones"; }

std::filesystem::path PathManager::manifestSesion(const std::string& idSesion) const {
    return sesiones() / (idSesion + "_manifest.json");
}

} // namespace evaluador