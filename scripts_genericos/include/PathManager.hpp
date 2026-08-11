#pragma once
#include <filesystem>
#include <string>

namespace evaluador {

class PathManager {
public:
    explicit PathManager(std::filesystem::path raiz);
    void asegurarEstructura() const;

    // Etapa 0 y 1
    std::filesystem::path videosCrudos() const;
    std::filesystem::path imagenesCrudas() const;
    std::filesystem::path segmentadas() const;
    std::filesystem::path preprocesadas() const;

    // Etapa 3: Features
    std::filesystem::path featuresRaw() const;
    std::filesystem::path featuresEstadisticosPorRegion() const;
    std::filesystem::path featuresEstadisticosGlobal() const;
    std::filesystem::path featuresVectorFinal() const;
    
    // Etapa 4: Comparación
    std::filesystem::path comparacionCovarianzas() const;
    std::filesystem::path comparacionDistancias() const;
    
    // Etapa 5: Fuzzificación
    std::filesystem::path fuzzyPercentiles() const;
    std::filesystem::path fuzzyPertenencias() const;
    
    // Etapa 6 y 7: Sistema Difuso y Rúbrica
    std::filesystem::path scoresElementales() const;
    std::filesystem::path rubricaReportes() const;
    std::filesystem::path sesiones() const;
    std::filesystem::path manifestSesion(const std::string& idSesion) const;

private:
    std::filesystem::path raiz_;
};

} // namespace evaluador