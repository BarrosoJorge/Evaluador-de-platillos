#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <optional>
#include <filesystem>

namespace evaluador {

// Una region nombrada dentro de la plantilla del chef (ej. "carne",
// "salsa", "pure"). Las coordenadas del rectangulo son RELATIVAS
// (0.0-1.0) al tamaño de la imagen preprocesada donde se definio,
// para que la plantilla siga siendo valida aunque la imagen del
// alumno no sea exactamente del mismo tamaño en pixeles (aunque en
// la practica, gracias a Preprocesador, ambas deberian ser 224x224).
struct RegionInfo {
    std::string nombre;
    double xRel = 0.0;
    double yRel = 0.0;
    double wRel = 0.0;
    double hRel = 0.0;
    cv::Vec3d colorPromedioBGR{0.0, 0.0, 0.0};

    // Convierte el rectangulo relativo a coordenadas de pixel para una
    // imagen de tamaño dado.
    cv::Rect aRectanguloPixeles(const cv::Size& tamanoImagen) const;
};

// Conjunto de regiones que el chef definio para un platillo/angulo
// especifico. Es la "plantilla" que Localizador usa como referencia
// para buscar las mismas regiones en cada imagen de alumno.
struct PlantillaRegiones {
    std::string claveDish; // ej. "Queretaro_Milanesa_Superior"
    std::vector<RegionInfo> regiones;

    // Guarda en un formato de texto plano simple (una linea por region).
    // No se uso JSON a proposito, para no agregar una dependencia
    // externa nueva al proyecto — si mas adelante prefieres JSON,
    // solo hay que cambiar guardar()/cargar(), la interfaz no cambia.
    bool guardar(const std::filesystem::path& ruta) const;

    static std::optional<PlantillaRegiones> cargar(const std::filesystem::path& ruta);
};

} // namespace evaluador
