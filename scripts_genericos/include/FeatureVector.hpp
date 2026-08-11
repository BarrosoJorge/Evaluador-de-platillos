#pragma once

#include "ColorEstadisticas.hpp"
#include "TexturaGLCM.hpp"
#include "ReductorEstadisticos.hpp"

#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace evaluador {

// Vector de features de UNA region (ej. "carne") para UNA imagen.
// Combina las 3 fuentes que definimos para la etapa 3: color, textura
// GLCM y LBP resumido.
struct RegionFeatureVector {
    std::string nombreRegion;
    EstadisticosColor color;
    FeaturesGLCM textura;
    EstadisticosHistograma lbp; // ya reducido via reducirHistograma()
};

// Features globales de UNA imagen completa (no por region).
struct GlobalFeatureVector {
    double simetria = 0.0;
    double limpieza = 0.0;
    double enfoque = 0.0;
    double volumenGeneral = 0.0; // <-- Reemplaza a las proporciones
};

// Guarda los vectores por region de una imagen a un archivo de texto
// (una linea por region). No se junta con GlobalFeatureVector en el
// mismo archivo a proposito: PorRegion y Global se calculan con
// VectorizadorPorRegion.cpp y VectorizadorGlobal.cpp por separado
// (dos ramas en paralelo, tal como se definio la etapa 3), y
// Ensamblador.cpp es quien despues los junta en un solo vector final.
bool guardarFeaturesPorRegion(const std::vector<RegionFeatureVector>& regiones,
                               const std::filesystem::path& ruta);

std::optional<std::vector<RegionFeatureVector>> cargarFeaturesPorRegion(
    const std::filesystem::path& ruta);

bool guardarFeaturesGlobal(const GlobalFeatureVector& global,
                            const std::filesystem::path& ruta);

std::optional<GlobalFeatureVector> cargarFeaturesGlobal(const std::filesystem::path& ruta);

// --- Aplanado a vector numerico (para la etapa 4: Mahalanobis) ---
//
// El orden de los campos DEBE ser el mismo para todos los alumnos de
// una sesion, o la matriz de covarianza queda sin sentido (columna 3
// significaria "contraste" para un alumno y "idf" para otro). Estas
// funciones son la unica fuente de verdad de ese orden — nadie mas
// deberia reconstruir el orden de campos a mano.

// RegionFeatureVector -> fila 1x17: [mediaB,mediaG,mediaR,stdB,stdG,stdR,
//   energia,contraste,correlacion,homogeneidad,idf,entropia,varianza,
//   lbp_energia,lbp_entropia,lbp_varianza,lbp_media]
cv::Mat regionAVector(const RegionFeatureVector& r);

// Sub-vectores de UNA sola fuente (color, textura o LBP), para calcular
// Mahalanobis a nivel de "dimension elemental" (ej. textura_carne por
// separado de color_carne) en vez de mezclar las 3 fuentes en un solo
// vector de 17 columnas. Ver comentario en ComparadorSesion.cpp (etapa 4)
// sobre por que se separaron.
cv::Mat colorAVector(const EstadisticosColor& c);       // 1x6
cv::Mat texturaAVector(const FeaturesGLCM& t);           // 1x7
cv::Mat lbpAVector(const EstadisticosHistograma& l);     // 1x4

// GlobalFeatureVector -> fila 1x(3+K): [simetria,limpieza,enfoque,
//   proporcion(orden[0]),...,proporcion(orden[K-1])]
// 'ordenRegiones' fija el orden de las proporciones (normalmente el
// orden de la PlantillaRegiones del chef) — si una region no aparece
// en global.proporciones, se rellena con 0.0 en vez de fallar, para
// que la matriz nunca cambie de ancho entre alumnos.
cv::Mat globalAVector(const GlobalFeatureVector& g, const std::vector<std::string>& ordenRegiones);

} // namespace evaluador
