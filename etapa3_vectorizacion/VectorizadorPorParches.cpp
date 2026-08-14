#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"
#include "ColorEstadisticas.hpp"
#include "TexturaGLCM.hpp"
#include "TexturaLBP.hpp"
#include "ReductorEstadisticos.hpp"
#include "FeatureVector.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

using namespace evaluador;
namespace fs = std::filesystem;

/// Calcula descriptores por zona espacial usando mascaras de grid.
/// Devuelve 0 si los vectores por region se guardan correctamente.
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: ./vectorizador_parches <ruta_imagen_pre> <ruta_mascara_global>\n";
        return 1;
    }

    fs::path rutaImagen = argv[1];
    fs::path rutaMascaraGlobal = argv[2]; 

    PathManager rutas(".");
    VideoMetadata metadata = parsearNombreArchivo(rutaImagen.filename().string());
    
    if (metadata.vacio()) return 1;
    std::string claveDish = metadata.ciudad + "_" + metadata.platillo + "_" + metadata.angulo;

    cv::Mat imagenBGR = cv::imread(rutaImagen.string(), cv::IMREAD_COLOR);
    cv::Mat mascaraGlobal = cv::imread(rutaMascaraGlobal.string(), cv::IMREAD_GRAYSCALE);
    
    if (imagenBGR.empty() || mascaraGlobal.empty()) {
        Logger::instance().error("VectorizadorParches", "Error cargando imagen o máscara global");
        return 1;
    }

    cv::Mat imagenHSV, imagenGris;
    cv::cvtColor(imagenBGR, imagenHSV, cv::COLOR_BGR2HSV);
    cv::cvtColor(imagenBGR, imagenGris, cv::COLOR_BGR2GRAY);

    std::vector<RegionFeatureVector> vectores;
    
    // Usa la carpeta local Grids generada en la etapa 2.
    fs::path dirGrids = rutaImagen.parent_path() / "Grids";

    int gridSize = 4;
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            std::string nombreZona = "zona_" + std::to_string(i) + "_" + std::to_string(j);
            fs::path rutaMascaraGrid = dirGrids / (nombreZona + ".png");

            cv::Mat mascaraGrid = cv::imread(rutaMascaraGrid.string(), cv::IMREAD_GRAYSCALE);
            if (mascaraGrid.empty()) continue;

            // Interseccion entre zona geometrica y alimento real.
            cv::Mat mascaraComidaEnZona;
            cv::bitwise_and(mascaraGrid, mascaraGlobal, mascaraComidaEnZona);

            // Ignora zonas sin contenido suficiente de alimento.
            if (cv::countNonZero(mascaraComidaEnZona) < 50) continue;

            RegionFeatureVector v;
            v.nombreRegion = nombreZona;
            v.color = calcularEstadisticosColor(imagenHSV, mascaraComidaEnZona);
            v.textura = calcularGLCM(imagenGris, mascaraComidaEnZona, 16, 1);
            std::vector<double> histLbp = histogramaLBP(imagenGris, mascaraComidaEnZona);
            v.lbp = reducirHistograma(histLbp);
            
            vectores.push_back(v);
        }
    }

    // Guarda salida con rutas del proyecto.
    fs::path rutaSalida = rutas.featuresEstadisticosPorRegion() / claveDish / (metadata.formatear() + ".csv");
    guardarFeaturesPorRegion(vectores, rutaSalida);
    
    Logger::instance().info("VectorizadorParches", "Vectores guardados para " + metadata.formatear());
    return 0;
}