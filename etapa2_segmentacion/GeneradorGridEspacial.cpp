/* Etapa 2: genera mascara global y zonas espaciales 4x4.
 * Usa la imagen preprocesada para construir mascaras por region.
 */

#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>

using namespace evaluador;
namespace fs = std::filesystem;

/// Genera mascara global y mascaras por zona para una imagen.
/// Devuelve 0 si todas las salidas se crean correctamente.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: ./generador_grid <ruta_imagen_preprocesada>\n";
        return 1;
    }

    fs::path rutaImagen = argv[1];
    int gridSize = 4; // Cuadricula 4x4 (16 zonas).

    // Carga la imagen preprocesada.
    cv::Mat imagen = cv::imread(rutaImagen.string(), cv::IMREAD_COLOR);
    if (imagen.empty()) {
        Logger::instance().error("GeneradorGrid", "No se pudo cargar la imagen: " + rutaImagen.string());
        return 1;
    }

    // Extrae mascara global de alimento sobre fondo negro.
    cv::Mat gris, mascaraGlobal;
    cv::cvtColor(imagen, gris, cv::COLOR_BGR2GRAY);
    cv::threshold(gris, mascaraGlobal, 1, 255, cv::THRESH_BINARY);

    // Crea carpeta local de salida.
    fs::path dirBase = rutaImagen.parent_path();
    fs::path dirGrids = dirBase / "Grids";
    std::error_code ec;
    fs::create_directories(dirGrids, ec);

    // Guarda mascara global para la etapa 3.
    fs::path rutaMascaraGlobal = dirGrids / "mascara_global.png";
    cv::imwrite(rutaMascaraGlobal.string(), mascaraGlobal);

    // Divide la imagen en zonas medibles.
    int width = imagen.cols;
    int height = imagen.rows;
    int stepX = width / gridSize;
    int stepY = height / gridSize;

    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            cv::Mat mascaraParche = cv::Mat::zeros(imagen.size(), CV_8UC1);
            
            int x = j * stepX;
            int y = i * stepY;
            int w = (j == gridSize - 1) ? (width - x) : stepX;
            int h = (i == gridSize - 1) ? (height - y) : stepY;
            
            cv::Rect roi(x, y, w, h);
            cv::rectangle(mascaraParche, roi, cv::Scalar(255), cv::FILLED);

            // Interseca la zona con la mascara global para conservar solo alimento.
            cv::bitwise_and(mascaraParche, mascaraGlobal, mascaraParche);

            std::string nombreMascara = "zona_" + std::to_string(i) + "_" + std::to_string(j) + ".png";
            cv::imwrite((dirGrids / nombreMascara).string(), mascaraParche);
        }
    }

    Logger::instance().info("GeneradorGrid", "Grid y mascara global creados en: " + dirGrids.string());
    return 0;
}