/* GeneradorGridEspacial.cpp
 * Etapa 2 - Segmentación Agnóstica (Grid Espacial)
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: ./generador_grid <ruta_imagen_preprocesada>\n";
        return 1;
    }

    fs::path rutaImagen = argv[1];
    int gridSize = 4; // Cuadrícula por defecto de 4x4 (16 zonas)

    // Cargar la imagen maestra preprocesada
    cv::Mat imagen = cv::imread(rutaImagen.string(), cv::IMREAD_COLOR);
    if (imagen.empty()) {
        Logger::instance().error("GeneradorGrid", "No se pudo cargar la imagen: " + rutaImagen.string());
        return 1;
    }

    // 1. EXTRAER MÁSCARA GLOBAL (Materia comestible vs plato/fondo vacío)
    // Como la imagen preprocesada ya tiene el fondo negro puro (0,0,0), todo lo que tenga valor > 1 es comida
    cv::Mat gris, mascaraGlobal;
    cv::cvtColor(imagen, gris, cv::COLOR_BGR2GRAY);
    cv::threshold(gris, mascaraGlobal, 1, 255, cv::THRESH_BINARY);

    // 2. CREAR CARPETA DE SALIDA LOCAL "Grids"
    fs::path dirBase = rutaImagen.parent_path();
    fs::path dirGrids = dirBase / "Grids";
    std::error_code ec;
    fs::create_directories(dirGrids, ec);

    // Guardar la máscara global alineada de 224x224 para la Etapa 3
    fs::path rutaMascaraGlobal = dirGrids / "mascara_global.png";
    cv::imwrite(rutaMascaraGlobal.string(), mascaraGlobal);

    // 3. SUBDIVIDIR EN ZONAS MEDIBLES (Grid Espacial)
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

            // Intersectar la zona geométrica con la máscara global 
            // Esto asegura que solo mediremos píxeles que contengan comida real en esa zona
            cv::bitwise_and(mascaraParche, mascaraGlobal, mascaraParche);

            std::string nombreMascara = "zona_" + std::to_string(i) + "_" + std::to_string(j) + ".png";
            cv::imwrite((dirGrids / nombreMascara).string(), mascaraParche);
        }
    }

    Logger::instance().info("GeneradorGrid", "Grid y Mascara Global creados en: " + dirGrids.string());
    return 0;
}