#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"
#include "FeatureVector.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace evaluador;
namespace fs = std::filesystem;

/// Calcula descriptores globales de una imagen y su mascara.
/// Devuelve 0 si el vector global se guarda correctamente.
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: ./vectorizador_global <ruta_imagen_pre> <ruta_mascara_global>\n";
        return 1;
    }

    fs::path rutaImagen = argv[1];
    fs::path rutaMascaraGlobal = argv[2]; 

    PathManager rutas(".");
    VideoMetadata metadata = parsearNombreArchivo(rutaImagen.filename().string());
    if(metadata.vacio()) return 1;
    std::string claveDish = metadata.ciudad + "_" + metadata.platillo + "_" + metadata.angulo;

    cv::Mat mascaraGlobal = cv::imread(rutaMascaraGlobal.string(), cv::IMREAD_GRAYSCALE);
    if (mascaraGlobal.empty()) {
        Logger::instance().error("VectorizadorGlobal", "Error cargando máscara global.");
        return 1;
    }

    GlobalFeatureVector global;

    // Simetria y equilibrio por centroide de la mascara.
    cv::Moments m = cv::moments(mascaraGlobal, true);
    if (m.m00 > 0) {
        double cx = m.m10 / m.m00;
        double cy = m.m01 / m.m00;
        double dx = cx - (mascaraGlobal.cols / 2.0);
        double dy = cy - (mascaraGlobal.rows / 2.0);
        global.simetria = std::sqrt(dx*dx + dy*dy);
    } else {
        global.simetria = 0.0;
    }

    // Limpieza por conteo de manchas pequenas.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mascaraGlobal, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int manchasPequenas = 0;
    double areaTotalComida = cv::countNonZero(mascaraGlobal);
    
    for (const auto& contorno : contours) {
        double areaContorno = cv::contourArea(contorno);
        if (areaContorno > 0 && areaContorno < (areaTotalComida * 0.01)) {
            manchasPequenas++;
        }
    }
    global.limpieza = static_cast<double>(manchasPequenas);

    // Enfoque por varianza del Laplaciano dentro de la mascara.
    cv::Mat imagenGris;
    cv::Mat imagenBGR = cv::imread(rutaImagen.string(), cv::IMREAD_COLOR);
    cv::cvtColor(imagenBGR, imagenGris, cv::COLOR_BGR2GRAY);
    cv::Mat laplaciano;
    cv::Laplacian(imagenGris, laplaciano, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplaciano, mean, stddev, mascaraGlobal);
    global.enfoque = stddev.val[0] * stddev.val[0];

    // Proporcion de area ocupada como aproximacion global.
    global.volumenGeneral = areaTotalComida / (mascaraGlobal.cols * mascaraGlobal.rows);

    // Guarda salida con rutas del proyecto.
    fs::path rutaSalida = rutas.featuresEstadisticosGlobal() / claveDish / (metadata.formatear() + ".csv");
    guardarFeaturesGlobal(global, rutaSalida);
    
    Logger::instance().info("VectorizadorGlobal", "Features globales guardados para " + metadata.formatear());
    return 0;
}