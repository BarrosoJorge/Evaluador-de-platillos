#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"
#include "FeatureVector.hpp"
#include "EstimadorLedoitWolf.hpp"
#include "CalculadorMahalanobis.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace evaluador;
namespace fs = std::filesystem;

struct FirmaPlatillo {
    std::string nombreArchivo;
    bool esChef;
    std::vector<RegionFeatureVector> regiones;
    GlobalFeatureVector global;
};

/// Calcula distancias de Mahalanobis para una dimension y las escribe en CSV.
void procesarDimension(const std::string& nombreDimension, const cv::Mat& xChef, const cv::Mat& XAlumnos,
                       const std::vector<std::string>& nombresAlumnos, std::ofstream& csvOut) {
    if (XAlumnos.rows == 0 || xChef.empty()) return;
    
    // Estimación de la matriz de covarianza de la población con Ledoit-Wolf
    ResultadoLedoitWolf lw = estimarCovarianzaLedoitWolf(XAlumnos);
    
    // Distancia de Mahalanobis de cada estudiante respecto al Chef
    for (int i = 0; i < XAlumnos.rows; ++i) {
        double d = calcularDistanciaMahalanobis(XAlumnos.row(i), xChef, lw.covarianza);
        csvOut << nombresAlumnos[i] << ";" << nombreDimension << ";" << d << "\n";
    }
}

/// Ejecuta la comparacion estadistica de una sesion completa.
/// Devuelve 0 si el archivo de distancias se genera correctamente.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: ./comparador <Clave_Sesion>" << std::endl;
        return 1;
    }
    std::string claveDish = argv[1];
    
    fs::path raiz = fs::current_path();
    PathManager rutas(raiz);
    
    // Lee las dos ramas de salida de la etapa 3.
    fs::path carpetaGlobal   = rutas.featuresEstadisticosGlobal() / claveDish;
    fs::path carpetaPorRegion = rutas.featuresEstadisticosPorRegion() / claveDish;
    
    fs::path dirOut = rutas.comparacionDistancias();
    std::error_code ec;
    fs::create_directories(dirOut, ec);
    fs::path rutaOut = dirOut / (claveDish + ".csv");

    if (!fs::exists(carpetaGlobal)) {
        Logger::instance().error("Comparador", "No existe la carpeta global para la sesion: " + claveDish);
        return 1;
    }

    std::vector<FirmaPlatillo> todos;
    for (const auto& entry : fs::directory_iterator(carpetaGlobal)) {
        if (entry.path().extension() != ".csv") continue;
        
        std::string nombreArchivo = entry.path().filename().string();
        fs::path rutaRegionFile = carpetaPorRegion / nombreArchivo;
        
        auto globOpt = cargarFeaturesGlobal(entry.path());
        auto regOpt  = cargarFeaturesPorRegion(rutaRegionFile);
        
        if (globOpt) {
            FirmaPlatillo f;
            f.nombreArchivo = entry.path().stem().string();
            std::string nLower = f.nombreArchivo;
            std::transform(nLower.begin(), nLower.end(), nLower.begin(), ::tolower);
            f.esChef = (nLower.find("chef") != std::string::npos);
            f.global = *globOpt;
            if (regOpt) {
                f.regiones = *regOpt;
            }
            todos.push_back(f);
        }
    }

    std::vector<FirmaPlatillo> chefs, alumnos;
    for (auto& f : todos) (f.esChef ? chefs : alumnos).push_back(f);

    if (chefs.empty() || alumnos.empty()) {
        std::cerr << "Faltan datos en la sesion (Chef: " << chefs.size() << ", Alumnos: " << alumnos.size() << ")" << std::endl;
        return 1;
    }

    FirmaPlatillo chef = chefs[0];
    std::ofstream csvDist(rutaOut);
    csvDist << "alumno;dimension;distancia\n";
    
    std::vector<std::string> nombresAlumnos;
    for (const auto& al : alumnos) {
        nombresAlumnos.push_back(al.nombreArchivo);
    }

    // Procesa el bloque de variables globales.
    cv::Mat xChefG(1, 4, CV_64F);
    xChefG.at<double>(0, 0) = chef.global.simetria;
    xChefG.at<double>(0, 1) = chef.global.limpieza;
    xChefG.at<double>(0, 2) = chef.global.enfoque;
    xChefG.at<double>(0, 3) = chef.global.volumenGeneral;

    cv::Mat XAlumG(alumnos.size(), 4, CV_64F);
    for (size_t i = 0; i < alumnos.size(); ++i) {
        XAlumG.at<double>(i, 0) = alumnos[i].global.simetria;
        XAlumG.at<double>(i, 1) = alumnos[i].global.limpieza;
        XAlumG.at<double>(i, 2) = alumnos[i].global.enfoque;
        XAlumG.at<double>(i, 3) = alumnos[i].global.volumenGeneral;
    }
    procesarDimension("global", xChefG, XAlumG, nombresAlumnos, csvDist);

    // Procesa cada zona espacial y sus subdescriptores.
    for (const auto& regChef : chef.regiones) {
        std::string nombreZona = regChef.nombreRegion; // ej. "zona_0_0"

        cv::Mat XAlumColor(alumnos.size(), 6, CV_64F, cv::Scalar(0.0));
        cv::Mat XAlumTex(alumnos.size(), 7, CV_64F, cv::Scalar(0.0));
        cv::Mat XAlumLbp(alumnos.size(), 4, CV_64F, cv::Scalar(0.0));

        cv::Mat xChefColor = colorAVector(regChef.color);
        cv::Mat xChefTex   = texturaAVector(regChef.textura);
        cv::Mat xChefLbp   = lbpAVector(regChef.lbp);

        for (size_t i = 0; i < alumnos.size(); ++i) {
            for (const auto& regAlum : alumnos[i].regiones) {
                if (regAlum.nombreRegion == nombreZona) {
                    colorAVector(regAlum.color).copyTo(XAlumColor.row(i));
                    texturaAVector(regAlum.textura).copyTo(XAlumTex.row(i));
                    lbpAVector(regAlum.lbp).copyTo(XAlumLbp.row(i));
                    break;
                }
            }
        }

        // Evalua cada subdescriptor por separado.
        procesarDimension(nombreZona + "_color",   xChefColor, XAlumColor, nombresAlumnos, csvDist);
        procesarDimension(nombreZona + "_textura", xChefTex,   XAlumTex,   nombresAlumnos, csvDist);
        procesarDimension(nombreZona + "_lbp",     xChefLbp,   XAlumLbp,   nombresAlumnos, csvDist);
    }

    Logger::instance().info("Comparador", "Comparacion estadistica completada para sesion: " + claveDish);
    return 0;
}