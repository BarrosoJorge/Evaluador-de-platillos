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

using namespace evaluador;
namespace fs = std::filesystem;

struct FirmaPlatillo {
    std::string nombreArchivo;
    bool esChef;
    std::vector<RegionFeatureVector> regiones;
    GlobalFeatureVector global;
};

void procesarDimension(const std::string& nombreDimension, const cv::Mat& xChef, const cv::Mat& XAlumnos,
                       const std::vector<std::string>& nombresAlumnos, std::ofstream& csvOut) {
    if (XAlumnos.rows == 0) return;
    ResultadoLedoitWolf lw = estimarCovarianzaLedoitWolf(XAlumnos);
    for (int i = 0; i < XAlumnos.rows; ++i) {
        double d = calcularDistanciaMahalanobis(XAlumnos.row(i), xChef, lw.covarianza);
        csvOut << nombresAlumnos[i] << ";" << nombreDimension << ";" << d << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string claveDish = argv[1];
    
    fs::path raiz = fs::current_path();
    PathManager rutas(raiz);
    
    fs::path carpetaSesion = rutas.featuresVectorFinal() / claveDish;
    fs::path dirOut = rutas.comparacionDistancias();
    
    std::error_code ec;
    fs::create_directories(dirOut, ec);
    
    fs::path rutaOut = dirOut / (claveDish + ".csv");

    std::vector<FirmaPlatillo> todos;
    for (const auto& entry : fs::directory_iterator(carpetaSesion)) {
        if (entry.path().extension() != ".csv") continue;
        
        auto regOpt = cargarFeaturesPorRegion(entry.path());
        auto globOpt = cargarFeaturesGlobal(entry.path());
        
        if (regOpt && globOpt) {
            FirmaPlatillo f;
            f.nombreArchivo = entry.path().stem().string();
            std::string nLower = f.nombreArchivo;
            std::transform(nLower.begin(), nLower.end(), nLower.begin(), ::tolower);
            f.esChef = (nLower.find("chef") != std::string::npos);
            f.regiones = *regOpt;
            f.global = *globOpt;
            todos.push_back(f);
        }
    }

    std::vector<FirmaPlatillo> chefs, alumnos;
    for(auto& f : todos) (f.esChef ? chefs : alumnos).push_back(f);

    if (chefs.empty() || alumnos.empty()) {
        std::cerr << "Faltan datos (Chef: " << chefs.size() << ", Alumnos: " << alumnos.size() << ")" << std::endl;
        return 1;
    }

    FirmaPlatillo chef = chefs[0];
    std::ofstream csvDist(rutaOut);
    csvDist << "alumno;dimension;distancia\n";
    
    // 1. PROCESAR VARIABLES GLOBALES
    cv::Mat xChefG(1, 4, CV_64F);
    xChefG.at<double>(0, 0) = chef.global.simetria;
    xChefG.at<double>(0, 1) = chef.global.limpieza;
    xChefG.at<double>(0, 2) = chef.global.enfoque;
    xChefG.at<double>(0, 3) = chef.global.volumenGeneral;

    cv::Mat XAlumG(alumnos.size(), 4, CV_64F);
    std::vector<std::string> nombres;
    for(size_t i = 0; i < alumnos.size(); ++i) {
        XAlumG.at<double>(i, 0) = alumnos[i].global.simetria;
        XAlumG.at<double>(i, 1) = alumnos[i].global.limpieza;
        XAlumG.at<double>(i, 2) = alumnos[i].global.enfoque;
        XAlumG.at<double>(i, 3) = alumnos[i].global.volumenGeneral;
        nombres.push_back(alumnos[i].nombreArchivo);
    }
    procesarDimension("global", xChefG, XAlumG, nombres, csvDist);

    return 0;
}