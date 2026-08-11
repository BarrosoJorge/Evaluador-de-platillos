/*  Ensamblador.cpp
 *
 *  Etapa 3 del pipeline — Ensamblado del vector final.
 *
 *  Junta lo que produjeron VectorizadorPorRegion y VectorizadorGlobal
 *  (las dos ramas en paralelo) para UNA imagen, en un solo archivo. No
 *  vuelve a calcular nada — solo lee los dos CSV ya generados y los
 *  concatena en un tercero. Este archivo final es el que consume la
 *  etapa 4 (comparacion estadistica / Mahalanobis).
 *
 *  Entrada:  metadatos de una imagen (misma ruta *_pre.png que se le
 *            paso antes a VectorizadorPorRegion/VectorizadorGlobal)
 *  Salida:   Data/Features/VectorFinal/<claveDish>/<nombreImagen>.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -I../scripts_genericos/include \
 *          ../scripts_genericos/src/Logger.cpp \
 *          ../scripts_genericos/src/PathManager.cpp \
 *          ../scripts_genericos/src/VideoMetadata.cpp \
 *          ../scripts_genericos/src/ColorEstadisticas.cpp \
 *          ../scripts_genericos/src/TexturaGLCM.cpp \
 *          ../scripts_genericos/src/ReductorEstadisticos.cpp \
 *          ../scripts_genericos/src/FeatureVector.cpp \
 *          Ensamblador.cpp -o ensamblador \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./ensamblador <ruta_imagen_preprocesada>
 */

#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"
#include "FeatureVector.hpp"

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace evaluador;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_imagen_preprocesada>" << std::endl;
        return 1;
    }

    fs::path rutaImagen = argv[1];
    PathManager rutas(".");

    std::string stem = rutaImagen.stem().string();
    if (stem.size() >= 4 && stem.substr(stem.size() - 4) == "_pre") {
        stem = stem.substr(0, stem.size() - 4);
    }
    VideoMetadata metadata = parsearNombreArchivo(stem);
    if (metadata.vacio()) {
        Logger::instance().error("Ensamblador", "No se pudo extraer metadato de: " + rutaImagen.string());
        return 1;
    }

    std::string claveDish = metadata.ciudad + "_" + metadata.platillo + "_" + metadata.angulo;
    std::string nombreImagen = metadata.formatear();

    fs::path rutaPorRegion = rutas.featuresEstadisticosPorRegion() / claveDish / (nombreImagen + ".csv");
    fs::path rutaGlobal = rutas.featuresEstadisticosGlobal() / claveDish / (nombreImagen + ".csv");


    std::cout << "Ruta de features por region: " << rutaPorRegion << std::endl;
    std::cout << "Ruta de features globales: " << rutaGlobal << std::endl;

    auto porRegion = cargarFeaturesPorRegion(rutaPorRegion);
    auto global = cargarFeaturesGlobal(rutaGlobal);

    if (!porRegion.has_value()) {
        Logger::instance().error("Ensamblador",
            "Falta el archivo de features por region: " + rutaPorRegion.string() +
            ". Corre VectorizadorPorRegion.cpp primero.");
        return 1;
    }
    if (!global.has_value()) {
        Logger::instance().error("Ensamblador",
            "Falta el archivo de features globales: " + rutaGlobal.string() +
            ". Corre VectorizadorGlobal.cpp primero.");
        return 1;
    }

    fs::path rutaSalida = rutas.featuresVectorFinal() / claveDish / (nombreImagen + ".csv");
    std::error_code ec;
    fs::create_directories(rutaSalida.parent_path(), ec);

    std::ofstream salida(rutaSalida);
    if (!salida.is_open()) {
        Logger::instance().error("Ensamblador", "No se pudo escribir: " + rutaSalida.string());
        return 1;
    }

    salida << "# vector_final\n";
    salida << "# imagen=" << nombreImagen << "\n";
    salida << "# dish=" << claveDish << "\n\n";

    // Copiar el contenido crudo de ambos archivos fuente, cada uno con
    // su propia seccion — no se re-serializa a mano para no arriesgar
    // una inconsistencia con el formato que ya escriben
    // guardarFeaturesPorRegion()/guardarFeaturesGlobal().
    salida << "## por_region\n";
    std::ifstream origenRegion(rutaPorRegion);
    salida << origenRegion.rdbuf();

    salida << "\n## global\n";
    std::ifstream origenGlobal(rutaGlobal);
    salida << origenGlobal.rdbuf();

    Logger::instance().info("Ensamblador", "Vector final ensamblado: " + rutaSalida.string());

    return 0;
}

// Compilación: ver bloque al inicio del archivo.
// Uso:
//   ./ensamblador <ruta_imagen_preprocesada>
