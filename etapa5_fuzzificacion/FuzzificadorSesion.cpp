/*  FuzzificadorSesion.cpp
 *
 *  Etapa 5 del pipeline — Fuzzificación.
 *
 *  Lee el CSV que dejo ComparadorSesion.cpp (etapa 4) — una fila por
 *  alumno por dimension (region o "global") con su distancia de
 *  Mahalanobis — y para cada dimension:
 *    1. CalculadorPercentiles: p33/p50/p66 sobre las distancias de
 *       TODOS los alumnos de esa dimension en esta sesion
 *    2. Fuzzificador: convierte cada distancia individual en 3 grados
 *       de pertenencia (similar / algo_distinto / muy_distinto)
 *
 *  Importante: las anclas se calculan POR DIMENSION Y POR SESION, no
 *  son valores globales del proyecto. Esto es a proposito — la
 *  distribucion de distancias de "textura de la salsa" no tiene por
 *  que parecerse a la de "simetria global", y una sesion con un
 *  platillo dificil puede tener distancias mas grandes en general que
 *  una sesion con un platillo simple. Fuzzificar con las anclas de la
 *  MISMA sesion es lo que evita tener que elegir umbrales fijos a mano.
 *
 *  Entrada:  Data/Comparacion/Distancias/<claveDish>.csv
 *  Salida:   Data/Fuzzy/Percentiles/<claveDish>.csv   (anclas por dimension)
 *            Data/Fuzzy/Pertenencias/<claveDish>.csv  (grados por alumno x dimension)
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -I../scripts_genericos/include \
 *          ../scripts_genericos/src/Logger.cpp \
 *          ../scripts_genericos/src/PathManager.cpp \
 *          ../scripts_genericos/src/CalculadorPercentiles.cpp \
 *          ../scripts_genericos/src/Fuzzificador.cpp \
 *          FuzzificadorSesion.cpp -o fuzzificador_sesion
 *
 *  Uso:
 *      ./fuzzificador_sesion <Ciudad_Platillo_Angulo>
 */

#include "Logger.hpp"
#include "PathManager.hpp"
#include "CalculadorPercentiles.hpp"
#include "Fuzzificador.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace evaluador;

struct FilaDistancia {
    std::string alumno;
    std::string dimension;
    double distancia = 0.0;
};

// Lee el CSV de distancias de la etapa 4 (separador ';', con header).
std::vector<FilaDistancia> leerDistancias(const fs::path& ruta) {
    std::vector<FilaDistancia> filas;
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        Logger::instance().error("FuzzificadorSesion", "No se pudo abrir: " + ruta.string());
        return filas;
    }

    std::string linea;
    bool primeraLinea = true;
    while (std::getline(archivo, linea)) {
        if (primeraLinea) { primeraLinea = false; continue; } // saltar header

        std::vector<std::string> campos;
        std::istringstream iss(linea);
        std::string campo;
        while (std::getline(iss, campo, ';')) campos.push_back(campo);

        if (campos.size() < 3) continue;

        try {
            FilaDistancia f;
            f.alumno = campos[0];
            f.dimension = campos[1];
            f.distancia = std::stod(campos[2]);
            filas.push_back(f);
        } catch (const std::exception& e) {
            Logger::instance().warn("FuzzificadorSesion", "Linea invalida ignorada: " + linea);
        }
    }

    return filas;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <Ciudad_Platillo_Angulo>" << std::endl;
        return 1;
    }

    std::string claveDish = argv[1];
    PathManager rutas(".");

    fs::path rutaDistancias = rutas.comparacionDistancias() / (claveDish + ".csv");
    std::vector<FilaDistancia> filas = leerDistancias(rutaDistancias);

    if (filas.empty()) {
        Logger::instance().error("FuzzificadorSesion", "No hay distancias para '" + claveDish + "'. Corre ComparadorSesion.cpp primero.");
        return 1;
    }

    // Agrupar distancias por dimension, para calcular las anclas de cada una
    std::map<std::string, std::vector<double>> distanciasPorDimension;
    for (const auto& f : filas) {
        distanciasPorDimension[f.dimension].push_back(f.distancia);
    }

    std::map<std::string, AnclasPercentiles> anclasPorDimension;
    for (const auto& [dimension, distancias] : distanciasPorDimension) {
        anclasPorDimension[dimension] = calcularAnclas(distancias);
        Logger::instance().info("FuzzificadorSesion",
            "Dimension '" + dimension + "': p33=" + std::to_string(anclasPorDimension[dimension].p33) +
            " p50=" + std::to_string(anclasPorDimension[dimension].p50) +
            " p66=" + std::to_string(anclasPorDimension[dimension].p66));
    }

    // --- Guardar anclas ---
    fs::path rutaPercentiles = rutas.fuzzyPercentiles() / (claveDish + ".csv");
    std::error_code ec;
    fs::create_directories(rutaPercentiles.parent_path(), ec);
    std::ofstream csvPercentiles(rutaPercentiles);
    csvPercentiles << "dimension;p33;p50;p66;n_muestras\n";
    for (const auto& [dimension, anclas] : anclasPorDimension) {
        csvPercentiles << dimension << ";" << anclas.p33 << ";" << anclas.p50 << ";" << anclas.p66
                        << ";" << distanciasPorDimension[dimension].size() << "\n";
    }

    // --- Fuzzificar cada fila y guardar pertenencias ---
    fs::path rutaPertenencias = rutas.fuzzyPertenencias() / (claveDish + ".csv");
    fs::create_directories(rutaPertenencias.parent_path(), ec);
    std::ofstream csvPertenencias(rutaPertenencias);
    csvPertenencias << "alumno;dimension;distancia;similar;algo_distinto;muy_distinto\n";

    int revisionesSugeridas = 0;
    for (const auto& f : filas) {
        const AnclasPercentiles& anclas = anclasPorDimension[f.dimension];
        GradosPertenencia grados = fuzzificar(f.distancia, anclas);

        csvPertenencias << f.alumno << ";" << f.dimension << ";" << f.distancia << ";"
                         << grados.similar << ";" << grados.algoDistinto << ";" << grados.muyDistinto << "\n";

        // Señal util para priorizar revision manual: si "muy_distinto"
        // domina claramente, es un caso que vale la pena que un humano
        // revise antes de confiar en la rubrica final.
        if (grados.muyDistinto > 0.66) {
            ++revisionesSugeridas;
        }
    }

    Logger::instance().info("FuzzificadorSesion",
        "Completado: " + std::to_string(filas.size()) + " fila(s) fuzzificadas, " +
        std::to_string(revisionesSugeridas) + " con muy_distinto dominante.");

    return 0;
}

// Compilación: ver bloque al inicio del archivo.
// Uso:
//   ./fuzzificador_sesion <Ciudad_Platillo_Angulo>
