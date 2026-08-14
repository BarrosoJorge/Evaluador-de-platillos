/* Etapa 5: fuzzifica distancias por dimension para una sesion.
 * Genera anclas percentiles y pertenencias por alumno.
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

/// Lee el archivo de distancias y devuelve filas validas.
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

/// Ejecuta la fuzzificacion por dimension para una sesion.
/// Devuelve 0 si los CSV de salida se generan correctamente.
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

    // Agrupa distancias por dimension para calcular anclas locales.
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

    // Guarda anclas percentiles.
    fs::path rutaPercentiles = rutas.fuzzyPercentiles() / (claveDish + ".csv");
    std::error_code ec;
    fs::create_directories(rutaPercentiles.parent_path(), ec);
    std::ofstream csvPercentiles(rutaPercentiles);
    csvPercentiles << "dimension;p33;p50;p66;n_muestras\n";
    for (const auto& [dimension, anclas] : anclasPorDimension) {
        csvPercentiles << dimension << ";" << anclas.p33 << ";" << anclas.p50 << ";" << anclas.p66
                        << ";" << distanciasPorDimension[dimension].size() << "\n";
    }

    // Fuzzifica cada fila y guarda pertenencias.
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

        // Marca casos dominados por "muy_distinto" para revision manual.
        if (grados.muyDistinto > 0.66) {
            ++revisionesSugeridas;
        }
    }

    Logger::instance().info("FuzzificadorSesion",
        "Completado: " + std::to_string(filas.size()) + " fila(s) fuzzificadas, " +
        std::to_string(revisionesSugeridas) + " con muy_distinto dominante.");

    return 0;
}
