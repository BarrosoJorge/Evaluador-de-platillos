/* Etapa 6: aplica reglas difusas y defuzzificacion por dimension.
 * Genera scores elementales por alumno para una sesion.
 */

#include "Logger.hpp"
#include "PathManager.hpp"
#include "MotorReglas.hpp"
#include "Defuzzificador.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace evaluador;

struct FilaPertenencia {
    std::string alumno;
    std::string dimension;
    double distancia = 0.0;
    GradosPertenencia grados;
};

/// Lee pertenencias fuzzificadas y devuelve filas validas.
std::vector<FilaPertenencia> leerPertenencias(const fs::path& ruta) {
    std::vector<FilaPertenencia> filas;
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        Logger::instance().error("SistemaDifusoSesion", "No se pudo abrir: " + ruta.string());
        return filas;
    }

    std::string linea;
    bool primeraLinea = true;
    while (std::getline(archivo, linea)) {
        if (primeraLinea) { primeraLinea = false; continue; }

        std::vector<std::string> campos;
        std::istringstream iss(linea);
        std::string campo;
        while (std::getline(iss, campo, ';')) campos.push_back(campo);

        if (campos.size() != 6) continue;

        try {
            FilaPertenencia f;
            f.alumno = campos[0];
            f.dimension = campos[1];
            f.distancia = std::stod(campos[2]);
            f.grados.similar = std::stod(campos[3]);
            f.grados.algoDistinto = std::stod(campos[4]);
            f.grados.muyDistinto = std::stod(campos[5]);
            filas.push_back(f);
        } catch (const std::exception& e) {
            Logger::instance().warn("SistemaDifusoSesion", "Linea invalida ignorada: " + linea);
        }
    }

    return filas;
}

/// Ejecuta inferencia difusa y defuzzificacion para una sesion.
/// Devuelve 0 si el archivo de scores se genera correctamente.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <Ciudad_Platillo_Angulo>" << std::endl;
        return 1;
    }

    std::string claveDish = argv[1];
    PathManager rutas(".");

    fs::path rutaPertenencias = rutas.fuzzyPertenencias() / (claveDish + ".csv");
    std::vector<FilaPertenencia> filas = leerPertenencias(rutaPertenencias);

    if (filas.empty()) {
        Logger::instance().error("SistemaDifusoSesion",
            "No hay pertenencias para '" + claveDish + "'. Corre FuzzificadorSesion.cpp primero.");
        return 1;
    }

    TablaReglas tabla;

    fs::path rutaSalida = rutas.scoresElementales() / (claveDish + ".csv");
    std::error_code ec;
    fs::create_directories(rutaSalida.parent_path(), ec);
    std::ofstream csvSalida(rutaSalida);
    csvSalida << "alumno;dimension;distancia;score\n";

    for (const auto& f : filas) {
        auto disparos = inferir(f.grados, tabla);
        double score = defuzzificar(disparos);

        csvSalida << f.alumno << ";" << f.dimension << ";" << f.distancia << ";" << score << "\n";
    }

    Logger::instance().info("SistemaDifusoSesion",
        "Completado: " + std::to_string(filas.size()) + " score(s) elementales calculados. Salida: " + rutaSalida.string());

    return 0;
}
