/*  SistemaDifusoSesion.cpp
 *
 *  Etapa 6 del pipeline — Sistema difuso (reglas + defuzzificación).
 *
 *  Lee el CSV que dejo FuzzificadorSesion.cpp (etapa 5) — grados de
 *  pertenencia por alumno x dimension — y para cada fila:
 *    1. MotorReglas: convierte los 3 grados en pares (fuerza, valor)
 *       usando la TablaReglas por defecto (ver MotorReglas.hpp)
 *    2. Defuzzificador: promedio ponderado -> un score 1-10
 *
 *  Esta es la salida MAS FINA del pipeline: un numero por cada
 *  dimension elemental (ej. "carne_textura", "carne_color",
 *  "salsa_lbp", "simetria", "enfoque"...) por alumno. La etapa 7 es
 *  quien agrega estos numeros en las 6 categorias de la rubrica final
 *  que ve el alumno — este script NO hace esa agregacion.
 *
 *  AjustadorReglas (pendiente, no implementado aqui): cuando existan
 *  calificaciones reales del chef por dimension, se podria ajustar
 *  TablaReglas via ANFIS o un algoritmo genetico en vez de usar los
 *  valores por defecto. Como TablaReglas ya es un parametro y no una
 *  constante fija, ese futuro ajustador seria un programa que
 *  produce una TablaReglas distinta — no requeriria tocar MotorReglas
 *  ni Defuzzificador.
 *
 *  Entrada:  Data/Fuzzy/Pertenencias/<claveDish>.csv
 *  Salida:   Data/SistemaDifuso/ScoresElementales/<claveDish>.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -I../scripts_genericos/include \
 *          ../scripts_genericos/src/Logger.cpp \
 *          ../scripts_genericos/src/PathManager.cpp \
 *          ../scripts_genericos/src/MotorReglas.cpp \
 *          ../scripts_genericos/src/Defuzzificador.cpp \
 *          SistemaDifusoSesion.cpp -o sistema_difuso_sesion
 *
 *  Uso:
 *      ./sistema_difuso_sesion <Ciudad_Platillo_Angulo>
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

    TablaReglas tabla; // valores por defecto — ver nota sobre AjustadorReglas

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

// Compilación: ver bloque al inicio del archivo.
// Uso:
//   ./sistema_difuso_sesion <Ciudad_Platillo_Angulo>
