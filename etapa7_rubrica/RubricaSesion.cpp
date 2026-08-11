/*  RubricaSesion.cpp
 *
 *  Etapa 7 del pipeline — Rúbrica final (agregación).
 *
 *  Ultima etapa. Lee el CSV que dejo SistemaDifusoSesion.cpp (etapa 6)
 *  — un score 1-10 por alumno x dimension elemental — y para cada
 *  alumno:
 *    1. Agregador: promedio simple de las dimensiones que caen en cada
 *       una de las 6 categorias de la rubrica (categoriaDeDimension()
 *       decide el mapeo por patron, no por lista fija de nombres)
 *    2. GeneradorReporte: arma el texto legible final
 *
 *  "Altura y volumen" en la salida SIEMPRE aparece como "Sin dato
 *  disponible" — no es un bug, es la etapa 7 siendo honesta sobre una
 *  limitacion que se identifico desde el inicio de la conversacion
 *  (no se puede medir volumen con una sola foto RGB sin profundidad,
 *  y nunca se implemento un proxy). Si en algun momento se agrega una
 *  forma de estimarlo (etapa 3, nueva dimension elemental con sufijo
 *  reconocido por categoriaDeDimension), esta etapa lo recogeria sin
 *  cambios.
 *
 *  Entrada:  Data/SistemaDifuso/ScoresElementales/<claveDish>.csv
 *  Salida:   Data/Rubrica/Reportes/<claveDish>.csv          (machine-readable)
 *            Data/Rubrica/Reportes/<claveDish>_reporte.txt  (legible)
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -I../scripts_genericos/include \
 *          ../scripts_genericos/src/Logger.cpp \
 *          ../scripts_genericos/src/PathManager.cpp \
 *          ../scripts_genericos/src/Agregador.cpp \
 *          ../scripts_genericos/src/GeneradorReporte.cpp \
 *          RubricaSesion.cpp -o rubrica_sesion
 *
 *  Uso:
 *      ./rubrica_sesion <Ciudad_Platillo_Angulo>
 */

#include "Logger.hpp"
#include "PathManager.hpp"
#include "Agregador.hpp"
#include "GeneradorReporte.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace evaluador;

struct FilaScoreCruda {
    std::string alumno;
    std::string dimension;
    double score = 0.0;
};

std::vector<FilaScoreCruda> leerScores(const fs::path& ruta) {
    std::vector<FilaScoreCruda> filas;
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        Logger::instance().error("RubricaSesion", "No se pudo abrir: " + ruta.string());
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

        if (campos.size() != 4) continue; // alumno;dimension;distancia;score

        try {
            FilaScoreCruda f;
            f.alumno = campos[0];
            f.dimension = campos[1];
            f.score = std::stod(campos[3]);
            filas.push_back(f);
        } catch (const std::exception& e) {
            Logger::instance().warn("RubricaSesion", "Linea invalida ignorada: " + linea);
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

    fs::path rutaScores = rutas.scoresElementales() / (claveDish + ".csv");
    std::vector<FilaScoreCruda> filas = leerScores(rutaScores);

    if (filas.empty()) {
        Logger::instance().error("RubricaSesion",
            "No hay scores elementales para '" + claveDish + "'. Corre SistemaDifusoSesion.cpp primero.");
        return 1;
    }

    // Agrupar por alumno
    std::map<std::string, std::vector<ScoreElemental>> scoresPorAlumno;
    for (const auto& f : filas) {
        scoresPorAlumno[f.alumno].push_back({f.dimension, f.score});
    }

    fs::path rutaCsv = rutas.rubricaReportes() / (claveDish + ".csv");
    fs::path rutaTxt = rutas.rubricaReportes() / (claveDish + "_reporte.txt");
    std::error_code ec;
    fs::create_directories(rutaCsv.parent_path(), ec);

    std::ofstream csv(rutaCsv);
    csv << "alumno;categoria;score;n_dimensiones\n";

    std::ofstream txt(rutaTxt);

    int alumnosConVolumenFaltante = 0;

    for (const auto& [alumno, scores] : scoresPorAlumno) {
        RubricaAlumno rubrica = agregarAlumno(alumno, scores);

        for (const auto& categoria : categoriasRubrica()) {
            auto valor = rubrica.valorPorCategoria.at(categoria);
            int n = rubrica.dimensionesUsadasPorCategoria.at(categoria);

            csv << alumno << ";" << categoria << ";";
            if (valor.has_value()) {
                csv << valor.value();
            } else {
                csv << "NA";
            }
            csv << ";" << n << "\n";
        }

        txt << generarReporteTexto(rubrica) << "\n";

        if (!rubrica.valorPorCategoria.at("Altura y volumen").has_value()) {
            ++alumnosConVolumenFaltante;
        }
    }

    Logger::instance().info("RubricaSesion",
        "Completado: " + std::to_string(scoresPorAlumno.size()) + " alumno(s). " +
        "CSV: " + rutaCsv.string() + " | Reporte: " + rutaTxt.string());

    if (alumnosConVolumenFaltante > 0) {
        Logger::instance().warn("RubricaSesion",
            "'Altura y volumen' sin dato para " + std::to_string(alumnosConVolumenFaltante) +
            " alumno(s) — esperado, ninguna dimension elemental la alimenta todavia.");
    }

    return 0;
}

// Compilación: ver bloque al inicio del archivo.
// Uso:
//   ./rubrica_sesion <Ciudad_Platillo_Angulo>
