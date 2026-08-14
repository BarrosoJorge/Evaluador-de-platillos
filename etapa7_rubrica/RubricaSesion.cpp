/* Etapa 7: agrega scores elementales por categoria de rubrica.
 * Genera salida CSV y reporte textual por sesion.
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

/// Lee scores elementales y devuelve filas validas por alumno/dimension.
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

/// Ejecuta la agregacion de rubrica para una sesion.
/// Devuelve 0 si los reportes se generan correctamente.
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

    // Agrupa los scores por alumno.
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
