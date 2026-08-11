#pragma once

#include "Agregador.hpp"
#include <string>

namespace evaluador {

// Genera el reporte de texto legible para UN alumno, mostrando las 6
// categorias con su score (o "Sin dato disponible" si no aplica) y
// cuantas dimensiones elementales se promediaron para cada una — esto
// ultimo es importante para que quien lea el reporte sepa si un
// numero viene de 3 regiones combinadas o de una sola.
std::string generarReporteTexto(const RubricaAlumno& rubrica);

} // namespace evaluador
