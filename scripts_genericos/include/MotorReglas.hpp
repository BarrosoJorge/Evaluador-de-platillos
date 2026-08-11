#pragma once

#include "Fuzzificador.hpp"
#include <utility>
#include <vector>

namespace evaluador {

// Tabla de reglas Sugeno de orden cero: cada termino linguistico de
// entrada dispara un valor de salida "singleton" fijo (no un conjunto
// difuso de salida) — es el diseño mas simple que produce un score
// crudo (1-10) sin necesitar definir funciones de membresia de salida
// por separado. La regla implicita es:
//   SI distancia es similar       ENTONCES score es valorSimilar
//   SI distancia es algo_distinto ENTONCES score es valorAlgoDistinto
//   SI distancia es muy_distinto  ENTONCES score es valorMuyDistinto
//
// Los 3 valores por defecto son un punto de partida razonable, NO
// calibrado con calificaciones reales del chef — es exactamente lo que
// AjustadorReglas (futuro, via ANFIS o algoritmo genetico) deberia
// reemplazar cuando existan esas calificaciones. Por eso son
// parametros del constructor y no constantes fijas en el codigo: para
// que ese reemplazo futuro sea cuestion de pasar otros numeros, no de
// reescribir MotorReglas ni Defuzzificador.
struct TablaReglas {
    double valorSimilar = 9.5;      // distancia baja = muy parecido al chef
    double valorAlgoDistinto = 6.0; // discrepancia moderada
    double valorMuyDistinto = 2.5;  // distancia alta = muy distinto al chef
};

// Evalua la tabla de reglas contra un grado de pertenencia ya calculado
// (etapa 5), devolviendo pares (fuerza_de_disparo, valor_singleton) —
// uno por cada regla que dispare con fuerza > 0. La fuerza de una regla
// difusa Sugeno de orden cero es directamente el grado de pertenencia
// del termino de entrada (no hace falta "min" ni "producto" porque cada
// regla tiene un solo antecedente).
std::vector<std::pair<double, double>> inferir(const GradosPertenencia& grados, const TablaReglas& tabla = TablaReglas());

} // namespace evaluador
