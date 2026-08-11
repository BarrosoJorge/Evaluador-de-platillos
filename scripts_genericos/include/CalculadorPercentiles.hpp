#pragma once

#include <vector>

namespace evaluador {

// Los 3 anclas que definen las funciones de membresia de esta etapa,
// calculados a partir de la distribucion real de distancias de
// Mahalanobis de la sesion (no valores fijos elegidos a mano).
struct AnclasPercentiles {
    double p33 = 0.0;
    double p50 = 0.0; // mediana
    double p66 = 0.0;
};

// Calcula un percentil (p en [0,1]) de un conjunto de valores usando
// interpolacion lineal entre los dos valores ordenados mas cercanos
// (el mismo metodo que usa numpy por defecto — 'linear'). No modifica
// el vector de entrada.
double calcularPercentil(std::vector<double> valores, double p);

// Calcula los 3 anclas (p33, p50, p66) de un conjunto de distancias.
// Requiere al menos 2 valores; con menos, todas las anclas quedan en 0
// y el Fuzzificador debe manejarlo como caso degenerado (ver Fuzzificador.hpp).
AnclasPercentiles calcularAnclas(const std::vector<double>& distancias);

} // namespace evaluador
