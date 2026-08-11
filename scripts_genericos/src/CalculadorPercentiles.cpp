#include "CalculadorPercentiles.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <cmath>

namespace evaluador {

double calcularPercentil(std::vector<double> valores, double p) {
    if (valores.empty()) return 0.0;
    if (valores.size() == 1) return valores[0];

    p = std::max(0.0, std::min(1.0, p));
    std::sort(valores.begin(), valores.end());

    // Metodo 'linear' (igual al default de numpy.percentile): la
    // posicion continua dentro del arreglo ordenado, interpolando
    // entre los dos vecinos mas cercanos.
    double posicion = p * (valores.size() - 1);
    size_t indiceInferior = static_cast<size_t>(std::floor(posicion));
    size_t indiceSuperior = static_cast<size_t>(std::ceil(posicion));

    if (indiceInferior == indiceSuperior) {
        return valores[indiceInferior];
    }

    double fraccion = posicion - indiceInferior;
    return valores[indiceInferior] + fraccion * (valores[indiceSuperior] - valores[indiceInferior]);
}

AnclasPercentiles calcularAnclas(const std::vector<double>& distancias) {
    AnclasPercentiles anclas;

    if (distancias.size() < 2) {
        Logger::instance().warn("CalculadorPercentiles",
            "Se necesitan al menos 2 distancias para calcular anclas; se recibieron " +
            std::to_string(distancias.size()) + ".");
        return anclas;
    }

    anclas.p33 = calcularPercentil(distancias, 0.33);
    anclas.p50 = calcularPercentil(distancias, 0.50);
    anclas.p66 = calcularPercentil(distancias, 0.66);

    return anclas;
}

} // namespace evaluador
