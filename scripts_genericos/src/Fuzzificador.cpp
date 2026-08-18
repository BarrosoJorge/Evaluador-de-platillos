#include "Fuzzificador.hpp"
#include "Logger.hpp"
#include <algorithm>
#include <cmath>

namespace evaluador {

GradosPertenencia fuzzificar(double distancia, const AnclasPercentiles& anclas) {
    GradosPertenencia grados;

    grados.similar = 0.0;
    grados.algoDistinto = 0.0;
    grados.muyDistinto = 0.0;

    const double p33 = anclas.p33;
    const double p50 = anclas.p50;
    const double p66 = anclas.p66;

    // Caso degenerado: evitamos divisiones por cero y usamos una regla simple.
    if (p50 <= p33 || p66 <= p50) {
        if (distancia <= p50) {
            grados.similar = 1.0;
        } else {
            grados.muyDistinto = 1.0;
        }
        return grados;
    }

    if (distancia <= p33) {
        grados.similar = 1.0;
    } else if (distancia < p50) {
        grados.similar = (p50 - distancia) / (p50 - p33);
    }

    if (distancia > p33 && distancia < p50) {
        grados.algoDistinto = (distancia - p33) / (p50 - p33);
    } else if (distancia >= p50 && distancia < p66) {
        grados.algoDistinto = (p66 - distancia) / (p66 - p50);
    }

    if (distancia >= p66) {
        grados.muyDistinto = 1.0;
    } else if (distancia > p50) {
        grados.muyDistinto = (distancia - p50) / (p66 - p50);
    }

    return grados;
}

} // namespace evaluador