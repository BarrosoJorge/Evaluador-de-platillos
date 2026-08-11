#include "Fuzzificador.hpp"
#include "Logger.hpp"
#include <algorithm>
#include <cmath>

namespace evaluador {

/**
 * Convierte una distancia de Mahalanobis en grados de pertenencia difusa.
 * Utiliza tres funciones de pertenencia triangulares/trapezoidales basadas 
 * en los percentiles (anclas) calculados para cada sesión y dimensión.
 */
GradosPertenencia fuzzificar(double distancia, const AnclasPercentiles& anclas) {
    GradosPertenencia grados;
    
    // Inicializar grados en cero
    grados.similar = 0.0;
    grados.algoDistinto = 0.0;
    grados.muyDistinto = 0.0;

    const double p33 = anclas.p33;
    const double p50 = anclas.p50;
    const double p66 = anclas.p66;

    // 1. Manejo de casos degenerados (anclas muy juntas o iguales)
    // Evitamos división por cero y forzamos reglas booleanas simples
    if (p50 <= p33 || p66 <= p50) {
        if (distancia <= p50) {
            grados.similar = 1.0;
        } else {
            grados.muyDistinto = 1.0;
        }
        return grados;
    }

    // 2. Grado de pertenencia: "Similar" (S)
    // S es 1.0 si está bajo el percentil 33, decae a 0 al llegar al 50.
    if (distancia <= p33) {
        grados.similar = 1.0;
    } else if (distancia < p50) {
        grados.similar = (p50 - distancia) / (p50 - p33);
    } else {
        grados.similar = 0.0;
    }

    // 3. Grado de pertenencia: "Algo Distinto" (AD)
    // Triangular con pico en p50, base en [p33, p66]
    if (distancia > p33 && distancia < p50) {
        grados.algoDistinto = (distancia - p33) / (p50 - p33);
    } else if (distancia >= p50 && distancia < p66) {
        grados.algoDistinto = (p66 - distancia) / (p66 - p50);
    } else {
        grados.algoDistinto = 0.0;
    }

    // 4. Grado de pertenencia: "Muy Distinto" (MD)
    // MD es 0 hasta p50, empieza a subir y llega a 1.0 en p66.
    if (distancia >= p66) {
        grados.muyDistinto = 1.0;
    } else if (distancia > p50) {
        grados.muyDistinto = (distancia - p50) / (p66 - p50);
    } else {
        grados.muyDistinto = 0.0;
    }

    return grados;
}

} // namespace evaluador