#include "ReductorEstadisticos.hpp"

#include <cmath>
#include <numeric>

namespace evaluador {

EstadisticosHistograma reducirHistograma(const std::vector<double>& histograma) {
    EstadisticosHistograma resultado;
    if (histograma.empty()) return resultado;

    double suma = std::accumulate(histograma.begin(), histograma.end(), 0.0);
    if (suma <= 0.0) return resultado;

    // Normalizar por si el histograma de entrada no sumaba exactamente 1.0
    std::vector<double> p(histograma.size());
    for (size_t i = 0; i < histograma.size(); ++i) {
        p[i] = histograma[i] / suma;
    }

    // Media (usando el indice de bin como valor de la variable)
    for (size_t i = 0; i < p.size(); ++i) {
        resultado.media += static_cast<double>(i) * p[i];
    }

    // Energia y entropia
    for (double pi : p) {
        resultado.energia += pi * pi;
        if (pi > 0.0) {
            resultado.entropia -= pi * std::log2(pi);
        }
    }

    // Varianza (respecto a la media ya calculada)
    for (size_t i = 0; i < p.size(); ++i) {
        double diff = static_cast<double>(i) - resultado.media;
        resultado.varianza += diff * diff * p[i];
    }

    return resultado;
}

} // namespace evaluador
