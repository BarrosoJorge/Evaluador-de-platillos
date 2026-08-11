#include "MotorReglas.hpp"

namespace evaluador {

std::vector<std::pair<double, double>> inferir(const GradosPertenencia& grados, const TablaReglas& tabla) {
    std::vector<std::pair<double, double>> disparos;

    if (grados.similar > 0.0) disparos.emplace_back(grados.similar, tabla.valorSimilar);
    if (grados.algoDistinto > 0.0) disparos.emplace_back(grados.algoDistinto, tabla.valorAlgoDistinto);
    if (grados.muyDistinto > 0.0) disparos.emplace_back(grados.muyDistinto, tabla.valorMuyDistinto);

    return disparos;
}

} // namespace evaluador
