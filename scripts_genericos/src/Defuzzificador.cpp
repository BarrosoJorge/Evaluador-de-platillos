#include "Defuzzificador.hpp"
#include "Logger.hpp"

namespace evaluador {

double defuzzificar(const std::vector<std::pair<double, double>>& disparos) {
    if (disparos.empty()) {
        Logger::instance().warn("Defuzzificador", "Sin disparos; usa valor medio.");
        return 5.5;
    }

    double sumaFuerzas = 0.0;
    double sumaPonderada = 0.0;

    for (const auto& [fuerza, valor] : disparos) {
        sumaFuerzas += fuerza;
        sumaPonderada += fuerza * valor;
    }

    if (sumaFuerzas <= 0.0) {
        Logger::instance().warn("Defuzzificador", "Suma de fuerzas <= 0; usa valor medio.");
        return 5.5;
    }

    return sumaPonderada / sumaFuerzas;
}

} // namespace evaluador
