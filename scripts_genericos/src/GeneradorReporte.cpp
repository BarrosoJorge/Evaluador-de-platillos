#include "GeneradorReporte.hpp"

#include <sstream>
#include <iomanip>

namespace evaluador {

std::string generarReporteTexto(const RubricaAlumno& rubrica) {
    std::ostringstream oss;

    oss << "Rubrica: " << rubrica.alumno << "\n";

    for (const auto& categoria : categoriasRubrica()) {
        auto valor = rubrica.valorPorCategoria.at(categoria);
        int n = rubrica.dimensionesUsadasPorCategoria.at(categoria);

        oss << std::left << std::setw(28) << categoria << ": ";

        if (valor.has_value()) {
            oss << std::fixed << std::setprecision(1) << valor.value() << " / 10"
                << "  (" << n << " dimension(es))";
        } else {
            oss << "Sin dato disponible";
        }
        oss << "\n";
    }

    return oss.str();
}

} // namespace evaluador
