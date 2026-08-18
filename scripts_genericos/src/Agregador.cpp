#include "Agregador.hpp"
#include "Logger.hpp"

namespace evaluador {

namespace {

bool terminaEn(const std::string& s, const std::string& sufijo) {
    if (s.size() < sufijo.size()) return false;
    return s.compare(s.size() - sufijo.size(), sufijo.size(), sufijo) == 0;
}

} // namespace

std::optional<std::string> categoriaDeDimension(const std::string& nombreDimension) {
    // Agrupa dimensiones por atributo visual dominante.
    if (terminaEn(nombreDimension, "_color") || terminaEn(nombreDimension, "_mediaR"))
        return "Color y contraste";

    if (terminaEn(nombreDimension, "_textura") || terminaEn(nombreDimension, "_lbp") ||
        terminaEn(nombreDimension, "_glcm"))
        return "Texturas y formas";

    if (nombreDimension == "global" || nombreDimension == "simetria")
        return "Equilibrio y simetria";

    if (nombreDimension == "limpieza")
        return "Limpieza y vajilla";

    if (nombreDimension == "enfoque")
        return "Proporcion y enfoque";

    return "Altura y volumen";
}

RubricaAlumno agregarAlumno(const std::string& nombreAlumno, const std::vector<ScoreElemental>& scores) {
    RubricaAlumno resultado;
    resultado.alumno = nombreAlumno;

    for (const auto& categoria : categoriasRubrica()) {
        resultado.valorPorCategoria[categoria] = std::nullopt;
        resultado.dimensionesUsadasPorCategoria[categoria] = 0;
    }

    std::map<std::string, double> sumaPorCategoria;

    for (const auto& s : scores) {
        auto categoria = categoriaDeDimension(s.dimension);
        if (!categoria.has_value()) {
            Logger::instance().debug("Agregador",
                "Dimension sin categoria: " + s.dimension);
            continue;
        }

        sumaPorCategoria[categoria.value()] += s.score;
        resultado.dimensionesUsadasPorCategoria[categoria.value()]++;
    }

    for (const auto& categoria : categoriasRubrica()) {
        int n = resultado.dimensionesUsadasPorCategoria[categoria];
        if (n > 0) {
            resultado.valorPorCategoria[categoria] = sumaPorCategoria[categoria] / n;
        }
    }

    return resultado;
}

} // namespace evaluador
