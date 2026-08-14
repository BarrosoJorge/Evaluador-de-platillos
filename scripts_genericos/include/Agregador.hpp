#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace evaluador {

/// Devuelve las categorias de la rubrica en orden fijo.
inline const std::vector<std::string>& categoriasRubrica() {
    static const std::vector<std::string> categorias = {
        "Color y contraste",
        "Equilibrio y simetria",
        "Altura y volumen",
        "Texturas y formas",
        "Limpieza y vajilla",
        "Proporcion y enfoque"
    };
    return categorias;
}

/// Asigna una dimension elemental a una categoria de rubrica por patron.
/// Devuelve std::nullopt cuando la dimension no aplica a ninguna categoria.
/// Nota: "Altura y volumen" se mantiene sin asignacion por diseno del dominio.
std::optional<std::string> categoriaDeDimension(const std::string& nombreDimension);

/// Score elemental generado para una dimension de un alumno.
struct ScoreElemental {
    std::string dimension;
    double score = 0.0;
};

/// Resultado de agregar scores por categoria para un alumno.
/// Las categorias sin datos se representan con std::nullopt.
struct RubricaAlumno {
    std::string alumno;
    std::map<std::string, std::optional<double>> valorPorCategoria;
    std::map<std::string, int> dimensionesUsadasPorCategoria;
};

/// Agrega los scores elementales de un alumno por categoria.
/// Devuelve una rubrica con promedio simple de dimensiones por categoria.
RubricaAlumno agregarAlumno(const std::string& nombreAlumno, const std::vector<ScoreElemental>& scores);

} // namespace evaluador
