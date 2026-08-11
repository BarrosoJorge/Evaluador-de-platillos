#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace evaluador {

// Las 6 categorias de la rubrica final, en el orden que se definio
// desde el inicio de la conversacion.
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

// Determina a que categoria de la rubrica pertenece una dimension
// elemental, por PATRON (sufijo/nombre exacto), no por lista fija de
// nombres — necesario porque los nombres de region varian por platillo
// (ej. "carne_color", "salsa_color" ambos son "Color y contraste", sin
// importar que "carne"/"salsa" sean especificos de ESTE platillo).
//
// Reglas:
//   termina en "_color"            -> Color y contraste
//   termina en "_textura" o "_lbp" -> Texturas y formas
//   == "simetria"                  -> Equilibrio y simetria
//   == "limpieza"                  -> Limpieza y vajilla
//   == "proporcion" o == "enfoque" -> Proporcion y enfoque
//   cualquier otra cosa            -> std::nullopt (no se agrega a nada)
//
// "Altura y volumen" NUNCA aparece aqui a proposito: ninguna dimension
// elemental del pipeline mide volumen (no es medible con una sola foto
// RGB sin profundidad — lo señalamos desde el inicio y nunca se
// resolvio con un proxy). Esa categoria queda sin dato por diseño, no
// por omision.
std::optional<std::string> categoriaDeDimension(const std::string& nombreDimension);

// Un score elemental de un alumno (misma fila que produjo la etapa 6).
struct ScoreElemental {
    std::string dimension;
    double score = 0.0;
};

// Resultado de agregar todos los scores elementales de UN alumno en
// las 6 categorias. Categorias sin ninguna dimension elemental
// asignada (por diseño, "Altura y volumen"; o por falta de datos en
// una sesion concreta) quedan en std::nullopt en vez de forzar un
// numero — GeneradorReporte debe mostrar eso explicitamente, no
// esconderlo.
struct RubricaAlumno {
    std::string alumno;
    std::map<std::string, std::optional<double>> valorPorCategoria;
    std::map<std::string, int> dimensionesUsadasPorCategoria;
};

// Agrega los scores elementales de un alumno en las 6 categorias,
// usando promedio simple (sin ponderar) de las dimensiones que caen en
// cada categoria segun categoriaDeDimension(). Es el metodo de
// agregacion por defecto — no calibrado con datos reales del chef, ver
// nota en el .cpp.
RubricaAlumno agregarAlumno(const std::string& nombreAlumno, const std::vector<ScoreElemental>& scores);

} // namespace evaluador
