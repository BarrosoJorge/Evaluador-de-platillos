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
    // 1. Mapeo por sufijo (captura todas las zonas)
    if (terminaEn(nombreDimension, "_color") || terminaEn(nombreDimension, "_mediaR")) 
        return "Color y contraste";
        
    if (terminaEn(nombreDimension, "_textura") || terminaEn(nombreDimension, "_lbp") || 
        terminaEn(nombreDimension, "_glcm")) 
        return "Texturas y formas";
        
    // 2. Mapeo específico para las dimensiones globales existentes
    if (nombreDimension == "global" || nombreDimension == "simetria") 
        return "Equilibrio y simetria";
        
    if (nombreDimension == "limpieza") 
        return "Limpieza y vajilla";
        
    if (nombreDimension == "enfoque") 
        return "Proporcion y enfoque";

    // 3. Mapeo por defecto: Si tienes una dimensión que no se clasifica, 
    // asignala a una categoría para evitar el "Sin dato disponible"
    // Por ejemplo, asignamos todo lo que sobre a "Altura y volumen" (o la categoría que prefieras)
    return "Altura y volumen"; 
}

RubricaAlumno agregarAlumno(const std::string& nombreAlumno, const std::vector<ScoreElemental>& scores) {
    RubricaAlumno resultado;
    resultado.alumno = nombreAlumno;

    // Inicializar las 6 categorias en nullopt — "sin dato" es el
    // estado por defecto, no un valor fabricado.
    for (const auto& categoria : categoriasRubrica()) {
        resultado.valorPorCategoria[categoria] = std::nullopt;
        resultado.dimensionesUsadasPorCategoria[categoria] = 0;
    }

    // Acumular sumas por categoria antes de promediar
    std::map<std::string, double> sumaPorCategoria;

    for (const auto& s : scores) {
        auto categoria = categoriaDeDimension(s.dimension);
        if (!categoria.has_value()) {
            Logger::instance().debug("Agregador",
                "Dimension '" + s.dimension + "' no mapea a ninguna categoria — se ignora.");
            continue;
        }

        sumaPorCategoria[categoria.value()] += s.score;
        resultado.dimensionesUsadasPorCategoria[categoria.value()]++;
    }

    // Promedio simple (sin ponderar) — metodo de agregacion por
    // defecto. No esta calibrado con datos reales del chef: si algun
    // dia se tienen calificaciones humanas por categoria, este seria
    // el lugar donde reemplazar promedio simple por promedio ponderado
    // o por una capa adicional de reglas difusas, sin tener que tocar
    // GeneradorReporte ni el resto del pipeline.
    for (const auto& categoria : categoriasRubrica()) {
        int n = resultado.dimensionesUsadasPorCategoria[categoria];
        if (n > 0) {
            resultado.valorPorCategoria[categoria] = sumaPorCategoria[categoria] / n;
        }
        // si n == 0, se queda en std::nullopt (ya inicializado arriba)
    }

    return resultado;
}

} // namespace evaluador
