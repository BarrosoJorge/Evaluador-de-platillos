#pragma once

#include <vector>

namespace evaluador {

// Estadisticos compactos derivados de un histograma normalizado
// (una distribucion de probabilidad discreta). Son genericos — no
// asumen que el histograma viene de GLCM, LBP, color, o cualquier otra
// fuente. GLCM tiene ademas sus propios descriptores especificos
// (Contraste, Correlacion, Homogeneidad, IDF) que SI dependen de la
// estructura 2D de co-ocurrencia y no se pueden derivar de un
// histograma 1D generico — esos viven en TexturaGLCM, no aqui.
struct EstadisticosHistograma {
    double energia = 0.0;   // suma de p_i^2 — que tan concentrada esta la distribucion
    double entropia = 0.0;  // -suma p_i*log2(p_i) — que tan dispersa/aleatoria es
    double varianza = 0.0;  // varianza de la variable aleatoria (usando el indice de bin como valor)
    double media = 0.0;     // media de la variable aleatoria
};

// Reduce un histograma normalizado (debe sumar ~1.0) a sus estadisticos
// compactos. Si el histograma no esta normalizado, se normaliza
// internamente antes de calcular (division por la suma total).
EstadisticosHistograma reducirHistograma(const std::vector<double>& histograma);

} // namespace evaluador
