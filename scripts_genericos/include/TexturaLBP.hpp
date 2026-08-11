#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace evaluador {

// Calcula el histograma LBP (Local Binary Pattern) clasico de 8 vecinos,
// restringido a los pixeles dentro de mascara (255 = incluir). El
// histograma resultante tiene 256 bins (LBP no-uniforme, el mas simple
// de calcular) y ya viene normalizado (suma 1.0), listo para pasar por
// reducirHistograma() y obtener Energia/Entropia/Varianza compactas —
// no tiene sentido interpretar los 256 valores crudos uno por uno.
//
// Nota: se uso LBP no-uniforme (256 bins) en vez de LBP uniforme (59
// bins) por simplicidad de implementacion. La diferencia practica es
// pequeña porque de cualquier forma el histograma se reduce a 3
// estadisticos compactos despues — si el numero de bins crudos
// importara para algo mas, ahi si valdria la pena migrar a la version
// uniforme.
std::vector<double> histogramaLBP(const cv::Mat& imagenGris, const cv::Mat& mascara);

} // namespace evaluador
