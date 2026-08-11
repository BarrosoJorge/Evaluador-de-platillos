#pragma once

#include <opencv2/opencv.hpp>

namespace evaluador {

// Media y desviacion estandar por canal (B,G,R) dentro de una region
// enmascarada. Es la parte "color" del vector de features por region.
struct EstadisticosColor {
    double mediaB = 0.0, mediaG = 0.0, mediaR = 0.0;
    double stdB = 0.0, stdG = 0.0, stdR = 0.0;
};

EstadisticosColor calcularEstadisticosColor(const cv::Mat& imagenBGR, const cv::Mat& mascara);

} // namespace evaluador
