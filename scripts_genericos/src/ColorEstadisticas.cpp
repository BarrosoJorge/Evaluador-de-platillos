#include "ColorEstadisticas.hpp"

namespace evaluador {

EstadisticosColor calcularEstadisticosColor(const cv::Mat& imagenBGR, const cv::Mat& mascara) {
    EstadisticosColor resultado;

    if (imagenBGR.empty()) return resultado;

    cv::Scalar media, desviacion;
    cv::meanStdDev(imagenBGR, media, desviacion, mascara);

    resultado.mediaB = media[0]; resultado.mediaG = media[1]; resultado.mediaR = media[2];
    resultado.stdB = desviacion[0]; resultado.stdG = desviacion[1]; resultado.stdR = desviacion[2];

    return resultado;
}

} // namespace evaluador
