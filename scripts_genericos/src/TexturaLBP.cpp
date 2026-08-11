#include "TexturaLBP.hpp"

namespace evaluador {

std::vector<double> histogramaLBP(const cv::Mat& imagenGris, const cv::Mat& mascara) {
    std::vector<double> histograma(256, 0.0);

    if (imagenGris.empty() || imagenGris.rows < 3 || imagenGris.cols < 3) {
        return histograma;
    }

    double total = 0.0;

    // LBP clasico: para cada pixel (excepto el borde de 1px), compara
    // con sus 8 vecinos en sentido horario y arma un byte donde cada
    // bit es 1 si el vecino es >= al pixel central.
    for (int y = 1; y < imagenGris.rows - 1; ++y) {
        for (int x = 1; x < imagenGris.cols - 1; ++x) {
            if (!mascara.empty() && mascara.at<uchar>(y, x) == 0) continue;

            uchar centro = imagenGris.at<uchar>(y, x);
            unsigned char codigo = 0;

            codigo |= (imagenGris.at<uchar>(y - 1, x - 1) >= centro) << 7;
            codigo |= (imagenGris.at<uchar>(y - 1, x)     >= centro) << 6;
            codigo |= (imagenGris.at<uchar>(y - 1, x + 1) >= centro) << 5;
            codigo |= (imagenGris.at<uchar>(y,     x + 1) >= centro) << 4;
            codigo |= (imagenGris.at<uchar>(y + 1, x + 1) >= centro) << 3;
            codigo |= (imagenGris.at<uchar>(y + 1, x)     >= centro) << 2;
            codigo |= (imagenGris.at<uchar>(y + 1, x - 1) >= centro) << 1;
            codigo |= (imagenGris.at<uchar>(y,     x - 1) >= centro) << 0;

            histograma[codigo] += 1.0;
            total += 1.0;
        }
    }

    if (total > 0.0) {
        for (double& v : histograma) v /= total;
    }

    return histograma;
}

} // namespace evaluador
