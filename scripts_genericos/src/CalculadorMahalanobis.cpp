#include "CalculadorMahalanobis.hpp"
#include "Logger.hpp"

namespace evaluador {

double calcularDistanciaMahalanobis(const cv::Mat& x, const cv::Mat& xChef, const cv::Mat& covarianza) {
    if (x.empty() || xChef.empty() || covarianza.empty()) {
        Logger::instance().error("CalculadorMahalanobis", "Entrada vacia.");
        return 0.0;
    }

    cv::Mat covInv;
    cv::invert(covarianza, covInv, cv::DECOMP_SVD);

    cv::Mat diff = x - xChef; // 1 x p
    cv::Mat resultado = diff * covInv * diff.t(); // 1 x 1

    double valor = resultado.at<double>(0, 0);
    if (valor < 0.0) valor = 0.0; // proteccion numerica: no deberia pasar con Sigma valida

    return std::sqrt(valor);
}

} // namespace evaluador
