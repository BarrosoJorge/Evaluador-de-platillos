#include "EstimadorLedoitWolf.hpp"
#include "Logger.hpp"

namespace evaluador {

ResultadoLedoitWolf estimarCovarianzaLedoitWolf(const cv::Mat& X) {
    ResultadoLedoitWolf resultado;

    const int n = X.rows;
    const int p = X.cols;

    if (n < 2 || p < 1) {
        Logger::instance().error("EstimadorLedoitWolf",
            "Se requieren al menos 2 muestras. n=" + std::to_string(n) + ", p=" + std::to_string(p));
        resultado.covarianza = cv::Mat::eye(std::max(p, 1), std::max(p, 1), CV_64F);
        resultado.intensidadShrinkage = 1.0;
        return resultado;
    }

    // Centrar los datos (restar la media de cada columna)
    cv::Mat media;
    cv::reduce(X, media, 0, cv::REDUCE_AVG, CV_64F); // 1 x p

    cv::Mat Y(n, p, CV_64F);
    for (int k = 0; k < n; ++k) {
        X.row(k).convertTo(Y.row(k), CV_64F);
        Y.row(k) -= media;
    }

    // S = (1/n) * Y^T * Y  (covarianza muestral, version sesgada — la
    // convencion que usa el paper de Ledoit-Wolf)
    cv::Mat S = (Y.t() * Y) / static_cast<double>(n);

    // Target: mu * I, con mu = traza(S)/p (promedio de las varianzas)
    double mu = cv::trace(S)[0] / p;
    cv::Mat F = cv::Mat::eye(p, p, CV_64F) * mu;

    // gamma_hat = ||S - F||_F^2
    cv::Mat diffSF = S - F;
    double gammaHat = cv::sum(diffSF.mul(diffSF))[0];

    // pi_hat = (1/n) * sum_k || y_k y_k^T - S ||_F^2
    double piHat = 0.0;
    for (int k = 0; k < n; ++k) {
        cv::Mat yk = Y.row(k); // 1 x p
        cv::Mat outer = yk.t() * yk; // p x p
        cv::Mat diff = outer - S;
        piHat += cv::sum(diff.mul(diff))[0];
    }
    piHat /= static_cast<double>(n);

    double delta = (gammaHat > 0.0) ? (piHat / (static_cast<double>(n) * gammaHat)) : 1.0;
    delta = std::max(0.0, std::min(1.0, delta));

    resultado.covarianza = delta * F + (1.0 - delta) * S;
    resultado.intensidadShrinkage = delta;

    Logger::instance().info("EstimadorLedoitWolf",
        "n=" + std::to_string(n) + " p=" + std::to_string(p) +
        " shrinkage=" + std::to_string(delta));

    return resultado;
}

} // namespace evaluador
