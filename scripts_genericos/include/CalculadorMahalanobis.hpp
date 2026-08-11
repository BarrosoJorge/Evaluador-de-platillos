#pragma once

#include <opencv2/opencv.hpp>

namespace evaluador {

// Distancia de Mahalanobis: D(x, x_chef) = sqrt( (x-x_chef) * Sigma^-1 * (x-x_chef)^T )
//
// Se usa pseudo-inversa (SVD) en vez de inversa directa: aunque el
// shrinkage de Ledoit-Wolf ya deberia dejar la matriz bien
// condicionada, la pseudo-inversa es una salvaguarda barata si algun
// dia esta funcion se usa con una covarianza que no paso por shrinkage.
//
// x, xChef: vectores fila 1xp (mismo formato que regionAVector/globalAVector).
// covarianza: p x p, tipicamente la salida de estimarCovarianzaLedoitWolf().
double calcularDistanciaMahalanobis(const cv::Mat& x, const cv::Mat& xChef, const cv::Mat& covarianza);

} // namespace evaluador
