#pragma once

#include <opencv2/opencv.hpp>

namespace evaluador {

// Estimador de covarianza con shrinkage hacia la identidad escalada,
// segun Ledoit & Wolf (2004), "A well-conditioned estimator for
// large-dimensional covariance matrices". Es una implementacion propia
// de la formula del paper (target = (traza(S)/p) * I), no un port de
// scikit-learn — los numeros pueden diferir ligeramente de
// sklearn.covariance.LedoitWolf en el ultimo decimal, pero el metodo
// es el mismo.
//
// Por que hace falta esto y no covarianza simple: con ~20 alumnos por
// sesion y vectores de 17 dimensiones por region, la covarianza
// muestral cruda es casi singular (mas dimensiones que margen de
// muestras) — shrinkage hacia la identidad la estabiliza sin que
// alguien tenga que elegir a mano cuanto "mezclar".
struct ResultadoLedoitWolf {
    cv::Mat covarianza;      // p x p, ya con shrinkage aplicado
    double intensidadShrinkage = 0.0; // delta en [0,1]: 0 = covarianza cruda, 1 = identidad pura
};

// X: matriz n x p (filas = muestras/alumnos, columnas = features).
// Requiere n >= 2. Si p > n, la covarianza cruda es singular por
// definicion — el shrinkage es lo que la hace invertible.
ResultadoLedoitWolf estimarCovarianzaLedoitWolf(const cv::Mat& X);

} // namespace evaluador
