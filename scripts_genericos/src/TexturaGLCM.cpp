#include "TexturaGLCM.hpp"

#include <cmath>
#include <vector>

namespace evaluador {

FeaturesGLCM calcularGLCM(const cv::Mat& imagenGris, const cv::Mat& mascara,
                            int niveles, int distancia) {
    FeaturesGLCM resultado;

    if (imagenGris.empty() || niveles < 2) return resultado;

    // Cuantizar la imagen a 'niveles' niveles de gris (0..niveles-1)
    cv::Mat cuantizada;
    imagenGris.convertTo(cuantizada, CV_32S, static_cast<double>(niveles) / 256.0);

    // Matriz de co-ocurrencia (niveles x niveles), acumulada como conteos
    std::vector<std::vector<double>> glcm(niveles, std::vector<double>(niveles, 0.0));
    double totalPares = 0.0;

    const int filas = imagenGris.rows;
    const int cols = imagenGris.cols;

    for (int y = 0; y < filas; ++y) {
        for (int x = 0; x < cols - distancia; ++x) {
            // Solo contar el par si AMBOS pixeles pertenecen a la mascara
            bool dentro = mascara.empty() ||
                          (mascara.at<uchar>(y, x) > 0 && mascara.at<uchar>(y, x + distancia) > 0);
            if (!dentro) continue;

            int i = std::min(cuantizada.at<int>(y, x), niveles - 1);
            int j = std::min(cuantizada.at<int>(y, x + distancia), niveles - 1);
            i = std::max(i, 0);
            j = std::max(j, 0);

            glcm[i][j] += 1.0;
            glcm[j][i] += 1.0; // matriz simetrica (par no dirigido)
            totalPares += 2.0;
        }
    }

    if (totalPares == 0.0) return resultado; // mascara vacia o region degenerada

    // Normalizar a probabilidades
    for (auto& fila : glcm) {
        for (double& v : fila) {
            v /= totalPares;
        }
    }

    // Medias y desviaciones marginales (necesarias para Correlacion)
    double mediaI = 0.0, mediaJ = 0.0;
    for (int i = 0; i < niveles; ++i) {
        for (int j = 0; j < niveles; ++j) {
            mediaI += i * glcm[i][j];
            mediaJ += j * glcm[i][j];
        }
    }

    double varI = 0.0, varJ = 0.0;
    for (int i = 0; i < niveles; ++i) {
        for (int j = 0; j < niveles; ++j) {
            varI += glcm[i][j] * (i - mediaI) * (i - mediaI);
            varJ += glcm[i][j] * (j - mediaJ) * (j - mediaJ);
        }
    }
    double desvI = std::sqrt(varI);
    double desvJ = std::sqrt(varJ);

    // Descriptores de Haralick
    for (int i = 0; i < niveles; ++i) {
        for (int j = 0; j < niveles; ++j) {
            double p = glcm[i][j];
            if (p <= 0.0) continue;

            double diff = static_cast<double>(i - j);

            resultado.energia += p * p;
            resultado.contraste += p * diff * diff;
            resultado.homogeneidad += p / (1.0 + diff * diff);
            resultado.idf += p / (1.0 + std::abs(diff));
            resultado.entropia -= p * std::log2(p);
            resultado.varianza += p * ((i - mediaI) * (i - mediaI) + (j - mediaJ) * (j - mediaJ)) / 2.0;

            if (desvI > 0.0 && desvJ > 0.0) {
                resultado.correlacion += p * (i - mediaI) * (j - mediaJ) / (desvI * desvJ);
            }
        }
    }

    return resultado;
}

} // namespace evaluador
