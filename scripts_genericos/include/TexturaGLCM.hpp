#pragma once

#include <opencv2/opencv.hpp>

namespace evaluador {

// Los 7 descriptores de Haralick calculados sobre la matriz de
// co-ocurrencia de niveles de gris (GLCM). Referencia: Haralick,
// Shanmugam & Dinstein (1973).
struct FeaturesGLCM {
    double energia = 0.0;       // suma p(i,j)^2 — uniformidad de la textura
    double contraste = 0.0;     // suma p(i,j)*(i-j)^2 — variacion local de intensidad
    double correlacion = 0.0;   // dependencia lineal entre pixeles vecinos
    double homogeneidad = 0.0;  // IDM: suma p(i,j)/(1+(i-j)^2) — suavidad
    double idf = 0.0;           // Inverse Difference: suma p(i,j)/(1+|i-j|) — variante de homogeneidad sin elevar al cuadrado
    double entropia = 0.0;      // -suma p(i,j)*log2(p(i,j)) — aleatoriedad de la textura
    double varianza = 0.0;      // dispersion de los niveles de gris ponderada por p(i,j)
};

// Calcula GLCM y sus 7 descriptores de Haralick sobre la region de
// imagenGris delimitada por mascara (255 = incluir, 0 = ignorar).
// niveles: a cuantos niveles de gris se cuantiza la imagen antes de
//          construir la matriz de co-ocurrencia (menos niveles = matriz
//          mas pequeña y estadisticamente mas estable con regiones
//          chicas; 8 es un punto de partida razonable).
// distancia/angulo: desplazamiento del par de pixeles que define la
//          co-ocurrencia. Angulo 0 = horizontal (par (i,j) e (i+distancia,j)).
//          Para una descripcion de textura mas completa se podrian
//          promediar varios angulos (0/45/90/135) — se dejo un solo
//          angulo por simplicidad inicial; es facil de extender.
FeaturesGLCM calcularGLCM(const cv::Mat& imagenGris, const cv::Mat& mascara,
                            int niveles = 8, int distancia = 1);

} // namespace evaluador
