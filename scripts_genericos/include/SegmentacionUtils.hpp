#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>   // cv::inpaint
#include <vector>

namespace evaluador {

// Utilidades compartidas para segmentación y limpieza morfológica.
// Se usan por la etapa de segmentación y por la fase de localización.
enum class FormaROI { Rectangulo, Elipse };

// Aplica GrabCut con el rectangulo ROI dado. Modela fondo y foreground
// con mezclas gaussianas (GMM) e itera hasta producir una mascara.
// Retorna mascara binaria: 255 = objeto (platillo o region), 0 = resto.
// forma=Rectangulo (default): el rectangulo completo es el ROI inicial.
// forma=Elipse: se inscribe una elipse dentro de ese mismo rectangulo
// y esa es la forma inicial que se le pasa a GrabCut.
cv::Mat aplicarGrabCut(const cv::Mat& imagen, const cv::Rect& roi, int iteraciones = 5,
                       FormaROI forma = FormaROI::Rectangulo);

// Postprocesamiento morfologico de la mascara binaria.
// Apertura: elimina filamentos/artefactos pequeños mal clasificados.
// Cierre: rellena huecos internos y da continuidad al area.
// mantenerSoloElMasGrande=true (default): descarta cualquier
// componente conexo que no sea el mas grande (util para aislar el
// platillo del fondo, donde un objeto secundario como un cubierto no
// deberia colarse). Pasar false cuando SI quieres conservar varios
// componentes desconectados a proposito (ej. fusionar 2 manchas de
// salsa separadas en una sola region).
cv::Mat postprocesamientoMorfologico(const cv::Mat& mascaraBinaria, bool mantenerSoloElMasGrande = true);

// Aplica la mascara a la imagen — fuera de la mascara queda en negro.
cv::Mat aplicarMascaraAImagen(const cv::Mat& imagen, const cv::Mat& mascara);

// Inpainting (metodo de Telea) sobre huecos internos del area enmascarada.
// Solo actua si los huecos son razonables (entre 1px y 35% del bbox);
// si la mascara esta muy incompleta, prefiere no inventar contenido.
cv::Mat rellenarHuecos(const cv::Mat& segmentada, const cv::Mat& mascara);

// Escala una imagen para que quepa en max_w x max_h sin ampliar ni
// distorsionar (util para mostrar en pantalla durante sesiones interactivas).
cv::Mat escalarParaVista(const cv::Mat& imagen, int maxAncho = 900, int maxAlto = 700);

// Reconstruye una mascara binaria a partir de una imagen ya aislada
// del fondo (fondo negro): cualquier pixel no-negro se considera
// contenido. Se usa en varias etapas para recuperar el area del
// platillo completo a partir de una imagen preprocesada, sin tener
// que rastrear la mascara original de la etapa 1 (que no coincide en
// resolucion/encuadre despues de que Preprocesador recorta/rota/escala).
cv::Mat mascaraDesdeNoNegro(const cv::Mat& imagen);

// --- Clustering de color (compartido entre Segmentador.cpp y
// EtiquetadorAutomatico.cpp) ---

// Un grupo de pixeles propuesto automaticamente por color.
struct ClusterColor {
    cv::Mat mascara; // binaria, mismo tamaño que la imagen de origen
    cv::Vec3d colorPromedioBGR{0.0, 0.0, 0.0};
};

// Color promedio (BGR) de una imagen dentro de una mascara binaria.
cv::Vec3d colorPromedioEnMascara(const cv::Mat& imagen, const cv::Mat& mascara);

// Corre k-means en espacio Lab sobre los pixeles dentro de mascaraArea,
// y arma un cluster (mascara binaria + color promedio) por grupo.
// etiquetasPorPixel (salida) mapea cada pixel de la imagen a su indice
// de cluster (-1 = fuera de mascaraArea) — permite resolver clicks u
// otras consultas por posicion en O(1).
std::vector<ClusterColor> proponerClusters(const cv::Mat& imagen, const cv::Mat& mascaraArea,
                                            int k, cv::Mat& etiquetasPorPixel);

// Paleta de colores distintos para visualizar hasta K clusters. No
// tiene significado semantico, solo necesita que colores vecinos se
// vean distintos entre si.
cv::Scalar colorDePaleta(int indice);

} // namespace evaluador