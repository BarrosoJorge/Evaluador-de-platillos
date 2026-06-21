/*  Caracteristicas.h
 *
 *  Header compartido para todos los extractores de características.
 *  Etapa 2 del pipeline — Extracción de características del evaluador de platillos.
 *
 *  Contenido:
 *    · Constantes globales (tamaños de ventana, distancias GLCM, ángulos)
 *    · Tipos comunes (ImageMetadata, DishImage, GLCMFeatures)
 *    · Carga de imágenes con parseo de metadatos desde la ruta
 *    · Ventaneo genérico — slideWindow<Func>
 *    · Cómputo de GLCM normalizada y 7 características Haralick
 *    · Cómputo de imagen LBP (8 vecinos)
 *    · Estadísticas de histograma 1D (media, varianza, energía, entropía, IDF, homogeneidad, correlación)
 *    · CSVWriter
 *    · Utilidades de normalización y guardado de mapas de características
 *
 *  Compilación de cada extractor:
 *      g++ -std=c++17 -O2 -o <nombre> <Extractor>.cpp `pkg-config --cflags --libs opencv4`
 *
 *  Referencia: EPráctico1 (ventaneo SDH, GLCM, LBP) y nota.md de esta carpeta.
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ============================================================================
// CONSTANTES GLOBALES
// ============================================================================

// 12 tamaños de ventana: 3, 5, 7, ..., 25 (EPráctico1 Parte 1)
inline const std::vector<int> WINDOW_SIZES = {3,5,7,9,11,13,15,17,19,21,23,25};

// Configuración GLCM (EPráctico1 Parte 2): 3 distancias × 4 ángulos = 12 matrices
inline const std::vector<int>    GLCM_DISTANCES = {1, 3, 7};
inline const std::vector<double> GLCM_ANGLES    = {0.0, 45.0, 90.0, 135.0};

// Número de niveles de cuantización para la GLCM
// (64 niveles es buen balance entre precisión y velocidad para ventanas pequeñas)
inline constexpr int GLCM_LEVELS = 64;

// ============================================================================
// TIPOS DE DATOS
// ============================================================================

struct ImageMetadata
{
    std::string city, dish, angle, author, quality;

    // Etiqueta compacta para CSV: Platillo_Autor[_Calidad]
    std::string label() const
    {
        std::string l = dish + "_" + author;
        if (!quality.empty()) l += "_" + quality;
        return l;
    }
};

struct DishImage
{
    cv::Mat color;         // BGR original
    cv::Mat gray;          // Escala de grises (uint8)
    std::string file_path;
    std::string filename;
    ImageMetadata metadata;

    explicit DishImage(const std::string& path)
        : file_path(path), filename(fs::path(path).filename().string())
    {
        color = cv::imread(path);
        if (color.empty())
            throw std::runtime_error("No se pudo cargar: " + path);
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    }
};

// 7 características Haralick extraídas de una GLCM normalizada
struct GLCMFeatures
{
    double energy      = 0.0;   // Energía / ASM
    double contrast    = 0.0;   // Contraste
    double correlation = 0.0;   // Correlación
    double homogeneity = 0.0;   // Homogeneidad
    double idf         = 0.0;   // Momento de Diferencia Inversa
    double entropy     = 0.0;   // Entropía
    double variance    = 0.0;   // Varianza

    std::vector<double> toVector() const
    {
        return {energy, contrast, correlation, homogeneity, idf, entropy, variance};
    }
    static std::vector<std::string> names()
    {
        return {"Energia","Contraste","Correlacion","Homogeneidad","IDF","Entropia","Varianza"};
    }
};

// ============================================================================
// CARGA DE IMÁGENES
// ============================================================================

// Extrae metadatos desde la jerarquía de carpetas: base/Ciudad/Platillo/Angulo/Autor/archivo
// (igual convención que el Preprocesador)
inline ImageMetadata parseMetadataFromPath(const fs::path& file_path, const fs::path& base_dir)
{
    ImageMetadata meta;
    fs::path rel = fs::relative(file_path, base_dir);
    std::vector<std::string> parts;
    for (const auto& comp : rel) parts.push_back(comp.string());

    if (parts.size() >= 5) {
        meta.city   = parts[0];
        meta.dish   = parts[1];
        meta.angle  = parts[2];
        meta.author = parts[3];
    }
    // Calidad desde el nombre del archivo (_B_, _R_, _M_)
    const std::string stem = fs::path(parts.back()).stem().string();
    for (const std::string& q : {"_B_","_R_","_M_"}) {
        if (stem.find(q) != std::string::npos) {
            meta.quality = std::string(1, q[1]);
            break;
        }
    }
    return meta;
}

inline std::vector<std::unique_ptr<DishImage>> loadImages(const std::string& directory)
{
    std::vector<std::unique_ptr<DishImage>> images;
    const fs::path base = directory;

    if (!fs::exists(base) || !fs::is_directory(base)) {
        std::cerr << "Error: directorio inválido: " << directory << std::endl;
        return images;
    }

    const std::vector<std::string> exts = {".jpg",".jpeg",".png",".bmp"};

    for (const auto& entry : fs::recursive_directory_iterator(base)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (!std::any_of(exts.begin(), exts.end(),
                         [&](const std::string& e){ return ext == e; })) continue;

        try {
            auto img = std::make_unique<DishImage>(entry.path().string());
            img->metadata = parseMetadataFromPath(entry.path(), base);
            std::cout << "Cargada: " << img->filename << std::endl;
            images.push_back(std::move(img));
        } catch (const std::exception& e) {
            std::cerr << "Advertencia: " << e.what() << std::endl;
        }
    }
    std::cout << "Total: " << images.size() << " imágenes." << std::endl;
    return images;
}

// ============================================================================
// VENTANEO — slideWindow<Func>
// ============================================================================

// Desliza una ventana cuadrada de tamaño w×w sobre la imagen gray con paso=1.
// Llama a f(roi, row, col) para cada posición válida.
// La imagen resultante de los valores de la función tiene tamaño (H-w+1)×(W-w+1).
template<typename Func>
inline void slideWindow(const cv::Mat& gray, int w, Func&& f)
{
    const int out_rows = gray.rows - w + 1;
    const int out_cols = gray.cols - w + 1;
    for (int r = 0; r < out_rows; ++r)
        for (int c = 0; c < out_cols; ++c)
            std::forward<Func>(f)(gray(cv::Rect(c, r, w, w)), r, c);
}

// Normaliza un mapa de características float a uint8 [0,255] para guardar como PNG
inline cv::Mat normalizeToImage(const cv::Mat& fmap)
{
    double mn, mx;
    cv::minMaxLoc(fmap, &mn, &mx);
    cv::Mat img;
    if (mx - mn < 1e-12)
        img = cv::Mat::zeros(fmap.size(), CV_8UC1);
    else
        fmap.convertTo(img, CV_8UC1, 255.0 / (mx - mn), -255.0 * mn / (mx - mn));
    return img;
}

// Guarda un mapa de características como imagen PNG bajo out_dir/img_stem/
inline bool saveFeatureMap(const cv::Mat& fmap,
                           const std::string& out_dir,
                           const std::string& img_stem,
                           const std::string& feature_name,
                           int window_size)
{
    fs::path dir = fs::path(out_dir) / img_stem;
    try { fs::create_directories(dir); } catch (...) { return false; }
    const std::string name = feature_name + "_w" + std::to_string(window_size) + ".png";
    return cv::imwrite((dir / name).string(), normalizeToImage(fmap));
}

// ============================================================================
// ESTADÍSTICAS DE HISTOGRAMA 1D
// ============================================================================

// Histograma normalizado de un ROI en escala de grises, con n_levels bins en [0, 255]
inline std::vector<double> computeHist1D(const cv::Mat& gray_roi, int n_levels = 256)
{
    std::vector<double> hist(n_levels, 0.0);
    const int total = gray_roi.rows * gray_roi.cols;
    for (int r = 0; r < gray_roi.rows; ++r)
        for (int c = 0; c < gray_roi.cols; ++c)
            ++hist[gray_roi.at<uchar>(r,c) * (n_levels - 1) / 255];
    if (total > 0)
        for (auto& h : hist) h /= total;
    return hist;
}

// Media del histograma: μ = Σ i · p[i]
inline double histMean(const std::vector<double>& h)
{
    double mu = 0.0;
    for (size_t i = 0; i < h.size(); ++i) mu += (double)i * h[i];
    return mu;
}

// Varianza: σ² = Σ (i − μ)² · p[i]
inline double histVariance(const std::vector<double>& h, double mu)
{
    double v = 0.0;
    for (size_t i = 0; i < h.size(); ++i) v += (i - mu) * (i - mu) * h[i];
    return v;
}

// Energía (ASM): Σ p[i]²
inline double histEnergy(const std::vector<double>& h)
{
    double e = 0.0;
    for (auto p : h) e += p * p;
    return e;
}

// Entropía de Shannon: −Σ p[i] · log₂(p[i])
inline double histEntropy(const std::vector<double>& h)
{
    double e = 0.0;
    for (auto p : h) if (p > 1e-12) e -= p * std::log2(p);
    return e;
}

// Contraste (segundo momento central respecto a la media)
inline double histContrast(const std::vector<double>& h, double mu)
{
    return histVariance(h, mu);
}

// Homogeneidad: Σ p[i] / (1 + |i − μ|)
inline double histHomogeneity(const std::vector<double>& h, double mu)
{
    double hom = 0.0;
    for (size_t i = 0; i < h.size(); ++i)
        hom += h[i] / (1.0 + std::abs((double)i - mu));
    return hom;
}

// Momento de Diferencia Inversa (IDF): Σ p[i] / (1 + (i − μ)²)
inline double histIDF(const std::vector<double>& h, double mu)
{
    double idf = 0.0;
    for (size_t i = 0; i < h.size(); ++i)
        idf += h[i] / (1.0 + (i - mu) * (i - mu));
    return idf;
}

// Correlación normalizada del histograma: (E[i²] − μ²) / σ²
inline double histCorrelation(const std::vector<double>& h)
{
    const double mu  = histMean(h);
    const double var = histVariance(h, mu);
    if (var < 1e-12) return 0.0;
    double ei2 = 0.0;
    for (size_t i = 0; i < h.size(); ++i) ei2 += (double)i * (double)i * h[i];
    return (ei2 - mu * mu) / var;
}

// ============================================================================
// GLCM — MATRICES DE CO-OCURRENCIA
// ============================================================================

// Computa la GLCM normalizada y simétrica de un ROI gris.
//   distance  : desplazamiento entre el par de píxeles (1, 3, 7)
//   angle_deg : dirección del par (0°, 45°, 90°, 135°)
//   levels    : número de niveles de cuantización del gris (reduce el tamaño de la matriz)
inline cv::Mat computeGLCM(const cv::Mat& gray_roi,
                            int distance, double angle_deg,
                            int levels = GLCM_LEVELS)
{
    // Cuantizar a `levels` niveles
    cv::Mat q(gray_roi.size(), CV_32S);
    for (int r = 0; r < gray_roi.rows; ++r)
        for (int c = 0; c < gray_roi.cols; ++c)
            q.at<int>(r,c) = static_cast<int>(gray_roi.at<uchar>(r,c)) * (levels - 1) / 255;

    // Dirección desde el ángulo
    int dr = 0, dc = 0;
    if      (angle_deg ==   0.0) { dr =  0;        dc =  distance; }
    else if (angle_deg ==  45.0) { dr = -distance;  dc =  distance; }
    else if (angle_deg ==  90.0) { dr = -distance;  dc =  0;        }
    else if (angle_deg == 135.0) { dr = -distance;  dc = -distance; }

    cv::Mat glcm = cv::Mat::zeros(levels, levels, CV_64F);

    for (int r = 0; r < q.rows; ++r) {
        for (int c = 0; c < q.cols; ++c) {
            const int nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= q.rows || nc < 0 || nc >= q.cols) continue;
            const int i = q.at<int>(r, c);
            const int j = q.at<int>(nr, nc);
            glcm.at<double>(i, j) += 1.0;
            glcm.at<double>(j, i) += 1.0;   // simétrica
        }
    }

    const double total = cv::sum(glcm)[0];
    if (total > 0) glcm /= total;
    return glcm;
}

// Extrae las 7 características Haralick de una GLCM normalizada.
// Referencia: Haralick, Shanmugam, Dinstein — IEEE Trans. SMC 1973.
inline GLCMFeatures extractGLCMFeatures(const cv::Mat& glcm)
{
    const int N = glcm.rows;

    // Distribuciones marginales px[i] y py[j]
    std::vector<double> px(N, 0.0), py(N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            px[i] += glcm.at<double>(i, j);
            py[j] += glcm.at<double>(i, j);
        }

    double mu_x = 0.0, mu_y = 0.0;
    for (int k = 0; k < N; ++k) {
        mu_x += k * px[k];
        mu_y += k * py[k];
    }

    double sigma_x = 0.0, sigma_y = 0.0;
    for (int k = 0; k < N; ++k) {
        sigma_x += (k - mu_x) * (k - mu_x) * px[k];
        sigma_y += (k - mu_y) * (k - mu_y) * py[k];
    }
    sigma_x = std::sqrt(sigma_x);
    sigma_y = std::sqrt(sigma_y);

    GLCMFeatures f{};

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double p = glcm.at<double>(i, j);
            if (p < 1e-15) continue;
            const double d = (double)(i - j);

            f.energy      += p * p;
            f.contrast    += d * d * p;
            f.homogeneity += p / (1.0 + std::abs(d));
            f.idf         += p / (1.0 + d * d);
            f.entropy     -= p * std::log2(p);
            f.variance    += (i - mu_x) * (i - mu_x) * p;

            if (sigma_x > 1e-10 && sigma_y > 1e-10)
                f.correlation += (i - mu_x) * (j - mu_y) * p / (sigma_x * sigma_y);
        }
    }
    return f;
}

// ============================================================================
// LBP — LOCAL BINARY PATTERN
// ============================================================================

// Genera la imagen LBP de 8 bits para toda la imagen gray.
// Cada píxel (r,c) con r∈[1,H-2], c∈[1,W-2] se compara con sus 8 vecinos
// en sentido horario (comenzando desde arriba-derecha):
//   bit k = 1 si vecino_k >= centro, 0 si no.
// La imagen LBP resultante tiene tamaño (H-2) × (W-2).
inline cv::Mat computeLBPImage(const cv::Mat& gray)
{
    CV_Assert(gray.type() == CV_8UC1);
    cv::Mat lbp = cv::Mat::zeros(gray.rows - 2, gray.cols - 2, CV_8UC1);

    // Desplazamientos de los 8 vecinos en sentido horario desde (-1,+1)
    constexpr int DR[8] = {-1,-1,-1, 0, 1, 1, 1, 0};
    constexpr int DC[8] = { 1, 0,-1,-1,-1, 0, 1, 1};

    for (int r = 1; r < gray.rows - 1; ++r) {
        for (int c = 1; c < gray.cols - 1; ++c) {
            const uchar center = gray.at<uchar>(r, c);
            uchar code = 0;
            for (int k = 0; k < 8; ++k)
                if (gray.at<uchar>(r + DR[k], c + DC[k]) >= center)
                    code |= static_cast<uchar>(1 << k);
            lbp.at<uchar>(r - 1, c - 1) = code;
        }
    }
    return lbp;
}

// ============================================================================
// ESCRITURA CSV
// ============================================================================

class CSVWriter
{
public:
    CSVWriter(const std::string& path, const std::vector<std::string>& header)
        : file_(path)
    {
        if (!file_.is_open())
            throw std::runtime_error("No se pudo abrir para escritura: " + path);

        for (size_t i = 0; i < header.size(); ++i) {
            if (i > 0) file_ << ',';
            file_ << header[i];
        }
        file_ << '\n';
        std::cout << "CSV abierto: " << path
                  << " (" << header.size() << " columnas)" << std::endl;
    }

    // Escribe una fila: etiqueta seguida de valores numéricos
    void writeRow(const std::string& label, const std::vector<double>& values)
    {
        file_ << label;
        for (double v : values) file_ << ',' << v;
        file_ << '\n';
    }

    ~CSVWriter() { if (file_.is_open()) file_.close(); }

private:
    std::ofstream file_;
};

// ============================================================================
// UTILIDADES COMUNES DE EXTRACCIÓN
// ============================================================================

// Calcula mean y std de un mapa de características float (CV_32F o CV_64F)
inline void mapStats(const cv::Mat& fmap, double& mean_val, double& std_val)
{
    cv::Scalar mean, std;
    cv::meanStdDev(fmap, mean, std);
    mean_val = mean[0];
    std_val  = std[0];
}

// Construye el prefijo de nombre de columna CSV para un extractor y ventana
inline std::string colPrefix(const std::string& extractor, int window_size)
{
    return extractor + "_w" + std::to_string(window_size) + "_";
}
