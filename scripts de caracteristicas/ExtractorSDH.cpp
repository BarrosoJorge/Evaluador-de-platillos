/*  ExtractorSDH.cpp
 *
 *  Extractor de histogramas de sumas y diferencias (Sum and Difference Histograms).
 *
 *  Referencia: Unser & Eden — "Multiresolution feature extraction and selection
 *              for texture segmentation", IEEE PAMI 1989.
 *              EPráctico1 — Parte 1.
 *
 *  Para cada imagen y tamaño de ventana w ∈ {3,5,...,25}:
 *    · Desliza ventana w×w con paso=1
 *    · Para cada ventana, itera sobre todos los pares de píxeles adyacentes
 *      (vecindad de 4 conectividad: horizontal + vertical, sin duplicados):
 *        s = (p1 + p2) / 2      [suma promedio, rango 0-255]
 *        d = |p1 - p2|          [diferencia absoluta, rango 0-255]
 *    · Construye histograma conjunto normalizado P_sd(s, d)
 *    · Extrae 7 características (EPráctico1):
 *        1. Media            — E[s] = Σ s · Ps(s)
 *        2. Varianza         — Var[s] = Σ (s−μs)² · Ps(s)
 *        3. Correlación      — Cov(s,d) / (σs · σd)
 *        4. Contraste        — E[d²] = Σ d² · Pd(d)
 *        5. Homogeneidad     — Σ Pd(d) / (1 + d)
 *        6. Clúster sombra   — Σ Σ (s+d−μs−μd)³ · P_sd(s,d)
 *        7. Clúster prominencia — Σ Σ (s+d−μs−μd)⁴ · P_sd(s,d)
 *
 *  Salida:
 *    · Mapas de características (PNG) en output_maps/
 *    · CSV con media y desviación estándar por mapa: features_sdh.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o extractor_sdh ExtractorSDH.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./extractor_sdh [ruta_preprocesadas] [ruta_salida_csv] [ruta_mapas]
 */

#include "Caracteristicas.h"

// ============================================================================
// CONFIGURACIÓN
// ============================================================================

static const std::string FEATURE_NAMES[] = {
    "Media", "Varianza", "Correlacion",
    "Contraste", "Homogeneidad", "ClusterSombra", "ClusterProminencia"
};
static constexpr int N_FEATURES = 7;
static constexpr int SDH_BINS   = 256;    // Bins para s y d

// ============================================================================
// HISTOGRAMA CONJUNTO SDH
// ============================================================================

// Construye el histograma conjunto P_sd[s][d] (normalizado) de una ventana.
// Pairs: vecinos horizontales (r,c)-(r,c+1) y verticales (r,c)-(r+1,c).
// s = floor((p1+p2)/2), d = |p1-p2|
inline std::vector<std::vector<double>>
computeSDHJoint(const cv::Mat& roi)
{
    // P_sd[s][d] — histograma 2D (bins: 0-255 para s, 0-255 para d)
    std::vector<std::vector<double>> P(SDH_BINS, std::vector<double>(SDH_BINS, 0.0));
    int count = 0;

    for (int r = 0; r < roi.rows; ++r) {
        for (int c = 0; c < roi.cols; ++c) {
            const int p1 = roi.at<uchar>(r, c);

            // Par horizontal
            if (c + 1 < roi.cols) {
                const int p2 = roi.at<uchar>(r, c + 1);
                const int s  = (p1 + p2) / 2;
                const int d  = std::abs(p1 - p2);
                P[s][d] += 1.0;
                ++count;
            }
            // Par vertical
            if (r + 1 < roi.rows) {
                const int p2 = roi.at<uchar>(r + 1, c);
                const int s  = (p1 + p2) / 2;
                const int d  = std::abs(p1 - p2);
                P[s][d] += 1.0;
                ++count;
            }
        }
    }

    // Normalizar
    if (count > 0)
        for (auto& row : P)
            for (auto& val : row) val /= count;

    return P;
}

// ============================================================================
// EXTRACCIÓN DE 7 CARACTERÍSTICAS SDH
// ============================================================================

// Extrae las 7 características del histograma conjunto P_sd.
inline std::vector<double> extractSDHFeatures(const std::vector<std::vector<double>>& P)
{
    // Distribuciones marginales
    std::vector<double> Ps(SDH_BINS, 0.0), Pd(SDH_BINS, 0.0);
    for (int s = 0; s < SDH_BINS; ++s)
        for (int d = 0; d < SDH_BINS; ++d) {
            Ps[s] += P[s][d];
            Pd[d] += P[s][d];
        }

    // Medias
    double mu_s = 0.0, mu_d = 0.0;
    for (int k = 0; k < SDH_BINS; ++k) {
        mu_s += k * Ps[k];
        mu_d += k * Pd[k];
    }

    // Varianzas y desviaciones estándar
    double var_s = 0.0, var_d = 0.0;
    for (int k = 0; k < SDH_BINS; ++k) {
        var_s += (k - mu_s) * (k - mu_s) * Ps[k];
        var_d += (k - mu_d) * (k - mu_d) * Pd[k];
    }
    const double sigma_s = std::sqrt(var_s);
    const double sigma_d = std::sqrt(var_d);

    // ── Calcular las 7 características ────────────────────────────────────────

    // 1. Media: E[s]
    const double media = mu_s;

    // 2. Varianza: Var[s]
    const double varianza = var_s;

    // 3. Contraste: E[d²] = Σ d² · Pd(d)
    double contraste = 0.0;
    for (int d = 0; d < SDH_BINS; ++d) contraste += (double)d * d * Pd[d];

    // 4. Homogeneidad: Σ Pd(d) / (1 + d)
    double homogeneidad = 0.0;
    for (int d = 0; d < SDH_BINS; ++d) homogeneidad += Pd[d] / (1.0 + d);

    // 5. Correlación: Cov(s,d) / (σs · σd)
    double cov_sd = 0.0;
    for (int s = 0; s < SDH_BINS; ++s)
        for (int d = 0; d < SDH_BINS; ++d)
            cov_sd += (s - mu_s) * (d - mu_d) * P[s][d];

    const double correlacion = (sigma_s > 1e-10 && sigma_d > 1e-10)
                               ? cov_sd / (sigma_s * sigma_d) : 0.0;

    // 6. Clúster sombra: Σ Σ (s+d−μs−μd)³ · P(s,d)
    double cluster_sombra = 0.0;
    for (int s = 0; s < SDH_BINS; ++s)
        for (int d = 0; d < SDH_BINS; ++d) {
            const double t = (s + d) - mu_s - mu_d;
            cluster_sombra += t * t * t * P[s][d];
        }

    // 7. Clúster prominencia: Σ Σ (s+d−μs−μd)⁴ · P(s,d)
    double cluster_prominencia = 0.0;
    for (int s = 0; s < SDH_BINS; ++s)
        for (int d = 0; d < SDH_BINS; ++d) {
            const double t = (s + d) - mu_s - mu_d;
            cluster_prominencia += t * t * t * t * P[s][d];
        }

    return {media, varianza, correlacion, contraste,
            homogeneidad, cluster_sombra, cluster_prominencia};
}

// ============================================================================
// MAPAS DE CARACTERÍSTICAS
// ============================================================================

// Genera N_FEATURES mapas (uno por característica) para una imagen y un tamaño de ventana.
std::vector<cv::Mat> computeFeatureMaps(const cv::Mat& gray, int w)
{
    const int out_rows = gray.rows - w + 1;
    const int out_cols = gray.cols - w + 1;

    std::vector<cv::Mat> maps(N_FEATURES,
        cv::Mat::zeros(out_rows, out_cols, CV_64F));

    slideWindow(gray, w, [&](const cv::Mat& roi, int r, int c) {
        const auto joint = computeSDHJoint(roi);
        const auto feat  = extractSDHFeatures(joint);
        for (int k = 0; k < N_FEATURES; ++k)
            maps[k].at<double>(r, c) = feat[k];
    });

    return maps;
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char** argv)
{
    std::string input_dir  = "/mnt/d/Data/Preprocesadas";
    std::string output_csv = "features_sdh.csv";
    std::string maps_dir   = "/mnt/d/Data/FeatureMaps/SDH";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_csv = argv[2];
    if (argc >= 4) maps_dir   = argv[3];

    std::cout << "Extractor SDH — Histogramas de Sumas y Diferencias" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    auto images = loadImages(input_dir);
    if (images.empty()) {
        std::cerr << "No se encontraron imágenes en: " << input_dir << std::endl;
        return 1;
    }

    // ── Cabecera CSV ──────────────────────────────────────────────────────────
    // SDH_w3_Media_mean, SDH_w3_Media_std, ..., SDH_w25_ClusterProminencia_std
    std::vector<std::string> header = {"label", "filename"};
    for (int w : WINDOW_SIZES) {
        const std::string prefix = colPrefix("SDH", w);
        for (const auto& name : FEATURE_NAMES) {
            header.push_back(prefix + name + "_mean");
            header.push_back(prefix + name + "_std");
        }
    }

    CSVWriter csv(output_csv, header);

    // ── Procesar cada imagen ──────────────────────────────────────────────────
    for (const auto& img : images) {
        std::cout << "\nProcesando: " << img->filename << std::endl;

        std::vector<double> row_values;
        const std::string img_stem = fs::path(img->filename).stem().string();

        for (int w : WINDOW_SIZES) {
            std::cout << "  Ventana " << w << "x" << w << "..." << std::flush;

            auto maps = computeFeatureMaps(img->gray, w);

            for (int k = 0; k < N_FEATURES; ++k) {
                double mean_val, std_val;
                mapStats(maps[k], mean_val, std_val);
                row_values.push_back(mean_val);
                row_values.push_back(std_val);

                saveFeatureMap(maps[k], maps_dir, img_stem,
                               "SDH_" + std::string(FEATURE_NAMES[k]), w);
            }
            std::cout << " OK" << std::endl;
        }

        csv.writeRow(img->metadata.label() + "," + img->filename, row_values);
    }

    std::cout << "\nExtractor SDH completado." << std::endl;
    std::cout << "CSV: " << output_csv << std::endl;
    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o extractor_sdh ExtractorSDH.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./extractor_sdh [ruta_preprocesadas] [csv_salida] [ruta_mapas]
//   ./extractor_sdh /mnt/d/Data/Preprocesadas features_sdh.csv /mnt/d/Data/FeatureMaps/SDH
