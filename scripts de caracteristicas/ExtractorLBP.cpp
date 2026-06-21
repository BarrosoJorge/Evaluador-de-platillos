/*  ExtractorLBP.cpp
 *
 *  Extractor de características basado en Local Binary Pattern (LBP).
 *
 *  Referencia: Ojala, Pietikäinen, Harwood — "A comparative study of texture measures
 *              with classification based on featured distributions", Pattern Recognition 1996.
 *              EPráctico1 — Parte 3.
 *
 *  Pipeline:
 *    1. Genera la imagen LBP de la imagen en escala de grises:
 *         Para cada píxel (r,c): compara con sus 8 vecinos (sentido horario).
 *         LBP[r,c] = código de 8 bits (vecino >= centro → bit 1, si no → bit 0).
 *         La imagen LBP tiene tamaño (H−2) × (W−2).
 *
 *    2. Para cada tamaño de ventana w ∈ {3,5,...,25}:
 *         Desliza ventana w×w con paso=1 sobre la imagen LBP.
 *         Calcula 5 características por ventana (EPráctico1 Parte 3):
 *           1. Media          — E[LBP]
 *           2. Varianza       — Var[LBP]
 *           3. Correlación    — (E[i²] − μ²) / σ²  (autocorrelación normalizada)
 *           4. Contraste      — Var[LBP]  (segundo momento central)
 *           5. Homogeneidad   — Σ p[i] / (1 + |i−μ|)
 *         El mapa resultante tiene tamaño (H_lbp−w+1) × (W_lbp−w+1).
 *
 *  Salida:
 *    · Imagen LBP de cada imagen de entrada (PNG) en maps_dir/<stem>/lbp.png
 *    · Mapas de características (PNG) en maps_dir/<stem>/
 *    · CSV con media+std por mapa: features_lbp.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o extractor_lbp ExtractorLBP.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./extractor_lbp [ruta_preprocesadas] [csv_salida] [ruta_mapas]
 */

#include "Caracteristicas.h"

// ============================================================================
// CONFIGURACIÓN
// ============================================================================

static const std::string FEATURE_NAMES[] = {
    "Media", "Varianza", "Correlacion", "Contraste", "Homogeneidad"
};
static constexpr int N_FEATURES = 5;

// ============================================================================
// EXTRACCIÓN DE CARACTERÍSTICAS LBP POR VENTANA
// ============================================================================

// Calcula las 5 características para una ventana de la imagen LBP.
// El ROI ya contiene valores LBP (uint8, 0-255).
inline std::vector<double> extractLBPFeatures(const cv::Mat& lbp_roi)
{
    const auto hist = computeHist1D(lbp_roi);   // 256 bins, normalizado
    const double mu  = histMean(hist);
    const double var = histVariance(hist, mu);

    return {
        mu,                              // 1. Media
        var,                             // 2. Varianza
        histCorrelation(hist),           // 3. Correlación
        var,                             // 4. Contraste ≈ varianza para histograma 1D
        histHomogeneity(hist, mu)        // 5. Homogeneidad
    };
}

// Genera N_FEATURES mapas de características de la imagen LBP para ventana w.
std::vector<cv::Mat> computeFeatureMaps(const cv::Mat& lbp_image, int w)
{
    const int out_rows = lbp_image.rows - w + 1;
    const int out_cols = lbp_image.cols - w + 1;

    std::vector<cv::Mat> maps(N_FEATURES,
        cv::Mat::zeros(out_rows, out_cols, CV_64F));

    slideWindow(lbp_image, w, [&](const cv::Mat& roi, int r, int c) {
        const auto feat = extractLBPFeatures(roi);
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
    std::string output_csv = "features_lbp.csv";
    std::string maps_dir   = "/mnt/d/Data/FeatureMaps/LBP";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_csv = argv[2];
    if (argc >= 4) maps_dir   = argv[3];

    std::cout << "Extractor LBP — Local Binary Pattern" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    auto images = loadImages(input_dir);
    if (images.empty()) {
        std::cerr << "No se encontraron imágenes en: " << input_dir << std::endl;
        return 1;
    }

    // ── Cabecera CSV ──────────────────────────────────────────────────────────
    // LBP_w3_Media_mean, LBP_w3_Media_std, ..., LBP_w25_Homogeneidad_std
    std::vector<std::string> header = {"label", "filename"};
    for (int w : WINDOW_SIZES) {
        const std::string prefix = colPrefix("LBP", w);
        for (const auto& name : FEATURE_NAMES) {
            header.push_back(prefix + name + "_mean");
            header.push_back(prefix + name + "_std");
        }
    }

    CSVWriter csv(output_csv, header);

    // ── Procesar cada imagen ──────────────────────────────────────────────────
    for (const auto& img : images) {
        std::cout << "\nProcesando: " << img->filename << std::endl;

        const std::string img_stem = fs::path(img->filename).stem().string();

        // Paso 1: generar imagen LBP
        const cv::Mat lbp = computeLBPImage(img->gray);

        // Guardar imagen LBP como referencia visual
        {
            fs::path dir = fs::path(maps_dir) / img_stem;
            try { fs::create_directories(dir); } catch (...) {}
            cv::imwrite((dir / "lbp.png").string(), lbp);
        }

        std::cout << "  Imagen LBP: " << lbp.cols << "×" << lbp.rows
                  << " px (original -2 en cada dimensión)" << std::endl;

        // Paso 2: ventaneo sobre la imagen LBP
        std::vector<double> row_values;

        for (int w : WINDOW_SIZES) {
            // Verificar que la imagen LBP es suficientemente grande para este tamaño de ventana
            if (lbp.rows < w || lbp.cols < w) {
                std::cerr << "  Ventana " << w << "x" << w
                          << " mayor que la imagen LBP — omitida." << std::endl;
                for (int k = 0; k < N_FEATURES; ++k) {
                    row_values.push_back(0.0);
                    row_values.push_back(0.0);
                }
                continue;
            }

            std::cout << "  Ventana LBP " << w << "x" << w
                      << " → mapa " << (lbp.rows - w + 1)
                      << "×" << (lbp.cols - w + 1) << "..." << std::flush;

            auto maps = computeFeatureMaps(lbp, w);

            for (int k = 0; k < N_FEATURES; ++k) {
                double mean_val, std_val;
                mapStats(maps[k], mean_val, std_val);
                row_values.push_back(mean_val);
                row_values.push_back(std_val);

                saveFeatureMap(maps[k], maps_dir, img_stem,
                               "LBP_" + std::string(FEATURE_NAMES[k]), w);
            }
            std::cout << " OK" << std::endl;
        }

        csv.writeRow(img->metadata.label() + "," + img->filename, row_values);
    }

    std::cout << "\nExtractor LBP completado." << std::endl;
    std::cout << "CSV: " << output_csv << std::endl;
    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o extractor_lbp ExtractorLBP.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./extractor_lbp [ruta_preprocesadas] [csv_salida] [ruta_mapas]
//   ./extractor_lbp /mnt/d/Data/Preprocesadas features_lbp.csv /mnt/d/Data/FeatureMaps/LBP
