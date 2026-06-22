/*  ExtractorRaw.cpp
 *
 *  Extractor de características sobre ventanas de píxeles raw (imagen original).
 *
 *  Para cada imagen y cada tamaño de ventana w ∈ {3,5,...,25}:
 *    · Desliza una ventana w×w con paso=1 sobre la imagen en escala de grises
 *    · Calcula 7 características por ventana usando el histograma de intensidades:
 *        1. Energía        — Σ p[i]²
 *        2. Contraste      — Σ (i−μ)² · p[i]  (varianza)
 *        3. Correlación    — (E[i²] − μ²) / σ²
 *        4. Homogeneidad   — Σ p[i] / (1 + |i−μ|)
 *        5. IDF            — Σ p[i] / (1 + (i−μ)²)
 *        6. Entropía       — −Σ p[i] · log₂(p[i])
 *        7. Varianza       — Σ (i−μ)² · p[i]
 *
 *  Salida:
 *    · Mapas de características (PNG por feature por ventana) en output_maps/
 *    · CSV con media y desviación estándar de cada mapa: features_raw.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o extractor_raw ExtractorRaw.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./extractor_raw [ruta_preprocesadas] [ruta_salida_csv] [ruta_mapas]
 */

#include "Caracteristicas.h"

// ============================================================================
// CONFIGURACIÓN
// ============================================================================

static const std::string FEATURE_NAMES[] = {
    "Energia", "Contraste", "Correlacion",
    "Homogeneidad", "IDF", "Entropia", "Varianza"
};
static constexpr int N_FEATURES = 7;

// ============================================================================
// FUNCIONES DE EXTRACCIÓN
// ============================================================================

// Calcula las 7 características Raw para un ROI.
// Retorna el vector [energía, contraste, correlación, homogeneidad, IDF, entropía, varianza]
inline std::vector<double> extractRawFeatures(const cv::Mat& roi)
{
    const auto hist = computeHist1D(roi);
    const double mu  = histMean(hist);
    const double var = histVariance(hist, mu);

    return {
        histEnergy(hist),
        var,                            // Contraste ≈ varianza para señal 1D
        histCorrelation(hist),
        histHomogeneity(hist, mu),
        histIDF(hist, mu),
        histEntropy(hist),
        var                             // Varianza
    };
}

// Para una imagen completa y un tamaño de ventana w, genera N_FEATURES mapas de
// características (uno por feature), cada uno de tamaño (H-w+1)×(W-w+1).
std::vector<cv::Mat> computeFeatureMaps(const cv::Mat& gray, int w)
{
    const int out_rows = gray.rows - w + 1;
    const int out_cols = gray.cols - w + 1;

    std::vector<cv::Mat> maps(N_FEATURES);
    for (int k = 0; k < N_FEATURES; ++k)
        maps[k] = cv::Mat::zeros(out_rows, out_cols, CV_64F);

    slideWindow(gray, w, [&](const cv::Mat& roi, int r, int c) {
        const auto feat = extractRawFeatures(roi);
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
    std::string input_dir  = "Data/Preprocesadas";
    std::string output_csv = "Data/Features/features_raw.csv";
    std::string maps_dir   = "Data/Features/maps/Raw";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_csv = argv[2];
    if (argc >= 4) maps_dir   = argv[3];

    std::cout << "Extractor Raw — ventaneo de píxeles" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // ── Cargar imágenes ──────────────────────────────────────────────────────
    auto images = loadImages(input_dir);
    if (images.empty()) {
        std::cerr << "No se encontraron imágenes en: " << input_dir << std::endl;
        return 1;
    }

    // ── Construir cabecera CSV ────────────────────────────────────────────────
    // Columnas: label, Raw_w3_Energia_mean, Raw_w3_Energia_std, ..., Raw_w25_Varianza_std
    std::vector<std::string> header = {"label", "filename"};
    for (int w : WINDOW_SIZES) {
        const std::string prefix = colPrefix("Raw", w);
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
                               "Raw_" + std::string(FEATURE_NAMES[k]), w);
            }
            std::cout << " OK" << std::endl;
        }

        csv.writeRow(img->metadata.label() + "," + img->filename, row_values);
    }

    std::cout << "\nExtractor Raw completado." << std::endl;
    std::cout << "CSV: " << output_csv << std::endl;
    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o extractor_raw ExtractorRaw.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./extractor_raw [ruta_preprocesadas] [csv_salida] [ruta_mapas]
//   ./extractor_raw /mnt/d/Data/Preprocesadas features_raw.csv /mnt/d/Data/FeatureMaps/Raw
