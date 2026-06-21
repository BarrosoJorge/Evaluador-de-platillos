/*  ExtractorGLCM.cpp
 *
 *  Extractor de características basado en Matrices de Co-ocurrencia de Nivel de Gris (GLCM).
 *
 *  Referencia: Haralick, Shanmugam, Dinstein — "Textural features for image classification",
 *              IEEE Trans. SMC 1973.  EPráctico1 — Parte 2.
 *
 *  Configuración (EPráctico1):
 *    · Distancias: d = 1, 3, 7
 *    · Ángulos:    θ = 0°, 45°, 90°, 135°
 *    · Total de matrices GLCM por ventana: 12 (3 distancias × 4 ángulos)
 *    · Características por GLCM (Haralick):
 *        1. Energía        — Σ Σ P(i,j)²
 *        2. Contraste      — Σ Σ (i−j)² · P(i,j)
 *        3. Correlación    — Σ Σ (i−μx)(j−μy)·P(i,j) / (σx·σy)
 *        4. Homogeneidad   — Σ Σ P(i,j) / (1 + |i−j|)
 *        5. IDF            — Σ Σ P(i,j) / (1 + (i−j)²)
 *        6. Entropía       — −Σ Σ P(i,j) · log₂(P(i,j))
 *        7. Varianza       — Σ Σ (i−μx)² · P(i,j)
 *    · Total de características por ventana: 12 × 7 = 84
 *
 *  Para cada imagen y tamaño de ventana w ∈ {3,5,...,25}:
 *    Genera 84 mapas de características de tamaño (H−w+1)×(W−w+1).
 *    El CSV guarda media y std de cada mapa → 168 columnas por ventana.
 *
 *  Salida:
 *    · Mapas PNG en maps_dir/
 *    · CSV con media+std por mapa: features_glcm.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o extractor_glcm ExtractorGLCM.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./extractor_glcm [ruta_preprocesadas] [csv_salida] [ruta_mapas]
 */

#include "Caracteristicas.h"

// ============================================================================
// CONFIGURACIÓN
// ============================================================================

static constexpr int N_DISTANCES = 3;
static constexpr int N_ANGLES    = 4;
static constexpr int N_HARALICK  = 7;
static constexpr int N_GLCM      = N_DISTANCES * N_ANGLES;     // 12
static constexpr int N_FEATURES  = N_GLCM * N_HARALICK;        // 84

// ============================================================================
// EXTRACCIÓN DE CARACTERÍSTICAS GLCM PARA UNA VENTANA
// ============================================================================

// Calcula las 84 características (12 GLCMs × 7 Haralick) para una ventana.
// Retorna un vector flat: [d1_a0_feat0..6, d1_a45_feat0..6, ..., d7_a135_feat0..6]
inline std::vector<double> extractGLCMWindowFeatures(const cv::Mat& roi)
{
    std::vector<double> features;
    features.reserve(N_FEATURES);

    for (int dist : GLCM_DISTANCES) {
        for (double angle : GLCM_ANGLES) {
            const cv::Mat glcm = computeGLCM(roi, dist, angle, GLCM_LEVELS);
            const GLCMFeatures f = extractGLCMFeatures(glcm);
            const auto v = f.toVector();
            features.insert(features.end(), v.begin(), v.end());
        }
    }
    return features;
}

// ============================================================================
// MAPAS DE CARACTERÍSTICAS
// ============================================================================

// Para una imagen completa y un tamaño de ventana w, genera N_FEATURES mapas.
// maps[k] corresponde a la k-ésima característica del vector flat de 84 elementos.
std::vector<cv::Mat> computeFeatureMaps(const cv::Mat& gray, int w)
{
    const int out_rows = gray.rows - w + 1;
    const int out_cols = gray.cols - w + 1;

    std::vector<cv::Mat> maps(N_FEATURES,
        cv::Mat::zeros(out_rows, out_cols, CV_64F));

    slideWindow(gray, w, [&](const cv::Mat& roi, int r, int c) {
        const auto feat = extractGLCMWindowFeatures(roi);
        for (int k = 0; k < N_FEATURES; ++k)
            maps[k].at<double>(r, c) = feat[k];
    });

    return maps;
}

// Construye el nombre de la k-ésima característica: GLCM_d{dist}_a{angle}_{feature}
inline std::string featureName(int idx)
{
    const int glcm_idx  = idx / N_HARALICK;
    const int feat_idx  = idx % N_HARALICK;
    const int dist_idx  = glcm_idx / N_ANGLES;
    const int angle_idx = glcm_idx % N_ANGLES;

    const int    dist  = GLCM_DISTANCES[dist_idx];
    const int    angle = static_cast<int>(GLCM_ANGLES[angle_idx]);
    const auto   fname = GLCMFeatures::names()[feat_idx];

    return "GLCM_d" + std::to_string(dist) + "_a" + std::to_string(angle) + "_" + fname;
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char** argv)
{
    std::string input_dir  = "/mnt/d/Data/Preprocesadas";
    std::string output_csv = "features_glcm.csv";
    std::string maps_dir   = "/mnt/d/Data/FeatureMaps/GLCM";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_csv = argv[2];
    if (argc >= 4) maps_dir   = argv[3];

    std::cout << "Extractor GLCM — Matrices de Co-ocurrencia" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Distancias: d=1,3,7 | Ángulos: 0,45,90,135° | "
              << N_FEATURES << " características/ventana" << std::endl;

    auto images = loadImages(input_dir);
    if (images.empty()) {
        std::cerr << "No se encontraron imágenes en: " << input_dir << std::endl;
        return 1;
    }

    // ── Cabecera CSV ──────────────────────────────────────────────────────────
    // GLCM_w3_d1_a0_Energia_mean, GLCM_w3_d1_a0_Energia_std, ...
    std::vector<std::string> header = {"label", "filename"};
    for (int w : WINDOW_SIZES) {
        const std::string w_prefix = "GLCM_w" + std::to_string(w) + "_";
        for (int k = 0; k < N_FEATURES; ++k) {
            const std::string fname = w_prefix + featureName(k);
            header.push_back(fname + "_mean");
            header.push_back(fname + "_std");
        }
    }

    CSVWriter csv(output_csv, header);

    // ── Procesar cada imagen ──────────────────────────────────────────────────
    for (const auto& img : images) {
        std::cout << "\nProcesando: " << img->filename << std::endl;

        std::vector<double> row_values;
        const std::string img_stem = fs::path(img->filename).stem().string();

        for (int w : WINDOW_SIZES) {
            std::cout << "  Ventana " << w << "x" << w
                      << " (84 features × " << (img->gray.rows - w + 1)
                      << "×" << (img->gray.cols - w + 1) << " posiciones)..."
                      << std::flush;

            auto maps = computeFeatureMaps(img->gray, w);

            for (int k = 0; k < N_FEATURES; ++k) {
                double mean_val, std_val;
                mapStats(maps[k], mean_val, std_val);
                row_values.push_back(mean_val);
                row_values.push_back(std_val);

                // Solo guardar mapas para la primera distancia/ángulo por limpieza
                // (descomentar la línea de abajo para guardar los 84 mapas por ventana)
                // saveFeatureMap(maps[k], maps_dir, img_stem, featureName(k), w);
            }
            // Guardar los primeros 7 mapas (d=1, θ=0°) como ejemplo visual
            const auto maps_d1_a0 = computeFeatureMaps(img->gray, w);
            for (int k = 0; k < N_HARALICK; ++k)
                saveFeatureMap(maps_d1_a0[k], maps_dir, img_stem,
                               "GLCM_d1_a0_" + GLCMFeatures::names()[k], w);

            std::cout << " OK" << std::endl;
        }

        csv.writeRow(img->metadata.label() + "," + img->filename, row_values);
    }

    std::cout << "\nExtractor GLCM completado." << std::endl;
    std::cout << "CSV: " << output_csv << std::endl;
    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o extractor_glcm ExtractorGLCM.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./extractor_glcm [ruta_preprocesadas] [csv_salida] [ruta_mapas]
//   ./extractor_glcm /mnt/d/Data/Preprocesadas features_glcm.csv /mnt/d/Data/FeatureMaps/GLCM
