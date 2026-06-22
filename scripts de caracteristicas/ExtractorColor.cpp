/*  ExtractorColor.cpp  [BONUS]
 *
 *  Extractor de características de color en espacios HSV y CIE L*a*b*.
 *
 *  Motivación para la evaluación de platillos:
 *    · El color es uno de los indicadores visuales más inmediatos de la calidad de un plato.
 *    · Un chef distribuye colores de forma balanceada y saturada.
 *    · El espacio HSV captura matiz, saturación y brillo por separado.
 *    · El espacio Lab separa luminosidad de cromaticidad (más cercano a la percepción humana).
 *
 *  Para cada imagen y tamaño de ventana w ∈ {3,5,...,25}:
 *    Desliza ventana w×w con paso=1 y extrae por cada canal:
 *
 *    HSV (canales H, S, V):
 *      · Media, Desviación estándar, Entropía, IDF (del histograma del canal)
 *
 *    Lab (canales L*, a*, b*):
 *      · Media, Desviación estándar, Entropía, IDF
 *
 *    Características adicionales (por ventana):
 *      · Varianza total de color = var(H) + var(S) + var(V)
 *      · Saturación media (canal S de HSV)
 *      · Contraste de luminosidad (var del canal L*)
 *
 *    Total de características por ventana:
 *      3 canales HSV × 4 stats + 3 canales Lab × 4 stats + 3 globales = 27
 *
 *  Salida:
 *    · Imágenes de los canales HSV y Lab (PNG) para inspección visual
 *    · Mapas de características por canal (PNG)
 *    · CSV con media+std por mapa: features_color.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o extractor_color ExtractorColor.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./extractor_color [ruta_preprocesadas] [csv_salida] [ruta_mapas]
 */

#include "Caracteristicas.h"

// ============================================================================
// CONFIGURACIÓN
// ============================================================================

static constexpr int N_CHANNELS = 6;   // H, S, V, L*, a*, b*
static constexpr int N_STATS    = 4;   // mean, std, entropy, IDF
static constexpr int N_GLOBAL   = 3;   // varianza total color, satMean, contrasteL
static constexpr int N_FEATURES = N_CHANNELS * N_STATS + N_GLOBAL;   // 27

static const std::string CHANNEL_NAMES[] = {"H","S","V","L","a","b"};
static const std::string STAT_NAMES[]    = {"mean","std","entropy","idf"};

// ============================================================================
// CONVERSIÓN DE CANALES
// ============================================================================

// Convierte la imagen BGR a HSV y CIE Lab de 32 bits (float) normalizados a [0,255].
// Devuelve: channels[0..2] = H(0-179→0-255), S, V y channels[3..5] = L(0-100→0-255), a, b
inline std::vector<cv::Mat> extractColorChannels(const cv::Mat& bgr)
{
    cv::Mat hsv_f, lab_f;
    cv::Mat hsv, lab;

    // HSV: H∈[0,179], S∈[0,255], V∈[0,255]
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // Lab: float — L∈[0,100], a∈[-127,127], b∈[-127,127]
    cv::Mat bgr_f;
    bgr.convertTo(bgr_f, CV_32F, 1.0 / 255.0);
    cv::cvtColor(bgr_f, lab_f, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> hsv_ch(3), lab_ch(3);
    cv::split(hsv,   hsv_ch);
    cv::split(lab_f, lab_ch);

    // Normalizar H al rango [0,255] (OpenCV: H ∈ [0,179])
    hsv_ch[0].convertTo(hsv_ch[0], CV_8U,  255.0 / 179.0);

    // Normalizar L∈[0,100] → [0,255]
    cv::Mat L_u8;
    lab_ch[0].convertTo(L_u8, CV_8U, 255.0 / 100.0);

    // Normalizar a∈[-127,127] → [0,255]
    cv::Mat a_u8, b_u8;
    lab_ch[1].convertTo(a_u8, CV_8U, 1.0, 127.0);
    lab_ch[2].convertTo(b_u8, CV_8U, 1.0, 127.0);

    return { hsv_ch[0], hsv_ch[1], hsv_ch[2], L_u8, a_u8, b_u8 };
}

// ============================================================================
// CARACTERÍSTICAS POR VENTANA
// ============================================================================

// Extrae los N_FEATURES valores de características para una ventana multi-canal.
// channels[0..5] deben estar en [0,255] uint8.
inline std::vector<double> extractColorFeatures(
    const std::vector<cv::Mat>& channels, int r, int c, int w)
{
    std::vector<double> feat;
    feat.reserve(N_FEATURES);

    std::vector<double> vars;    // Para varianza total de color (solo HSV)

    for (int ch = 0; ch < N_CHANNELS; ++ch) {
        const cv::Mat roi = channels[ch](cv::Rect(c, r, w, w));
        const auto hist   = computeHist1D(roi);
        const double mu   = histMean(hist);
        const double var  = histVariance(hist, mu);

        feat.push_back(mu);                      // mean
        feat.push_back(std::sqrt(var));          // std
        feat.push_back(histEntropy(hist));       // entropy
        feat.push_back(histIDF(hist, mu));       // idf

        if (ch < 3) vars.push_back(var);        // H,S,V para varianza total
    }

    // Varianza total de color (suma de varianzas HSV) — mide diversidad cromática
    feat.push_back(vars[0] + vars[1] + vars[2]);

    // Saturación media: promedio del canal S (índice 1) dentro de la ventana
    {
        const cv::Mat roi_s = channels[1](cv::Rect(c, r, w, w));
        const auto hist_s   = computeHist1D(roi_s);
        feat.push_back(histMean(hist_s));
    }

    // Contraste de luminosidad: varianza del canal L* (índice 3)
    {
        const cv::Mat roi_l = channels[3](cv::Rect(c, r, w, w));
        const auto hist_l   = computeHist1D(roi_l);
        const double mu_l   = histMean(hist_l);
        feat.push_back(histVariance(hist_l, mu_l));
    }

    return feat;
}

// ============================================================================
// MAPAS DE CARACTERÍSTICAS
// ============================================================================

// Genera N_FEATURES mapas para una imagen (todos sus canales) y ventana w.
std::vector<cv::Mat> computeFeatureMaps(
    const std::vector<cv::Mat>& channels, int w, int img_rows, int img_cols)
{
    const int out_rows = img_rows - w + 1;
    const int out_cols = img_cols - w + 1;

    std::vector<cv::Mat> maps(N_FEATURES);
    for (int k = 0; k < N_FEATURES; ++k)
        maps[k] = cv::Mat::zeros(out_rows, out_cols, CV_64F);

    for (int r = 0; r < out_rows; ++r) {
        for (int c = 0; c < out_cols; ++c) {
            const auto feat = extractColorFeatures(channels, r, c, w);
            for (int k = 0; k < N_FEATURES; ++k)
                maps[k].at<double>(r, c) = feat[k];
        }
    }
    return maps;
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char** argv)
{
    std::string input_dir  = "Data/Preprocesadas";
    std::string output_csv = "Data/Features/features_color.csv";
    std::string maps_dir   = "Data/Features/maps/Color";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_csv = argv[2];
    if (argc >= 4) maps_dir   = argv[3];

    std::cout << "Extractor Color — HSV + CIE Lab" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Canales: H,S,V (HSV) + L*,a*,b* (Lab) | "
              << N_FEATURES << " características/ventana" << std::endl;

    auto images = loadImages(input_dir);
    if (images.empty()) {
        std::cerr << "No se encontraron imágenes en: " << input_dir << std::endl;
        return 1;
    }

    // ── Cabecera CSV ──────────────────────────────────────────────────────────
    // Color_w3_H_mean_mean, Color_w3_H_mean_std, ..., Color_w25_contrasteL_std
    std::vector<std::string> header = {"label", "filename"};
    for (int w : WINDOW_SIZES) {
        const std::string prefix = colPrefix("Color", w);
        // 6 canales × 4 estadísticas
        for (const auto& ch : CHANNEL_NAMES)
            for (const auto& st : STAT_NAMES) {
                header.push_back(prefix + ch + "_" + st + "_mean");
                header.push_back(prefix + ch + "_" + st + "_std");
            }
        // 3 globales
        header.push_back(prefix + "var_total_color_mean");
        header.push_back(prefix + "var_total_color_std");
        header.push_back(prefix + "saturacion_media_mean");
        header.push_back(prefix + "saturacion_media_std");
        header.push_back(prefix + "contraste_luminosidad_mean");
        header.push_back(prefix + "contraste_luminosidad_std");
    }

    CSVWriter csv(output_csv, header);

    // ── Procesar cada imagen ──────────────────────────────────────────────────
    for (const auto& img : images) {
        std::cout << "\nProcesando: " << img->filename << std::endl;

        const std::string img_stem = fs::path(img->filename).stem().string();

        // Extraer y guardar los 6 canales de color
        auto channels = extractColorChannels(img->color);

        {
            fs::path dir = fs::path(maps_dir) / img_stem;
            try { fs::create_directories(dir); } catch (...) {}

            const std::string ch_names[] = {"H","S","V","L","a","b"};
            for (int k = 0; k < N_CHANNELS; ++k)
                cv::imwrite((dir / (std::string(ch_names[k]) + "_canal.png")).string(),
                            channels[k]);
        }

        std::vector<double> row_values;

        for (int w : WINDOW_SIZES) {
            std::cout << "  Ventana " << w << "x" << w << "..." << std::flush;

            auto maps = computeFeatureMaps(channels, w, img->color.rows, img->color.cols);

            // Nombres de los N_FEATURES mapas (mismo orden que extractColorFeatures)
            std::vector<std::string> map_names;
            for (const auto& ch : CHANNEL_NAMES)
                for (const auto& st : STAT_NAMES)
                    map_names.push_back(ch + "_" + st);
            map_names.push_back("var_total_color");
            map_names.push_back("saturacion_media");
            map_names.push_back("contraste_luminosidad");

            for (int k = 0; k < N_FEATURES; ++k) {
                double mean_val, std_val;
                mapStats(maps[k], mean_val, std_val);
                row_values.push_back(mean_val);
                row_values.push_back(std_val);

                saveFeatureMap(maps[k], maps_dir, img_stem,
                               "Color_" + map_names[k], w);
            }
            std::cout << " OK" << std::endl;
        }

        csv.writeRow(img->metadata.label() + "," + img->filename, row_values);
    }

    std::cout << "\nExtractor Color completado." << std::endl;
    std::cout << "CSV: " << output_csv << std::endl;
    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o extractor_color ExtractorColor.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./extractor_color [ruta_preprocesadas] [csv_salida] [ruta_mapas]
//   ./extractor_color /mnt/d/Data/Preprocesadas features_color.csv /mnt/d/Data/FeatureMaps/Color
