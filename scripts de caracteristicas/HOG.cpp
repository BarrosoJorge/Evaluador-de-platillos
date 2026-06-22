/*  HOG.cpp  [BONUS]
 *
 *  Extractor de características HOG — Histogram of Oriented Gradients.
 *
 *  Referencia: Dalal & Triggs — "Histograms of Oriented Gradients for Human Detection",
 *              CVPR 2005.
 *
 *  HOG es útil para la evaluación de platillos porque captura la distribución espacial
 *  de bordes y texturas: filetes bien marcados, capas diferenciadas, presentación
 *  estructurada del emplatado generan patrones de gradiente distintos a un emplatado
 *  descuidado.
 *
 *  Modo de operación:
 *    · Calcula el gradiente (magnitud + ángulo) para cada píxel usando filtros Sobel
 *    · Divide la imagen en celdas de cell_size × cell_size píxeles
 *    · Para cada celda, acumula un histograma de n_bins orientaciones (sin signo, 0°-180°)
 *    · Agrupa celdas en bloques de block_cells × block_cells (solapados, paso=1 celda)
 *      y normaliza el descriptor de cada bloque (L2-norm)
 *    · El descriptor HOG final es la concatenación de todos los bloques normalizados
 *
 *  Además (ventaneo por tamaños):
 *    · Para cada ventana w ∈ {3×cell_size, 4×cell_size, ...} calcula HOG local
 *    · Genera mapa de magnitud de gradiente y mapa de orientación dominante
 *
 *  Salida:
 *    · Imagen de magnitud de gradiente (PNG): maps_dir/<stem>/grad_mag.png
 *    · Imagen de orientación dominante (PNG): maps_dir/<stem>/grad_angle.png
 *    · CSV con el descriptor HOG completo + magnitud media/std: features_hog.csv
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o extractor_hog HOG.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./extractor_hog [ruta_preprocesadas] [csv_salida] [ruta_mapas]
 */

#include "Caracteristicas.h"

// ============================================================================
// CONFIGURACIÓN HOG
// ============================================================================

static constexpr int   CELL_SIZE   = 8;    // Tamaño de celda en píxeles
static constexpr int   N_BINS      = 9;    // Bins del histograma (0°-180°, paso 20°)
static constexpr int   BLOCK_CELLS = 2;    // Celdas por bloque en cada dimensión (2×2)
static constexpr double BIN_WIDTH  = 180.0 / N_BINS;   // 20° por bin

// ============================================================================
// GRADIENTES
// ============================================================================

// Calcula imágenes de magnitud y ángulo (sin signo, 0°-180°) del gradiente.
inline void computeGradients(const cv::Mat& gray,
                              cv::Mat& magnitude, cv::Mat& angle_deg)
{
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 1);

    // Magnitud euclidea
    cv::magnitude(gx, gy, magnitude);

    // Ángulo sin signo: 0°-180°
    cv::phase(gx, gy, angle_deg, true);   // 'true' → grados
    // Mapear 180°-360° al rango 0°-180° (sin signo)
    for (int r = 0; r < angle_deg.rows; ++r)
        for (int c = 0; c < angle_deg.cols; ++c) {
            float& a = angle_deg.at<float>(r, c);
            if (a >= 180.0f) a -= 180.0f;
        }
}

// ============================================================================
// HISTOGRAMA DE CELDA
// ============================================================================

// Acumula el histograma HOG de orientaciones para una celda de tamaño cell_size×cell_size.
// Ponderado por la magnitud del gradiente (votación soft).
// Retorna vector de N_BINS bins.
inline std::vector<double> cellHistogram(const cv::Mat& mag_cell,
                                          const cv::Mat& ang_cell)
{
    std::vector<double> hist(N_BINS, 0.0);

    for (int r = 0; r < mag_cell.rows; ++r) {
        for (int c = 0; c < mag_cell.cols; ++c) {
            const double mag = mag_cell.at<float>(r, c);
            const double ang = ang_cell.at<float>(r, c);

            // Interpolación bilineal entre los dos bins más cercanos
            const double bin_exact = ang / BIN_WIDTH;
            const int    bin_lo    = static_cast<int>(bin_exact) % N_BINS;
            const int    bin_hi    = (bin_lo + 1) % N_BINS;
            const double weight_hi = bin_exact - std::floor(bin_exact);
            const double weight_lo = 1.0 - weight_hi;

            hist[bin_lo] += mag * weight_lo;
            hist[bin_hi] += mag * weight_hi;
        }
    }
    return hist;
}

// ============================================================================
// DESCRIPTOR HOG COMPLETO
// ============================================================================

struct HOGDescriptor
{
    std::vector<double> descriptor;    // Descriptor HOG normalizado (completo)
    int n_cells_x = 0, n_cells_y = 0; // Dimensiones en celdas
    int n_blocks_x = 0, n_blocks_y = 0;
    int descriptor_dim = 0;
};

// Calcula el descriptor HOG completo de una imagen gray.
HOGDescriptor computeHOGDescriptor(const cv::Mat& gray)
{
    HOGDescriptor result;

    cv::Mat magnitude, angle_deg;
    computeGradients(gray, magnitude, angle_deg);

    // Número de celdas completas en cada dimensión
    result.n_cells_x = gray.cols / CELL_SIZE;
    result.n_cells_y = gray.rows / CELL_SIZE;

    // Construir grid de histogramas de celda
    // cell_hists[r][c] = vector de N_BINS
    std::vector<std::vector<std::vector<double>>> cell_hists(
        result.n_cells_y,
        std::vector<std::vector<double>>(result.n_cells_x,
            std::vector<double>(N_BINS, 0.0)));

    for (int cy = 0; cy < result.n_cells_y; ++cy) {
        for (int cx = 0; cx < result.n_cells_x; ++cx) {
            const cv::Rect roi(cx * CELL_SIZE, cy * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE);
            cell_hists[cy][cx] = cellHistogram(magnitude(roi), angle_deg(roi));
        }
    }

    // Normalizar bloques de BLOCK_CELLS × BLOCK_CELLS celdas (solapados, paso=1)
    result.n_blocks_x = result.n_cells_x - BLOCK_CELLS + 1;
    result.n_blocks_y = result.n_cells_y - BLOCK_CELLS + 1;

    if (result.n_blocks_x <= 0 || result.n_blocks_y <= 0) {
        // Imagen demasiado pequeña para al menos un bloque
        return result;
    }

    result.descriptor_dim = result.n_blocks_y * result.n_blocks_x
                            * BLOCK_CELLS * BLOCK_CELLS * N_BINS;
    result.descriptor.reserve(result.descriptor_dim);

    for (int by = 0; by < result.n_blocks_y; ++by) {
        for (int bx = 0; bx < result.n_blocks_x; ++bx) {
            // Concatenar histogramas de las BLOCK_CELLS² celdas del bloque
            std::vector<double> block_vec;
            block_vec.reserve(BLOCK_CELLS * BLOCK_CELLS * N_BINS);
            for (int dy = 0; dy < BLOCK_CELLS; ++dy) {
                for (int dx = 0; dx < BLOCK_CELLS; ++dx) {
                    const auto& h = cell_hists[by + dy][bx + dx];
                    block_vec.insert(block_vec.end(), h.begin(), h.end());
                }
            }

            // Normalización L2 del bloque
            double norm_sq = 0.0;
            for (double v : block_vec) norm_sq += v * v;
            const double norm = std::sqrt(norm_sq + 1e-8);
            for (double& v : block_vec) v /= norm;

            result.descriptor.insert(result.descriptor.end(),
                                     block_vec.begin(), block_vec.end());
        }
    }

    return result;
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char** argv)
{
    std::string input_dir  = "Data/Preprocesadas";
    std::string output_csv = "Data/Features/features_hog.csv";
    std::string maps_dir   = "Data/Features/maps/HOG";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_csv = argv[2];
    if (argc >= 4) maps_dir   = argv[3];

    std::cout << "Extractor HOG — Histograma de Gradientes Orientados" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Celda: " << CELL_SIZE << "×" << CELL_SIZE
              << "px | Bloque: " << BLOCK_CELLS << "×" << BLOCK_CELLS
              << " celdas | " << N_BINS << " bins (0°-180°)" << std::endl;

    auto images = loadImages(input_dir);
    if (images.empty()) {
        std::cerr << "No se encontraron imágenes en: " << input_dir << std::endl;
        return 1;
    }

    // ── Calcular dimensión del descriptor ──────────────────────────────────────
    // Para imagen 224×224: n_cells = 28×28, n_blocks = 27×27
    // descriptor_dim = 27×27×2×2×9 = 26,244
    // (se verifica con la primera imagen)
    HOGDescriptor sample_hog = computeHOGDescriptor(images[0]->gray);
    const int hog_dim = sample_hog.descriptor_dim;
    std::cout << "Dimensión descriptor HOG: " << hog_dim
              << " (" << sample_hog.n_blocks_y << "×" << sample_hog.n_blocks_x
              << " bloques × " << BLOCK_CELLS*BLOCK_CELLS*N_BINS << ")" << std::endl;

    // ── Cabecera CSV ──────────────────────────────────────────────────────────
    // label, filename, grad_mag_mean, grad_mag_std, HOG_0, HOG_1, ..., HOG_N
    std::vector<std::string> header = {"label", "filename",
                                       "grad_magnitud_mean", "grad_magnitud_std"};
    for (int k = 0; k < hog_dim; ++k)
        header.push_back("HOG_" + std::to_string(k));

    CSVWriter csv(output_csv, header);

    // ── Procesar cada imagen ──────────────────────────────────────────────────
    for (const auto& img : images) {
        std::cout << "Procesando: " << img->filename << "..." << std::flush;

        const std::string img_stem = fs::path(img->filename).stem().string();

        // Calcular gradientes y guardar mapas visuales
        cv::Mat magnitude, angle_deg;
        computeGradients(img->gray, magnitude, angle_deg);

        {
            fs::path dir = fs::path(maps_dir) / img_stem;
            try { fs::create_directories(dir); } catch (...) {}
            cv::imwrite((dir / "grad_magnitud.png").string(), normalizeToImage(magnitude));
            cv::imwrite((dir / "grad_angulo.png").string(),   normalizeToImage(angle_deg));
        }

        // Estadísticas de magnitud de gradiente
        double mag_mean, mag_std;
        mapStats(magnitude, mag_mean, mag_std);

        // Descriptor HOG
        const HOGDescriptor hog = computeHOGDescriptor(img->gray);

        // Ensamblar fila CSV
        std::vector<double> row_values = {mag_mean, mag_std};
        row_values.insert(row_values.end(), hog.descriptor.begin(), hog.descriptor.end());

        // Padding si el descriptor es más corto de lo esperado
        while ((int)row_values.size() < hog_dim + 2) row_values.push_back(0.0);

        csv.writeRow(img->metadata.label() + "," + img->filename, row_values);
        std::cout << " OK (dim=" << hog.descriptor.size() << ")" << std::endl;
    }

    std::cout << "\nExtractor HOG completado." << std::endl;
    std::cout << "CSV: " << output_csv << std::endl;
    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o extractor_hog HOG.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./extractor_hog [ruta_preprocesadas] [csv_salida] [ruta_mapas]
//   ./extractor_hog /mnt/d/Data/Preprocesadas features_hog.csv /mnt/d/Data/FeatureMaps/HOG
