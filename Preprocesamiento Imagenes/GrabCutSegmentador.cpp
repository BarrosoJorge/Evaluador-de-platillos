/*  GrabCutSegmentador.cpp
 *
 *  Etapa 1 del pipeline — Segmentación interactiva de imágenes de platillos.
 *
 *  Para cada imagen el usuario delimita el platillo con un rectángulo (ROI).
 *  El sistema aplica GrabCut (OpenCV) seguido de postprocesamiento morfológico
 *  e inpainting con el método de Telea para producir una máscara limpia.
 *
 *  Pipeline por imagen:
 *    1. Usuario selecciona ROI  →  2. GrabCut (5 iter.)  →  3. Apertura morfológica
 *    →  4. Cierre morfológico   →  5. Inpainting (Telea)  →  6. Aplicar máscara
 *
 *  Referencia metodológica: sección 3.1 del ReporteFinal (Peñaran Prieto, 2024).
 *
 *  Entrada:  Data/Raw/Imagenes/  (jerarquía Ciudad/Platillo/Angulo/Autor/)
 *  Salida:   Data/Segmentadas/   (misma jerarquía; guarda *_mask.png y *_seg.png)
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o grabcut_seg GrabCutSegmentador.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./grabcut_seg [ruta_entrada] [ruta_salida]
 *      ./grabcut_seg /mnt/d/Data/Raw/Imagenes /mnt/d/Data/Segmentadas
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>   // cv::inpaint — requiere módulo photo de OpenCV

namespace fs = std::filesystem;

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

struct ImageMetadata
{
    std::string city;       // Ciudad
    std::string dish;       // Platillo
    std::string angle;      // Superior | Lateral
    std::string author;     // Chef | Estudiante
    std::string quality;    // B | R | M  (solo Estudiante)

    std::string format() const
    {
        std::string r = city + "_" + dish + "_" + angle + "_" + author;
        if (!quality.empty()) r += "_" + quality;
        return r;
    }
};

struct DishImage
{
    cv::Mat original;          // Imagen original cargada de disco
    cv::Mat mask;              // Máscara binaria final: 255 = platillo, 0 = fondo
    cv::Mat segmented;         // Imagen con fondo negro (máscara aplicada + inpainting)
    std::string file_path;     // Ruta completa al archivo fuente
    std::string filename;      // Nombre del archivo (con extensión)
    ImageMetadata metadata;    // Metadatos extraídos del nombre
    bool processed = false;    // ¿Fue segmentado con éxito?

    explicit DishImage(const std::string& path)
        : file_path(path), filename(fs::path(path).filename().string())
    {
        original = cv::imread(path);
        if (original.empty())
            throw std::runtime_error("No se pudo cargar la imagen: " + path);
    }
};

// ============================================================================
// DECLARACIONES DE FUNCIONES
// ============================================================================

ImageMetadata parseFilename(const std::string& filename);
std::vector<std::unique_ptr<DishImage>> batchLoad(const std::string& directory);

cv::Mat applyGrabCut(const cv::Mat& image, const cv::Rect& roi, int iterations = 5);
cv::Mat morphologicalPostprocess(const cv::Mat& binary_mask);
cv::Mat applyMaskToImage(const cv::Mat& image, const cv::Mat& mask);
cv::Mat inpaintHoles(const cv::Mat& segmented, const cv::Mat& mask);

cv::Mat scaleForDisplay(const cv::Mat& image, int max_w = 900, int max_h = 700);
bool saveSegmented(const DishImage& img, const std::string& output_dir);
void interactiveSegmentation(std::vector<std::unique_ptr<DishImage>>& images,
                              const std::string& output_dir);

// ============================================================================
// IMPLEMENTACIONES
// ============================================================================

// Extrae metadatos del nombre de archivo (Ciudad_Platillo_Angulo_Autor[_Calidad][_N])
ImageMetadata parseFilename(const std::string& filename)
{
    ImageMetadata meta;
    std::string name = filename.substr(0, filename.find_last_of('.'));

    std::vector<std::string> parts;
    size_t start = 0, end = 0;
    while ((end = name.find('_', start)) != std::string::npos) {
        parts.push_back(name.substr(start, end - start));
        start = end + 1;
    }
    parts.push_back(name.substr(start));

    if (parts.size() >= 4) {
        meta.city   = parts[0];
        meta.dish   = parts[1];
        meta.angle  = parts[2];
        meta.author = parts[3];
        // quality es B, R o M — ignorar si parts[4] es un número (frame index)
        if (parts.size() >= 5 &&
            (parts[4] == "B" || parts[4] == "R" || parts[4] == "M")) {
            meta.quality = parts[4];
        }
    }
    return meta;
}

// Carga recursivamente todas las imágenes soportadas del directorio dado
std::vector<std::unique_ptr<DishImage>> batchLoad(const std::string& directory)
{
    std::vector<std::unique_ptr<DishImage>> images;

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Error: directorio inválido: " << directory << std::endl;
        return images;
    }

    const std::vector<std::string> supported_exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
    };

    try {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            bool supported = std::any_of(
                supported_exts.begin(), supported_exts.end(),
                [&ext](const std::string& e) { return ext == e; });

            if (!supported) continue;

            try {
                auto img = std::make_unique<DishImage>(entry.path().string());
                img->metadata = parseFilename(img->filename);
                std::cout << "Cargada: " << img->filename << std::endl;
                images.push_back(std::move(img));
            } catch (const std::exception& e) {
                std::cerr << "Advertencia al cargar " << entry.path().filename()
                          << ": " << e.what() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error al recorrer directorio: " << e.what() << std::endl;
    }

    std::cout << "Total imágenes cargadas: " << images.size() << std::endl;
    return images;
}

// Aplica GrabCut con el rectángulo ROI dado.
// El algoritmo modela fondo y foreground con mezclas gaussianas (GMM)
// e itera hasta producir una máscara de segmentación.
// Retorna máscara binaria: 255 = platillo, 0 = fondo.
cv::Mat applyGrabCut(const cv::Mat& image, const cv::Rect& roi, int iterations)
{
    // Inicializar como fondo probable (GC_BGD = 0)
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_model, fgd_model;   // Modelos GMM internos de GrabCut

    cv::grabCut(image, mask, roi,
                bgd_model, fgd_model,
                iterations, cv::GC_INIT_WITH_RECT);

    // GC_FGD (1) = foreground definitivo
    // GC_PR_FGD (3) = foreground probable → ambos son platillo
    return (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
}

// Postprocesamiento morfológico de la máscara binaria.
//
// Apertura (OPEN):  elimina elementos delgados, filamentos y artefactos pequeños
//                   que GrabCut puede haber clasificado erróneamente como platillo.
// Cierre (CLOSE):   rellena huecos internos y garantiza continuidad del área
//                   del platillo (bordes irregulares o texturas conflictivas).
cv::Mat morphologicalPostprocess(const cv::Mat& binary_mask)
{
    cv::Mat kernel_open  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size( 5,  5));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));

    cv::Mat opened, closed;
    cv::morphologyEx(binary_mask, opened, cv::MORPH_OPEN,  kernel_open,  cv::Point(-1,-1), 2);
    cv::morphologyEx(opened,      closed, cv::MORPH_CLOSE, kernel_close, cv::Point(-1,-1), 3);

    return closed;
}

// Aplica la máscara binaria a la imagen — el fondo queda negro (ceros)
cv::Mat applyMaskToImage(const cv::Mat& image, const cv::Mat& mask)
{
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());
    image.copyTo(result, mask);
    return result;
}

// Inpainting con el método de Telea sobre regiones inconsistentes.
// Las inconsistencias son píxeles dentro del bounding box del platillo
// que la máscara no incluyó: huecos internos que GrabCut dejó como fondo.
// Solo se aplica si los huecos son entre 1 px y 35% del bbox (umbral conservador).
cv::Mat inpaintHoles(const cv::Mat& segmented, const cv::Mat& mask)
{
    std::vector<cv::Point> fg_pts;
    cv::findNonZero(mask, fg_pts);
    if (fg_pts.empty()) return segmented.clone();

    cv::Rect bbox = cv::boundingRect(fg_pts);

    // Región interior del bounding box que NO pertenece al platillo = huecos
    cv::Mat interior = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::rectangle(interior, bbox, cv::Scalar(255), cv::FILLED);

    cv::Mat holes;
    cv::bitwise_and(interior, ~mask, holes);

    const int hole_count = cv::countNonZero(holes);
    // Sin huecos, o demasiados (segmentación deficiente) → no inpaintar
    if (hole_count == 0 || hole_count > bbox.area() * 0.35)
        return segmented.clone();

    cv::Mat inpainted;
    cv::inpaint(segmented, holes, inpainted, 5, cv::INPAINT_TELEA);

    // Re-aplicar la máscara para que el fondo vuelva a ser negro
    cv::Mat result = cv::Mat::zeros(inpainted.size(), inpainted.type());
    inpainted.copyTo(result, mask);
    return result;
}

// Escala una imagen para que quepa en max_w × max_h sin ampliar ni distorsionar
cv::Mat scaleForDisplay(const cv::Mat& image, int max_w, int max_h)
{
    if (image.empty()) return image;

    const double scale = std::min({
        static_cast<double>(max_w) / image.cols,
        static_cast<double>(max_h) / image.rows,
        1.0   // nunca ampliar
    });

    if (scale == 1.0) return image.clone();

    cv::Mat display;
    cv::resize(image, display, cv::Size(), scale, scale, cv::INTER_AREA);
    return display;
}

// Guarda la máscara (*_mask.png) y la imagen segmentada (*_seg.png)
// reproduciendo la jerarquía Ciudad/Platillo/Angulo/Autor/ en output_dir.
bool saveSegmented(const DishImage& img, const std::string& output_dir)
{
    fs::path out = fs::path(output_dir)
                   / img.metadata.city
                   / img.metadata.dish
                   / img.metadata.angle
                   / img.metadata.author;

    try {
        fs::create_directories(out);
    } catch (const std::exception& e) {
        std::cerr << "Error creando directorio: " << e.what() << std::endl;
        return false;
    }

    const std::string stem = fs::path(img.filename).stem().string();

    bool ok = true;
    ok &= cv::imwrite((out / (stem + "_mask.png")).string(), img.mask);
    ok &= cv::imwrite((out / (stem + "_seg.png")).string(),  img.segmented);

    if (ok)
        std::cout << "Guardado: " << (out / stem).string() << " [_mask.png + _seg.png]" << std::endl;
    else
        std::cerr << "Error al escribir archivos para: " << stem << std::endl;

    return ok;
}

// Sesión interactiva de segmentación.
// Para cada imagen: el usuario delimita el platillo con el mouse (ROI),
// se aplica GrabCut + postprocesamiento, se muestra la comparación
// original / segmentada y se guarda el resultado en disco.
void interactiveSegmentation(std::vector<std::unique_ptr<DishImage>>& images,
                              const std::string& output_dir)
{
    if (images.empty()) {
        std::cout << "No hay imágenes para procesar." << std::endl;
        return;
    }

    std::cout << "\nControles de GrabCut Segmentador:" << std::endl;
    std::cout << "  Arrastra el mouse para delimitar el ROI del platillo" << std::endl;
    std::cout << "  ESPACIO / ENTER : confirmar ROI y aplicar segmentación" << std::endl;
    std::cout << "  C / ESC         : cancelar — omitir imagen actual" << std::endl;
    std::cout << "  Q (ventana resultado) : interrumpir y guardar lo procesado" << std::endl;

    for (size_t i = 0; i < images.size(); ++i) {
        DishImage* img = images[i].get();

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Imagen " << (i + 1) << " de " << images.size()
                  << ": " << img->filename << std::endl;
        if (!img->metadata.city.empty()) {
            std::cout << "  [" << img->metadata.city << " / "
                      << img->metadata.dish  << " / "
                      << img->metadata.angle << " / "
                      << img->metadata.author << "]" << std::endl;
        }
        std::cout << std::string(70, '=') << std::endl;

        // Escalar para que quepa en pantalla manteniendo la proporción
        cv::Mat display = scaleForDisplay(img->original);
        const double sx = static_cast<double>(img->original.cols) / display.cols;
        const double sy = static_cast<double>(img->original.rows) / display.rows;

        // selectROI: OpenCV muestra la imagen y captura el rectángulo del usuario
        const std::string win_name = "GrabCut — " + img->filename
                                   + "  [ESPACIO=OK  ESC=Omitir]";
        cv::Rect roi_scaled = cv::selectROI(win_name, display, false, false);
        cv::destroyWindow(win_name);

        if (roi_scaled.empty()) {
            std::cout << "ROI vacío — imagen omitida." << std::endl;
            continue;
        }

        // Escalar el ROI al tamaño real de la imagen
        cv::Rect roi(
            static_cast<int>(roi_scaled.x      * sx),
            static_cast<int>(roi_scaled.y      * sy),
            static_cast<int>(roi_scaled.width  * sx),
            static_cast<int>(roi_scaled.height * sy)
        );
        // Recortar al área válida de la imagen
        roi &= cv::Rect(0, 0, img->original.cols, img->original.rows);

        if (roi.width < 20 || roi.height < 20) {
            std::cout << "ROI demasiado pequeño — imagen omitida." << std::endl;
            continue;
        }

        std::cout << "Aplicando GrabCut (5 iteraciones) + postprocesamiento..." << std::endl;

        const cv::Mat binary  = applyGrabCut(img->original, roi);
        const cv::Mat clean   = morphologicalPostprocess(binary);
        const cv::Mat masked  = applyMaskToImage(img->original, clean);
        const cv::Mat final_img = inpaintHoles(masked, clean);

        img->mask      = clean;
        img->segmented = final_img;
        img->processed = true;

        // Preparar comparación lado a lado: Original | Segmentada
        cv::Mat left  = scaleForDisplay(img->original, 450, 400);
        cv::Mat right = scaleForDisplay(final_img,     450, 400);

        // Igualar alturas para concatenar horizontalmente
        const int h = std::max(left.rows, right.rows);
        cv::Mat pad_l = cv::Mat::zeros(h, left.cols,  left.type());
        cv::Mat pad_r = cv::Mat::zeros(h, right.cols, right.type());
        left.copyTo(pad_l(cv::Rect(0, 0, left.cols,  left.rows)));
        right.copyTo(pad_r(cv::Rect(0, 0, right.cols, right.rows)));

        cv::Mat comparison;
        cv::hconcat(pad_l, pad_r, comparison);

        const cv::Scalar label_color = {0, 220, 0};
        cv::putText(comparison, "Original",   {10, 28},
                    cv::FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2);
        cv::putText(comparison, "Segmentada", {pad_l.cols + 10, 28},
                    cv::FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2);

        cv::imshow("Resultado — cualquier tecla para continuar, Q para salir", comparison);
        const int key = cv::waitKey(0) & 0xFF;
        cv::destroyAllWindows();

        saveSegmented(*img, output_dir);

        if (key == 'q' || key == 'Q') {
            std::cout << "Proceso interrumpido por el usuario." << std::endl;
            break;
        }
    }

    cv::destroyAllWindows();

    const long done = std::count_if(images.begin(), images.end(),
        [](const std::unique_ptr<DishImage>& d){ return d->processed; });

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Segmentación finalizada: " << done
              << " de " << images.size() << " imágenes procesadas." << std::endl;
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char** argv)
{
    std::string input_dir  = "/mnt/d/Data/Raw/Imagenes";
    std::string output_dir = "/mnt/d/Data/Segmentadas";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_dir = argv[2];

    std::cout << "GrabCut Segmentador de Platillos" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Entrada:  " << input_dir  << std::endl;
    std::cout << "Salida:   " << output_dir << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    try {
        auto images = batchLoad(input_dir);

        if (images.empty()) {
            std::cerr << "No se encontraron imágenes en: " << input_dir << std::endl;
            return 1;
        }

        interactiveSegmentation(images, output_dir);

    } catch (const std::exception& e) {
        std::cerr << "Error fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o grabcut_seg GrabCutSegmentador.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./grabcut_seg [ruta_entrada] [ruta_salida]
//   ./grabcut_seg /mnt/d/Data/Raw/Imagenes /mnt/d/Data/Segmentadas
