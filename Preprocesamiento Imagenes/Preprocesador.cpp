/*  Preprocesador.cpp
 *
 *  Etapa 1 del pipeline — Preprocesamiento y alineación de imágenes segmentadas.
 *
 *  Toma las imágenes producidas por GrabCutSegmentador (*_seg.png) y aplica:
 *    1. Redimensionamiento uniforme a 224×224 px (compatibilidad con VGG16)
 *    2. Corrección de orientación:  detecta el ángulo del platillo con minAreaRect
 *       y rota de forma segura (expandiendo el canvas, sin recortar bordes).
 *    3. Alineación entre imágenes:  para cada imagen de Estudiante, compara la
 *       versión original y la rotada 180° contra la referencia del Chef usando
 *       SSIM (Wang et al., 2004) y conserva la de mayor similitud estructural.
 *
 *  Agrupación: las imágenes se agrupan por (Ciudad, Platillo, Ángulo).
 *  Dentro de cada grupo, la imagen del Chef es la referencia de alineación.
 *
 *  Referencia metodológica: sección 3.2 del ReporteFinal (Peñaran Prieto, 2024).
 *
 *  Entrada:  Data/Segmentadas/   (jerarquía Ciudad/Platillo/Angulo/Autor/*_seg.png)
 *  Salida:   Data/Preprocesadas/ (misma jerarquía; guarda *_pre.png, 224×224)
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o preprocesador Preprocesador.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./preprocesador [ruta_segmentadas] [ruta_salida]
 *      ./preprocesador /mnt/d/Data/Segmentadas /mnt/d/Data/Preprocesadas
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

struct ImageMetadata
{
    std::string city;
    std::string dish;
    std::string angle;
    std::string author;
    std::string quality;

    std::string format() const
    {
        std::string r = city + "_" + dish + "_" + angle + "_" + author;
        if (!quality.empty()) r += "_" + quality;
        return r;
    }

    bool isChef()      const { return author == "Chef"; }
    bool isEstudiante() const { return author == "Estudiante"; }
};

struct ProcessedImage
{
    cv::Mat original;        // Imagen de entrada (segmentada desde GrabCutSegmentador)
    cv::Mat preprocessed;    // Imagen final preprocesada (224×224, orientada, alineada)
    std::string file_path;
    std::string filename;
    ImageMetadata metadata;

    explicit ProcessedImage(const std::string& path)
        : file_path(path), filename(fs::path(path).filename().string())
    {
        original = cv::imread(path);
        if (original.empty())
            throw std::runtime_error("No se pudo cargar la imagen: " + path);
    }
};

// Clave de agrupación: las imágenes con mismo (city, dish, angle) comparten referencia
struct GroupKey
{
    std::string city, dish, angle;

    bool operator<(const GroupKey& o) const
    {
        return std::tie(city, dish, angle) < std::tie(o.city, o.dish, o.angle);
    }
};

// ============================================================================
// DECLARACIONES DE FUNCIONES
// ============================================================================

ImageMetadata parseFromPath(const fs::path& file_path, const fs::path& base_dir);
std::vector<std::unique_ptr<ProcessedImage>> batchLoad(const std::string& directory);

cv::Mat resizeUniform(const cv::Mat& image, int target_size = 224);
double  getOrientationAngle(const cv::Mat& image);
cv::Mat rotateSafe(const cv::Mat& image, double angle_deg, cv::Point2f center);
cv::Mat correctOrientation(const cv::Mat& image);
double  computeSSIM(const cv::Mat& img1, const cv::Mat& img2);
cv::Mat alignToReference(const cv::Mat& student, const cv::Mat& reference);
cv::Mat preprocessImage(const cv::Mat& image, const cv::Mat& reference = cv::Mat());

bool savePreprocessed(const ProcessedImage& img, const std::string& output_dir);
void runBatchPipeline(std::vector<std::unique_ptr<ProcessedImage>>& images,
                       const std::string& output_dir);

// ============================================================================
// IMPLEMENTACIONES — CARGA DE DATOS
// ============================================================================

// Extrae metadatos desde la ruta del archivo relativa al directorio base.
// Más confiable que parsear el nombre: el árbol de carpetas define la jerarquía.
// Estructura esperada: base/Ciudad/Platillo/Angulo/Autor/[Calidad/]archivo.png
ImageMetadata parseFromPath(const fs::path& file_path, const fs::path& base_dir)
{
    ImageMetadata meta;

    fs::path rel = fs::relative(file_path, base_dir);
    std::vector<std::string> parts;
    for (const auto& component : rel)
        parts.push_back(component.string());

    // parts[0]=Ciudad, parts[1]=Platillo, parts[2]=Angulo, parts[3]=Autor, parts[4]=archivo
    if (parts.size() >= 5) {
        meta.city   = parts[0];
        meta.dish   = parts[1];
        meta.angle  = parts[2];
        meta.author = parts[3];
    }

    // Intentar extraer quality (B/R/M) del nombre del archivo
    std::string stem = fs::path(parts.back()).stem().string();
    // Remover sufijo _seg si está presente
    if (stem.size() >= 4 && stem.substr(stem.size() - 4) == "_seg")
        stem = stem.substr(0, stem.size() - 4);

    for (const std::string& q : {"_B_", "_R_", "_M_"}) {
        if (stem.find(q) != std::string::npos) {
            meta.quality = std::string(1, q[1]);   // extrae 'B', 'R' o 'M'
            break;
        }
    }
    // Caso borde: calidad al final del stem (ej. "..._Estudiante_B_0" → "B" ya en parts)
    if (meta.quality.empty()) {
        for (const std::string& q : {"_B", "_R", "_M"}) {
            if (stem.size() >= 2 && stem.substr(stem.size() - 2) == q) {
                meta.quality = std::string(1, q[1]);
                break;
            }
        }
    }

    return meta;
}

// Carga todas las imágenes segmentadas (*_seg.png) del directorio
std::vector<std::unique_ptr<ProcessedImage>> batchLoad(const std::string& directory)
{
    std::vector<std::unique_ptr<ProcessedImage>> images;
    const fs::path base = directory;

    if (!fs::exists(base) || !fs::is_directory(base)) {
        std::cerr << "Error: directorio inválido: " << directory << std::endl;
        return images;
    }

    const std::vector<std::string> supported_exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
    };

    try {
        for (const auto& entry : fs::recursive_directory_iterator(base)) {
            if (!entry.is_regular_file()) continue;

            // Solo procesar imágenes segmentadas (sufijo _seg)
            const std::string stem = entry.path().stem().string();
            if (stem.size() < 4 || stem.substr(stem.size() - 4) != "_seg")
                continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            bool supported = std::any_of(
                supported_exts.begin(), supported_exts.end(),
                [&ext](const std::string& e){ return ext == e; });

            if (!supported) continue;

            try {
                auto img = std::make_unique<ProcessedImage>(entry.path().string());
                img->metadata = parseFromPath(entry.path(), base);

                std::cout << "Cargada: " << img->filename
                          << "  [" << img->metadata.city   << "/"
                                   << img->metadata.dish   << "/"
                                   << img->metadata.angle  << "/"
                                   << img->metadata.author << "]" << std::endl;

                images.push_back(std::move(img));
            } catch (const std::exception& e) {
                std::cerr << "Advertencia: " << e.what() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error al recorrer directorio: " << e.what() << std::endl;
    }

    std::cout << "Total: " << images.size() << " imágenes cargadas." << std::endl;
    return images;
}

// ============================================================================
// IMPLEMENTACIONES — REDIMENSIONAMIENTO
// ============================================================================

// Redimensiona la imagen a target_size × target_size píxeles.
// Se usa interpolación INTER_AREA (recomendada para reducir resolución)
// para minimizar el aliasing en imágenes de platillos con texturas finas.
cv::Mat resizeUniform(const cv::Mat& image, int target_size)
{
    if (image.empty()) return image;
    if (image.cols == target_size && image.rows == target_size) return image.clone();

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_size, target_size), 0, 0, cv::INTER_AREA);
    return resized;
}

// ============================================================================
// IMPLEMENTACIONES — CORRECCIÓN DE ORIENTACIÓN
// ============================================================================

// Detecta el ángulo de inclinación del platillo usando el contorno principal
// y el rectángulo mínimo de área (minAreaRect).
// Retorna 0.0 si no se detecta un contorno válido.
double getOrientationAngle(const cv::Mat& image)
{
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    // Binarizar: cualquier píxel no negro pertenece al platillo
    cv::Mat binary;
    cv::threshold(gray, binary, 1, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return 0.0;

    // Contorno más grande = platillo (los más pequeños son ruido residual)
    auto it_largest = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
            return cv::contourArea(a) < cv::contourArea(b);
        });

    // Área mínima para considerar el contorno como el platillo real
    if (cv::contourArea(*it_largest) < 500.0) return 0.0;

    cv::RotatedRect min_rect = cv::minAreaRect(*it_largest);

    double angle = min_rect.angle;    // Rango: [-90, 0) por convención de OpenCV

    // Si el lado "ancho" del rectángulo es el vertical (portrait), corregir
    if (min_rect.size.width < min_rect.size.height)
        angle += 90.0;

    return angle;
}

// Rota la imagen de forma segura: expande el canvas para que no se recorten
// las esquinas al rotar. El centro de rotación es el punto dado.
cv::Mat rotateSafe(const cv::Mat& image, double angle_deg, cv::Point2f center)
{
    if (std::abs(angle_deg) < 0.5) return image.clone();

    const double rad   = std::abs(angle_deg) * CV_PI / 180.0;
    const double cos_a = std::cos(rad);
    const double sin_a = std::sin(rad);

    // Nuevo tamaño del canvas — usar abs(cos) y abs(sin) para que no sean negativos
    // para ángulos en el segundo/tercer cuadrante (error en la versión original)
    const int new_w = static_cast<int>(std::ceil(image.cols * std::abs(cos_a) + image.rows * std::abs(sin_a)));
    const int new_h = static_cast<int>(std::ceil(image.cols * std::abs(sin_a) + image.rows * std::abs(cos_a)));

    cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);

    // Trasladar para que el centro de rotación quede en el centro del nuevo canvas
    rot.at<double>(0, 2) += new_w / 2.0 - center.x;
    rot.at<double>(1, 2) += new_h / 2.0 - center.y;

    cv::Mat result;
    cv::warpAffine(image, result, rot, cv::Size(new_w, new_h),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return result;
}

// Corrige la orientación del platillo detectando su inclinación con minAreaRect.
// Rota la imagen para alinear el eje mayor del platillo horizontalmente,
// usando el centro del rectángulo mínimo como punto de rotación.
cv::Mat correctOrientation(const cv::Mat& image)
{
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    cv::Mat binary;
    cv::threshold(gray, binary, 1, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cout << "  Orientación: no se detectó contorno — sin rotación." << std::endl;
        return image.clone();
    }

    auto it_largest = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
            return cv::contourArea(a) < cv::contourArea(b);
        });

    if (cv::contourArea(*it_largest) < 500.0) {
        std::cout << "  Orientación: contorno demasiado pequeño — sin rotación." << std::endl;
        return image.clone();
    }

    cv::RotatedRect min_rect = cv::minAreaRect(*it_largest);
    double angle = min_rect.angle;
    if (min_rect.size.width < min_rect.size.height) angle += 90.0;

    if (std::abs(angle) < 1.0) {
        std::cout << "  Orientación: ángulo < 1° — sin rotación necesaria." << std::endl;
        return image.clone();
    }

    // Ángulos > 30° suelen indicar detección errónea (máscara con bordes irregulares)
    if (std::abs(angle) > 30.0) {
        std::cout << "  Orientación: ángulo " << angle
                  << "° fuera de rango — no se corrige." << std::endl;
        return image.clone();
    }

    std::cout << "  Ángulo corregido: " << angle << "°" << std::endl;

    // Rotar centrado en el centro de la imagen (más estable que min_rect.center)
    const cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat rotated = rotateSafe(image, -angle, center);

    // Recortar al bounding box del contenido no-negro
    cv::Mat g;
    if (rotated.channels() == 3) cv::cvtColor(rotated, g, cv::COLOR_BGR2GRAY);
    else g = rotated.clone();
    cv::Mat bin2;
    cv::threshold(g, bin2, 1, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> pts;
    cv::findNonZero(bin2, pts);
    if (!pts.empty()) rotated = rotated(cv::boundingRect(pts)).clone();
    return rotated;
}

// ============================================================================
// IMPLEMENTACIONES — SSIM Y ALINEACIÓN
// ============================================================================

// Calcula el Índice de Similitud Estructural (SSIM) entre dos imágenes en
// escala de grises del mismo tamaño.
// Referencia: Wang et al., IEEE Transactions on Image Processing, 2004.
// Retorna un valor en [0, 1]: 1.0 = imágenes idénticas.
double computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
    if (img1.empty() || img2.empty() || img1.size() != img2.size()) {
        std::cerr << "computeSSIM: imágenes vacías o de tamaño distinto." << std::endl;
        return 0.0;
    }

    cv::Mat g1, g2;
    if (img1.channels() == 3) cv::cvtColor(img1, g1, cv::COLOR_BGR2GRAY);
    else g1 = img1.clone();
    if (img2.channels() == 3) cv::cvtColor(img2, g2, cv::COLOR_BGR2GRAY);
    else g2 = img2.clone();

    cv::Mat I1, I2;
    g1.convertTo(I1, CV_64F);
    g2.convertTo(I2, CV_64F);

    // Constantes de estabilidad numérica (Wang et al. 2004, L=255)
    const double C1 = 6.5025;    // (0.01 × 255)²
    const double C2 = 58.5225;   // (0.03 × 255)²

    // Medias locales mediante filtro gaussiano (ventana 11×11, σ=1.5)
    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    const cv::Mat mu1_sq  = mu1.mul(mu1);
    const cv::Mat mu2_sq  = mu2.mul(mu2);
    const cv::Mat mu1_mu2 = mu1.mul(mu2);

    // Varianzas y covarianza locales
    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(I1.mul(I1), sigma1_sq, cv::Size(11,11), 1.5); sigma1_sq -= mu1_sq;
    cv::GaussianBlur(I2.mul(I2), sigma2_sq, cv::Size(11,11), 1.5); sigma2_sq -= mu2_sq;
    cv::GaussianBlur(I1.mul(I2), sigma12,   cv::Size(11,11), 1.5); sigma12   -= mu1_mu2;

    // Mapa SSIM píxel a píxel: ssim(x) = (2μ₁μ₂+C₁)(2σ₁₂+C₂) / (μ₁²+μ₂²+C₁)(σ₁²+σ₂²+C₂)
    cv::Mat numerator, denominator, ssim_map;
    cv::multiply(2.0 * mu1_mu2 + C1,   2.0 * sigma12 + C2,           numerator);
    cv::multiply(mu1_sq + mu2_sq + C1, sigma1_sq + sigma2_sq + C2,   denominator);
    cv::divide(numerator, denominator, ssim_map);

    return cv::mean(ssim_map)[0];
}

// Alinea la imagen del Estudiante contra la referencia del Chef.
// Compara la orientación original con la rotada 180° usando SSIM
// y retorna la versión con mayor similitud estructural.
// La comparación se hace a 224×224 para normalizar la escala.
cv::Mat alignToReference(const cv::Mat& student, const cv::Mat& reference)
{
    // Redimensionar a 224×224 para que la comparación SSIM sea en la misma escala
    cv::Mat stu_224, ref_224;
    cv::resize(student,   stu_224, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
    cv::resize(reference, ref_224, cv::Size(224, 224), 0, 0, cv::INTER_AREA);

    cv::Mat rotated_180;
    cv::rotate(stu_224, rotated_180, cv::ROTATE_180);

    const double ssim_orig = computeSSIM(stu_224,      ref_224);
    const double ssim_rot  = computeSSIM(rotated_180,  ref_224);

    std::cout << "  SSIM original: " << ssim_orig
              << "  |  SSIM 180°: " << ssim_rot << std::endl;

    if (ssim_rot > ssim_orig) {
        std::cout << "  -> Se usa la versión rotada 180°." << std::endl;
        cv::Mat result;
        cv::rotate(student, result, cv::ROTATE_180);
        return result;
    }

    std::cout << "  -> Se mantiene la orientación original." << std::endl;
    return student.clone();
}

// ============================================================================
// IMPLEMENTACIONES — PIPELINE COMPLETO
// ============================================================================

// Pipeline de preprocesamiento para una imagen:
//   1. Resize 224×224
//   2. Corrección de orientación (minAreaRect)
//   3. Alineación SSIM contra referencia (si se provee)
//   4. Resize final a 224×224 (el canvas puede haberse expandido en paso 2)
cv::Mat preprocessImage(const cv::Mat& image, const cv::Mat& reference)
{
    // 0. Crop al bounding box del contenido segmentado con margen 5%
    //    Garantiza que el platillo llene el cuadro antes de resize (igual que el PDF)
    cv::Mat cropped;
    {
        cv::Mat g;
        if (image.channels() == 3) cv::cvtColor(image, g, cv::COLOR_BGR2GRAY);
        else g = image.clone();
        cv::Mat bin;
        cv::threshold(g, bin, 1, 255, cv::THRESH_BINARY);
        std::vector<cv::Point> pts;
        cv::findNonZero(bin, pts);
        if (!pts.empty()) {
            cv::Rect bb = cv::boundingRect(pts);
            int px = static_cast<int>(bb.width  * 0.05);
            int py = static_cast<int>(bb.height * 0.05);
            bb.x     = std::max(0, bb.x - px);
            bb.y     = std::max(0, bb.y - py);
            bb.width  = std::min(image.cols - bb.x, bb.width  + 2 * px);
            bb.height = std::min(image.rows - bb.y, bb.height + 2 * py);
            cropped = image(bb).clone();
        } else {
            cropped = image.clone();
        }
    }

    // 1. Resize uniforme
    cv::Mat resized = resizeUniform(cropped, 224);

    // 2. Corrección de orientación
    cv::Mat oriented = correctOrientation(resized);

    // 3. Alineación contra la referencia del Chef
    cv::Mat aligned;
    if (!reference.empty()) {
        cv::Mat ref_oriented = correctOrientation(resizeUniform(reference, 224));
        aligned = alignToReference(oriented, ref_oriented);
    } else {
        aligned = oriented.clone();
    }

    // 4. Resize final: la rotación segura puede haber cambiado las dimensiones
    cv::Mat final_img;
    cv::resize(aligned, final_img, cv::Size(224, 224), 0, 0, cv::INTER_AREA);

    return final_img;
}

// Guarda la imagen preprocesada manteniendo la jerarquía Ciudad/Platillo/Angulo/Autor/
// Cambia el sufijo de _seg a _pre para distinguir la etapa de procesamiento.
bool savePreprocessed(const ProcessedImage& img, const std::string& output_dir)
{
    fs::path out = fs::path(output_dir)
                   / img.metadata.city
                   / img.metadata.dish
                   / img.metadata.angle
                   / img.metadata.author;

    try {
        fs::create_directories(out);
    } catch (const std::exception& e) {
        std::cerr << "Error al crear directorio: " << e.what() << std::endl;
        return false;
    }

    // Reemplazar sufijo _seg por _pre
    std::string stem = fs::path(img.filename).stem().string();
    if (stem.size() >= 4 && stem.substr(stem.size() - 4) == "_seg")
        stem = stem.substr(0, stem.size() - 4) + "_pre";
    else
        stem += "_pre";

    const std::string out_path = (out / (stem + ".png")).string();
    const bool ok = cv::imwrite(out_path, img.preprocessed);

    if (ok) std::cout << "  Guardado: " << out_path << std::endl;
    else    std::cerr << "  Error al guardar: " << out_path << std::endl;

    return ok;
}

// Agrupa todas las imágenes por (Ciudad, Platillo, Ángulo), identifica la
// imagen del Chef como referencia dentro de cada grupo y corre el pipeline
// completo de preprocesamiento para cada imagen del grupo.
void runBatchPipeline(std::vector<std::unique_ptr<ProcessedImage>>& images,
                       const std::string& output_dir)
{
    if (images.empty()) {
        std::cout << "No hay imágenes para procesar." << std::endl;
        return;
    }

    // Construir grupos: clave = (city, dish, angle)
    std::map<GroupKey, std::vector<ProcessedImage*>> groups;
    for (auto& img : images) {
        GroupKey key{ img->metadata.city, img->metadata.dish, img->metadata.angle };
        groups[key].push_back(img.get());
    }

    std::cout << "\nGrupos detectados: " << groups.size() << std::endl;

    int total_processed = 0;

    for (auto& [key, group] : groups) {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Grupo: " << key.city << " / " << key.dish
                  << " / " << key.angle << "  ("
                  << group.size() << " imágenes)" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // Buscar imagen del Chef como referencia de alineación
        ProcessedImage* chef_ref = nullptr;
        for (auto* img : group) {
            if (img->metadata.isChef()) {
                chef_ref = img;
                break;
            }
        }

        if (chef_ref) {
            std::cout << "Referencia (Chef): " << chef_ref->filename << std::endl;
        } else {
            std::cout << "Advertencia: sin imagen de Chef — alineación omitida." << std::endl;
        }

        // Preprocess del Chef (no necesita alineación — ES la referencia)
        if (chef_ref) {
            std::cout << "\n[Chef] " << chef_ref->filename << std::endl;
            chef_ref->preprocessed = preprocessImage(chef_ref->original);
            savePreprocessed(*chef_ref, output_dir);
            ++total_processed;
        }

        // Preprocess del resto: Estudiantes (con alineación) y Chefs adicionales
        for (auto* img : group) {
            if (img == chef_ref) continue;   // la referencia ya fue procesada

            const bool is_student = img->metadata.isEstudiante();
            std::cout << "\n[" << (is_student ? "Estudiante" : "Chef")
                      << "] " << img->filename << std::endl;

            const cv::Mat ref = chef_ref ? chef_ref->original : cv::Mat();
            img->preprocessed = preprocessImage(img->original, ref);

            savePreprocessed(*img, output_dir);
            ++total_processed;
        }
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Pipeline completado: " << total_processed
              << " de " << images.size() << " imágenes preprocesadas." << std::endl;
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char** argv)
{
    std::string input_dir  = "Data/Segmentadas";
    std::string output_dir = "Data/Preprocesadas";

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_dir = argv[2];

    std::cout << "Preprocesador de Imágenes de Platillos" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Entrada:  " << input_dir  << " (imágenes *_seg.png)" << std::endl;
    std::cout << "Salida:   " << output_dir << " (imágenes *_pre.png, 224×224)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    try {
        auto images = batchLoad(input_dir);

        if (images.empty()) {
            std::cerr << "No se encontraron imágenes *_seg.png en: " << input_dir << std::endl;
            return 1;
        }

        runBatchPipeline(images, output_dir);

    } catch (const std::exception& e) {
        std::cerr << "Error fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -o preprocesador Preprocesador.cpp \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./preprocesador [ruta_segmentadas] [ruta_salida]
//   ./preprocesador /mnt/d/Data/Segmentadas /mnt/d/Data/Preprocesadas
