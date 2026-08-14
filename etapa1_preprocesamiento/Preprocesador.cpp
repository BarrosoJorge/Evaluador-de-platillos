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
 *  Entrada:  Data/Segmentadas/    (jerarquía Ciudad/Platillo/Angulo/Autor/*_seg.png)
 *  Salida:   Data/Preprocesadas/  (misma jerarquía; guarda *_pre.png, 224×224)
 *
 *  Cambios respecto a tu version original:
 *    - ImageMetadata se movio a scripts_genericos como VideoMetadata.
 *    - parseFromPath (que leia Ciudad/Platillo/Angulo/Autor de la RUTA
 *      de carpetas) se reemplazo por parsearNombreArchivo (el mismo
 *      parseo compartido que usan Labeler/VideoToImage/GrabCutSegmentador).
 *      Motivo: tu comentario original decia que parsear por ruta era
 *      "mas confiable" que por nombre, pero en realidad depende de que
 *      la profundidad de carpetas sea siempre igual — y no lo es: las
 *      imagenes de Estudiante tienen un nivel extra (.../Autor/Calidad/)
 *      que las de Chef no tienen, asi que parts[4] no siempre es lo
 *      mismo segun el autor. El nombre de archivo, en cambio, siempre
 *      trae el metadato completo sin importar cuantos niveles de
 *      carpeta haya encima. Ya con el guard B/R/M corregido en
 *      VideoMetadata, parsear por nombre es igual de confiable y no
 *      depende de la estructura de carpetas.
 *    - Rutas por defecto ahora vienen de PathManager en vez de estar
 *      hardcodeadas, para quedar conectado con GrabCutSegmentador.
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -I../scripts_genericos/include \
 *          ../scripts_genericos/src/Logger.cpp \
 *          ../scripts_genericos/src/PathManager.cpp \
 *          ../scripts_genericos/src/VideoMetadata.cpp \
 *          Preprocesador.cpp -o preprocesador \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./preprocesador [ruta_segmentadas] [ruta_salida]
 */

#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"

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
using namespace evaluador;

// ESTRUCTURAS DE DATOS

// ProcessedImage se queda local: carga el estado especifico de esta etapa
// (original + preprocessed) que no le corresponde a un modulo generico.
struct ProcessedImage
{
    cv::Mat original;        // Imagen de entrada (segmentada desde GrabCutSegmentador)
    cv::Mat preprocessed;    // Imagen final preprocesada (224×224, orientada, alineada)
    std::string file_path;
    std::string filename;
    VideoMetadata metadata;

    explicit ProcessedImage(const std::string& path)
        : file_path(path), filename(fs::path(path).filename().string())
    {
        original = cv::imread(path);
        if (original.empty())
            throw std::runtime_error("No se pudo cargar la imagen: " + path);
    }
};

// Clave de agrupación: las imágenes con mismo (city, dish, angle) comparten
// referencia. Es logica de negocio propia de esta etapa (como decides
// agrupar para alinear), no algo que le corresponda a VideoMetadata.
struct GroupKey
{
    std::string city, dish, angle;

    bool operator<(const GroupKey& o) const
    {
        return std::tie(city, dish, angle) < std::tie(o.city, o.dish, o.angle);
    }
};

// DECLARACIONES DE FUNCIONES

std::map<GroupKey, std::vector<fs::path>> recolectarRutasPorGrupo(const std::string& directory);
std::vector<std::unique_ptr<ProcessedImage>> cargarGrupo(const std::vector<fs::path>& rutas);

cv::Mat resizeUniform(const cv::Mat& image, int target_size = 224);
double  getOrientationAngle(const cv::Mat& image);
cv::Mat rotateSafe(const cv::Mat& image, double angle_deg, cv::Point2f center);
cv::Mat correctOrientation(const cv::Mat& image);
double  computeSSIM(const cv::Mat& img1, const cv::Mat& img2);
cv::Mat alignToReference(const cv::Mat& student, const cv::Mat& reference);
cv::Mat preprocessImage(const cv::Mat& image, const cv::Mat& reference = cv::Mat());

bool savePreprocessed(const ProcessedImage& img, const std::string& output_dir);
int procesarGrupo(const GroupKey& key, std::vector<std::unique_ptr<ProcessedImage>>& images,
                   const std::string& output_dir);

// IMPLEMENTACIONES — CARGA DE DATOS

// Solo LISTA las rutas de imagenes *_seg.png y las agrupa por
// (Ciudad, Platillo, Angulo) — NO carga ninguna imagen en memoria
// todavia. Es la parte barata; separarla de la carga real es lo que
// permite procesar grupo por grupo sin tener el dataset completo en
// RAM a la vez (el mismo problema de memoria que ya resolvimos en
// GrabCutSegmentador aplica aqui igual — Preprocesador tenia la misma
// arquitectura de "cargar todo primero").
std::map<GroupKey, std::vector<fs::path>> recolectarRutasPorGrupo(const std::string& directory)
{
    std::map<GroupKey, std::vector<fs::path>> grupos;
    const fs::path base = directory;

    if (!fs::exists(base) || !fs::is_directory(base)) {
        Logger::instance().error("Preprocesador", "Directorio invalido: " + directory);
        return grupos;
    }

    const std::vector<std::string> supported_exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
    };

    for (const auto& entry : fs::recursive_directory_iterator(base)) {
        if (!entry.is_regular_file()) continue;

        const std::string stem = entry.path().stem().string();
        if (stem.size() < 4 || stem.substr(stem.size() - 4) != "_seg")
            continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        bool supported = std::any_of(
            supported_exts.begin(), supported_exts.end(),
            [&ext](const std::string& e){ return ext == e; });

        if (!supported) continue;

        std::string stem_sin_sufijo = stem.substr(0, stem.size() - 4);
        VideoMetadata metadata = parsearNombreArchivo(stem_sin_sufijo);

        if (metadata.vacio()) {
            Logger::instance().warn("Preprocesador",
                "No se pudo extraer metadato de: " + entry.path().filename().string() + " — se omite.");
            continue;
        }

        GroupKey key{ metadata.ciudad, metadata.platillo, metadata.angulo };
        grupos[key].push_back(entry.path());
    }

    size_t totalImagenes = 0;
    for (const auto& [key, rutas] : grupos) totalImagenes += rutas.size();

    Logger::instance().info("Preprocesador",
        "Total: " + std::to_string(totalImagenes) + " imagenes en " + std::to_string(grupos.size()) + " grupo(s).");

    return grupos;
}

// Carga en memoria SOLO las rutas de un grupo especifico. El vector
// devuelto se libera automaticamente al salir de scope en main(), lo
// que baja el pico de memoria de "todo el dataset" a "un grupo a la vez"
// (chef + sus propios alumnos, que es de todas formas lo minimo que
// necesita tener en memoria simultaneamente para poder alinear).
std::vector<std::unique_ptr<ProcessedImage>> cargarGrupo(const std::vector<fs::path>& rutas)
{
    std::vector<std::unique_ptr<ProcessedImage>> images;
    images.reserve(rutas.size());

    for (const auto& ruta : rutas) {
        try {
            auto img = std::make_unique<ProcessedImage>(ruta.string());

            std::string stem = ruta.stem().string();
            std::string stem_sin_sufijo = stem.substr(0, stem.size() - 4);
            img->metadata = parsearNombreArchivo(stem_sin_sufijo);

            Logger::instance().debug("Preprocesador",
                "Cargada: " + img->filename + " [" + img->metadata.ciudad + "/" +
                img->metadata.platillo + "/" + img->metadata.angulo + "/" +
                img->metadata.autor + "]");

            images.push_back(std::move(img));
        } catch (const std::exception& e) {
            Logger::instance().warn("Preprocesador",
                "No se pudo cargar " + ruta.filename().string() + ": " + e.what());
        }
    }

    return images;
}

// IMPLEMENTACIONES — REDIMENSIONAMIENTO

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

// IMPLEMENTACIONES — CORRECCIÓN DE ORIENTACIÓN

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
        Logger::instance().debug("Preprocesador", "Orientacion: no se detecto contorno — sin rotacion.");
        return image.clone();
    }

    auto it_largest = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
            return cv::contourArea(a) < cv::contourArea(b);
        });

    if (cv::contourArea(*it_largest) < 500.0) {
        Logger::instance().debug("Preprocesador", "Orientacion: contorno demasiado pequeno — sin rotacion.");
        return image.clone();
    }

    cv::RotatedRect min_rect = cv::minAreaRect(*it_largest);
    double angle = min_rect.angle;
    if (min_rect.size.width < min_rect.size.height) angle += 90.0;

    if (std::abs(angle) < 1.0) {
        Logger::instance().debug("Preprocesador", "Orientacion: angulo < 1 grado — sin rotacion necesaria.");
        return image.clone();
    }

    // Ángulos > 30° suelen indicar detección errónea (máscara con bordes irregulares)
    if (std::abs(angle) > 30.0) {
        Logger::instance().warn("Preprocesador",
            "Orientacion: angulo " + std::to_string(angle) + " fuera de rango — no se corrige.");
        return image.clone();
    }

    Logger::instance().debug("Preprocesador", "Angulo corregido: " + std::to_string(angle));

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

// IMPLEMENTACIONES — SSIM Y ALINEACIÓN

// Calcula el Índice de Similitud Estructural (SSIM) entre dos imágenes en
// escala de grises del mismo tamaño.
// Referencia: Wang et al., IEEE Transactions on Image Processing, 2004.
// Retorna un valor en [0, 1]: 1.0 = imágenes idénticas.
double computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
    if (img1.empty() || img2.empty() || img1.size() != img2.size()) {
        Logger::instance().warn("Preprocesador", "computeSSIM: imagenes vacias o de tamano distinto.");
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

    Logger::instance().debug("Preprocesador",
        "SSIM original: " + std::to_string(ssim_orig) + " | SSIM 180: " + std::to_string(ssim_rot));

    if (ssim_rot > ssim_orig) {
        Logger::instance().debug("Preprocesador", "-> Se usa la version rotada 180.");
        cv::Mat result;
        cv::rotate(student, result, cv::ROTATE_180);
        return result;
    }

    Logger::instance().debug("Preprocesador", "-> Se mantiene la orientacion original.");
    return student.clone();
}

// IMPLEMENTACIONES — PIPELINE COMPLETO

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
                   / img.metadata.ciudad
                   / img.metadata.platillo
                   / img.metadata.angulo
                   / img.metadata.autor;

    try {
        fs::create_directories(out);
    } catch (const std::exception& e) {
        Logger::instance().error("Preprocesador", "Error al crear directorio: " + std::string(e.what()));
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

    if (ok) Logger::instance().info("Preprocesador", "Guardado: " + out_path);
    else    Logger::instance().error("Preprocesador", "Error al guardar: " + out_path);

    return ok;
}

// Procesa UN grupo ya cargado en memoria: identifica al Chef como
// referencia, preprocesa al Chef primero, y despues alinea/preprocesa
// al resto contra esa referencia. Devuelve cuantas imagenes se
// procesaron con exito.
int procesarGrupo(const GroupKey& key, std::vector<std::unique_ptr<ProcessedImage>>& images,
                   const std::string& output_dir)
{
    if (images.empty()) return 0;

    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "Grupo: " << key.city << " / " << key.dish
              << " / " << key.angle << "  ("
              << images.size() << " imágenes)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    // Buscar imagen del Chef como referencia de alineación
    ProcessedImage* chef_ref = nullptr;
    for (auto& img : images) {
        if (img->metadata.esChef()) {
            chef_ref = img.get();
            break;
        }
    }

    if (chef_ref) {
        Logger::instance().info("Preprocesador", "Referencia (Chef): " + chef_ref->filename);
    } else {
        Logger::instance().warn("Preprocesador", "Sin imagen de Chef en este grupo — alineacion omitida.");
    }

    int procesadas = 0;

    // Preprocess del Chef (no necesita alineación — ES la referencia)
    if (chef_ref) {
        std::cout << "\n[Chef] " << chef_ref->filename << std::endl;
        chef_ref->preprocessed = preprocessImage(chef_ref->original);
        savePreprocessed(*chef_ref, output_dir);
        ++procesadas;
    }

    // Preprocess del resto: Estudiantes (con alineación) y Chefs adicionales
    for (auto& img : images) {
        if (img.get() == chef_ref) continue;   // la referencia ya fue procesada

        const bool is_student = img->metadata.esEstudiante();
        std::cout << "\n[" << (is_student ? "Estudiante" : "Chef")
                  << "] " << img->filename << std::endl;

        const cv::Mat ref = chef_ref ? chef_ref->original : cv::Mat();
        img->preprocessed = preprocessImage(img->original, ref);

        savePreprocessed(*img, output_dir);
        ++procesadas;
    }

    return procesadas;
}

// FUNCIÓN PRINCIPAL

// Verifica si YA existe el _pre.png de salida para esta ruta *_seg.png,
// sin cargar la imagen — permite reanudar sin repetir trabajo despues
// de una interrupcion.
bool yaProcesada(const fs::path& rutaSeg, const std::string& output_dir)
{
    std::string stem = rutaSeg.stem().string();
    std::string stem_sin_sufijo = stem.substr(0, stem.size() - 4);
    VideoMetadata metadata = parsearNombreArchivo(stem_sin_sufijo);
    if (metadata.vacio()) return false;

    fs::path out = fs::path(output_dir) / metadata.ciudad / metadata.platillo / metadata.angulo / metadata.autor;
    fs::path rutaPre = out / (stem_sin_sufijo + "_pre.png");

    return fs::exists(rutaPre);
}

int main(int argc, char** argv)
{
    PathManager rutas(".");
    std::string input_dir  = rutas.segmentadas().string();
    std::string output_dir = rutas.preprocesadas().string();

    if (argc >= 2) input_dir  = argv[1];
    if (argc >= 3) output_dir = argv[2];

    Logger::instance().info("Preprocesador", "Entrada: " + input_dir + " (imagenes *_seg.png)");
    Logger::instance().info("Preprocesador", "Salida: " + output_dir + " (imagenes *_pre.png, 224x224)");

    auto grupos = recolectarRutasPorGrupo(input_dir);

    if (grupos.empty()) {
        Logger::instance().error("Preprocesador", "No se encontraron imagenes *_seg.png en: " + input_dir);
        return 1;
    }

    int total_processed = 0;
    int total_omitidas = 0;
    int indiceGrupo = 0;

    for (auto& [key, rutasDelGrupo] : grupos) {
        ++indiceGrupo;

        // Filtrar las que ya tienen _pre.png guardado
        std::vector<fs::path> rutasPendientes;
        for (const auto& ruta : rutasDelGrupo) {
            if (yaProcesada(ruta, output_dir)) {
                ++total_omitidas;
            } else {
                rutasPendientes.push_back(ruta);
            }
        }

        if (rutasPendientes.empty()) {
            Logger::instance().debug("Preprocesador",
                "Grupo " + key.city + "/" + key.dish + "/" + key.angle + " ya completo — se omite.");
            continue;
        }

        // Helper local: extrae metadata de una ruta *_seg.png
        auto metadataDeRutaSeg = [](const fs::path& p) {
            std::string stem = p.stem().string();
            return parsearNombreArchivo(stem.substr(0, stem.size() - 4));
        };

        bool grupoTieneChef = std::any_of(rutasDelGrupo.begin(), rutasDelGrupo.end(),
            [&](const fs::path& p) { return metadataDeRutaSeg(p).esChef(); });
        bool chefEstaPendiente = std::any_of(rutasPendientes.begin(), rutasPendientes.end(),
            [&](const fs::path& p) { return metadataDeRutaSeg(p).esChef(); });

        // Si el grupo tiene Chef pero ya fue procesado en una corrida
        // anterior (por eso no aparece en rutasPendientes), los alumnos
        // pendientes de este grupo se preprocesaran SIN alinear en esta
        // corrida — procesarGrupo no vuelve a cargar al Chef ya guardado.
        if (grupoTieneChef && !chefEstaPendiente) {
            Logger::instance().warn("Preprocesador",
                "Grupo " + key.city + "/" + key.dish + "/" + key.angle +
                ": el Chef ya fue procesado antes y no se recarga — los alumnos pendientes de este grupo se alinearan sin referencia.");
        }

        Logger::instance().info("Preprocesador",
            "--- Grupo " + std::to_string(indiceGrupo) + "/" + std::to_string(grupos.size()) +
            ": " + key.city + "/" + key.dish + "/" + key.angle +
            " (" + std::to_string(rutasPendientes.size()) + " pendiente(s)) ---");

        try {
            // 'images' vive solo dentro de este bloque — al terminar
            // cada iteracion del for, su destructor libera la memoria
            // de este grupo antes de cargar el siguiente.
            std::vector<std::unique_ptr<ProcessedImage>> images = cargarGrupo(rutasPendientes);

            if (images.empty()) {
                Logger::instance().warn("Preprocesador", "Grupo vacio (fallo de carga en todas), se omite.");
                continue;
            }

            total_processed += procesarGrupo(key, images, output_dir);

        } catch (const std::exception& e) {
            Logger::instance().error("Preprocesador",
                "Error en grupo " + key.city + "/" + key.dish + "/" + key.angle + ": " + std::string(e.what()));
            // continua con el siguiente grupo en vez de abortar todo
        }
    }

    if (total_omitidas > 0) {
        Logger::instance().info("Preprocesador",
            std::to_string(total_omitidas) + " imagen(es) ya procesadas se omitieron.");
    }

    Logger::instance().info("Preprocesador",
        "Pipeline completado: " + std::to_string(total_processed) + " imagenes preprocesadas en " +
        std::to_string(grupos.size()) + " grupo(s).");

    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -I../scripts_genericos/include \
//       ../scripts_genericos/src/Logger.cpp \
//       ../scripts_genericos/src/PathManager.cpp \
//       ../scripts_genericos/src/VideoMetadata.cpp \
//       Preprocesador.cpp -o preprocesador \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./preprocesador [ruta_segmentadas] [ruta_salida]