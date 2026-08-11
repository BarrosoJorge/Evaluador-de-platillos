/*  GrabCutSegmentador.cpp
 *
 *  Etapa 1 del pipeline — Aislamiento del platillo respecto al fondo.
 *
 *  Es un paso previo a Preprocesador.cpp, no la segmentación multi-región
 *  de la etapa 2 (esa la hacen Segmentador.cpp / Localizador.cpp). Aquí
 *  solo se separa "platillo completo" de "fondo" con una sola región —
 *  el resultado (*_seg.png) es lo que Preprocesador usa como entrada
 *  para redimensionar/orientar/alinear, y más adelante Segmentador vuelve
 *  a tomar esa imagen ya preprocesada para dividirla en regiones nombradas.
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
 *  Entrada:  Data/Imagenes/Crudas/  (jerarquía Ciudad/Platillo/Angulo/Autor[/Calidad]/)
 *  Salida:   Data/Segmentadas/      (misma jerarquía; guarda *_mask.png y *_seg.png)
 *
 *  Cambios respecto a tu version original:
 *    - ImageMetadata + parseFilename se movieron a scripts_genericos como
 *      VideoMetadata/parsearNombreArchivo. Tu guard de "solo tomar
 *      partes[4] como calidad si es B/R/M" era correcto y lo tenias
 *      ANTES que yo — el VideoMetadata que te entregue en la etapa 0
 *      no lo tenia, se lo agregue ahora tomandolo de aqui.
 *    - applyGrabCut/morphologicalPostprocess/applyMaskToImage/inpaintHoles/
 *      scaleForDisplay se movieron a scripts_genericos/SegmentacionUtils,
 *      porque Segmentador.cpp y Localizador.cpp (etapa 2) necesitan
 *      exactamente la misma logica de GrabCut y no tenia sentido
 *      copiarla una tercera vez.
 *    - Ruta de entrada por defecto: cambie "Data/Raw/Imagenes" por
 *      PathManager::imagenesCrudas() (Data/Imagenes/Crudas), para que
 *      quede conectada con lo que VideoToImage ya produce.
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -I../scripts_genericos/include \
 *          ../scripts_genericos/src/Logger.cpp \
 *          ../scripts_genericos/src/PathManager.cpp \
 *          ../scripts_genericos/src/VideoMetadata.cpp \
 *          ../scripts_genericos/src/SegmentacionUtils.cpp \
 *          GrabCutSegmentador.cpp -o grabcut_seg \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./grabcut_seg [--auto] [ruta_entrada] [ruta_salida]
 */

#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"
#include "SegmentacionUtils.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace evaluador;

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

// DishImage se queda local (no es generico): carga el estado especifico de
// esta etapa (mask, segmented, processed) que ImageIO no necesita conocer.
struct DishImage
{
    cv::Mat original;          // Imagen original cargada de disco
    cv::Mat mask;              // Máscara binaria final: 255 = platillo, 0 = fondo
    cv::Mat segmented;         // Imagen con fondo negro (máscara aplicada + inpainting)
    std::string file_path;     // Ruta completa al archivo fuente
    std::string filename;      // Nombre del archivo (con extensión)
    VideoMetadata metadata;    // Metadatos extraídos del nombre (modulo compartido)
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

std::vector<fs::path> recolectarRutas(const std::string& directory);
std::vector<std::unique_ptr<DishImage>> cargarLote(const std::vector<fs::path>& rutas);
bool yaProcesada(const fs::path& rutaImagen, const std::string& output_dir);

bool saveSegmented(const DishImage& img, const std::string& output_dir);
void interactiveSegmentation(std::vector<std::unique_ptr<DishImage>>& images,
                              const std::string& output_dir);

// ============================================================================
// IMPLEMENTACIONES — CARGA DE DATOS
// ============================================================================

// Solo LISTA las rutas de imagenes soportadas — no las carga en memoria.
// Es la parte barata; separarla de la carga real es lo que permite
// procesar por lotes sin tener todo el dataset en RAM a la vez.
std::vector<fs::path> recolectarRutas(const std::string& directory)
{
    std::vector<fs::path> rutas;

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        Logger::instance().error("GrabCutSegmentador", "Directorio invalido: " + directory);
        return rutas;
    }

    const std::vector<std::string> supported_exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
    };

    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        bool supported = std::any_of(
            supported_exts.begin(), supported_exts.end(),
            [&ext](const std::string& e) { return ext == e; });

        if (supported) rutas.push_back(entry.path());
    }

    Logger::instance().info("GrabCutSegmentador", "Total imagenes encontradas: " + std::to_string(rutas.size()));
    return rutas;
}

// Verifica si YA existe el _seg.png de salida para esta imagen, sin
// tener que cargarla (el metadato sale del nombre de archivo). Permite
// reanudar despues de una interrupcion (ej. el OOM kill) sin repetir
// trabajo ya guardado en disco.
bool yaProcesada(const fs::path& rutaImagen, const std::string& output_dir)
{
    VideoMetadata metadata = parsearNombreArchivo(rutaImagen.filename().string());
    if (metadata.vacio()) return false; // si no se puede parsear, mejor intentar procesarla

    fs::path out = fs::path(output_dir) / metadata.ciudad / metadata.platillo / metadata.angulo / metadata.autor;
    fs::path rutaSeg = out / (rutaImagen.stem().string() + "_seg.png");

    return fs::exists(rutaSeg);
}

// Carga en memoria SOLO las rutas que se le pasen — pensado para
// llamarse con un lote de tamaño N, no con el dataset completo. El
// vector devuelto se libera automaticamente al salir de scope en
// main(), lo que baja el pico de memoria de "todo el dataset" a
// "un lote a la vez".
std::vector<std::unique_ptr<DishImage>> cargarLote(const std::vector<fs::path>& rutas)
{
    std::vector<std::unique_ptr<DishImage>> images;
    images.reserve(rutas.size());

    for (const auto& ruta : rutas) {
        try {
            auto img = std::make_unique<DishImage>(ruta.string());
            img->metadata = parsearNombreArchivo(img->filename);
            Logger::instance().debug("GrabCutSegmentador", "Cargada: " + img->filename);
            images.push_back(std::move(img));
        } catch (const std::exception& e) {
            Logger::instance().warn("GrabCutSegmentador",
                "No se pudo cargar " + ruta.filename().string() + ": " + e.what());
        }
    }

    return images;
}

// ============================================================================
// IMPLEMENTACIONES — GUARDADO
// ============================================================================

// Guarda la máscara (*_mask.png) y la imagen segmentada (*_seg.png)
// reproduciendo la jerarquía Ciudad/Platillo/Angulo/Autor/ en output_dir.
bool saveSegmented(const DishImage& img, const std::string& output_dir)
{
    fs::path out = fs::path(output_dir)
                   / img.metadata.ciudad
                   / img.metadata.platillo
                   / img.metadata.angulo
                   / img.metadata.autor;

    try {
        fs::create_directories(out);
    } catch (const std::exception& e) {
        Logger::instance().error("GrabCutSegmentador", "Error creando directorio: " + std::string(e.what()));
        return false;
    }

    const std::string stem = fs::path(img.filename).stem().string();

    bool ok = true;
    ok &= cv::imwrite((out / (stem + "_mask.png")).string(), img.mask);
    ok &= cv::imwrite((out / (stem + "_seg.png")).string(),  img.segmented);

    if (ok)
        Logger::instance().info("GrabCutSegmentador", "Guardado: " + (out / stem).string() + " [_mask.png + _seg.png]");
    else
        Logger::instance().error("GrabCutSegmentador", "Error al escribir archivos para: " + stem);

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
        Logger::instance().info("GrabCutSegmentador", "No hay imagenes para procesar.");
        return;
    }

    std::cout << "\nControles de GrabCut Segmentador:" << std::endl;
    std::cout << "  Arrastra el mouse para delimitar el ROI del platillo" << std::endl;
    std::cout << "  ESPACIO / ENTER : confirmar ROI y aplicar segmentación" << std::endl;
    std::cout << "  C / ESC         : cancelar — omitir imagen actual" << std::endl;
    std::cout << "  Q (ventana resultado) : interrumpir y guardar lo procesado" << std::endl;

    // Se recuerda la ultima forma elegida como default para la
    // siguiente imagen — no todos los platillos son rectangulares
    // (la mayoria de los platos son redondos), asi que se pregunta
    // por imagen en vez de fijarla una sola vez para toda la sesion.
    FormaROI formaActual = FormaROI::Rectangulo;

    for (size_t i = 0; i < images.size(); ++i) {
        DishImage* img = images[i].get();

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Imagen " << (i + 1) << " de " << images.size()
                  << ": " << img->filename << std::endl;
        if (!img->metadata.vacio()) {
            std::cout << "  [" << img->metadata.ciudad << " / "
                      << img->metadata.platillo << " / "
                      << img->metadata.angulo << " / "
                      << img->metadata.autor << "]" << std::endl;
        }
        std::cout << std::string(70, '=') << std::endl;

        std::cout << "Forma del ROI — (r)ectangulo / (e)lipse ["
                  << (formaActual == FormaROI::Rectangulo ? "r" : "e") << "]: ";
        std::string entradaForma;
        std::getline(std::cin, entradaForma);
        if (!entradaForma.empty()) {
            char c = static_cast<char>(std::tolower(entradaForma[0]));
            if (c == 'e') formaActual = FormaROI::Elipse;
            else if (c == 'r') formaActual = FormaROI::Rectangulo;
        }

        // Escalar para que quepa en pantalla manteniendo la proporción
        cv::Mat display = escalarParaVista(img->original);
        const double sx = static_cast<double>(img->original.cols) / display.cols;
        const double sy = static_cast<double>(img->original.rows) / display.rows;

        const std::string etiquetaForma = (formaActual == FormaROI::Rectangulo) ? "Rectangulo" : "Elipse";

        // selectROI: OpenCV muestra la imagen y captura el rectángulo del usuario.
        // En modo Elipse, el rectangulo que dibujas sigue siendo el
        // "delimitador" — la elipse se inscribe dentro de el, no
        // dibujas la elipse directamente (selectROI no soporta eso).
        const std::string win_name = "GrabCut [" + etiquetaForma + "] - " + img->filename
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

        std::cout << "Aplicando GrabCut (5 iteraciones, forma " << etiquetaForma << ") + postprocesamiento..." << std::endl;

        const cv::Mat binary  = aplicarGrabCut(img->original, roi, 5, formaActual);
        const cv::Mat clean   = postprocesamientoMorfologico(binary);
        const cv::Mat masked  = aplicarMascaraAImagen(img->original, clean);
        const cv::Mat final_img = rellenarHuecos(masked, clean);

        img->mask      = clean;
        img->segmented = final_img;
        img->processed = true;

        // Preparar comparación lado a lado: Original | Segmentada
        cv::Mat left  = escalarParaVista(img->original, 450, 400);
        cv::Mat right = escalarParaVista(final_img,     450, 400);

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

        cv::imshow("Resultado - cualquier tecla para continuar, Q para salir", comparison);
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

    Logger::instance().info("GrabCutSegmentador",
        "Segmentacion finalizada: " + std::to_string(done) + " de " + std::to_string(images.size()) + " imagenes procesadas.");
}

// ============================================================================
// MODO AUTOMÁTICO — sin GUI
// ============================================================================

// Segmenta todas las imágenes sin intervención del usuario.
// Usa un ROI central al 70% de la imagen (margen del 15% en cada lado).
// Equivalente al modo interactivo pero no requiere pantalla ni ratón.
//
// Nota: un ROI central fijo asume que el platillo esta razonablemente
// centrado en cada foto. Si tus fotos varian mucho en encuadre, este
// modo va a dar mascaras peores que el interactivo — sirve para
// pruebas rapidas o lotes grandes donde la supervision manual no es
// viable, no como reemplazo permanente del modo interactivo.
void autoSegmentAll(std::vector<std::unique_ptr<DishImage>>& images,
                    const std::string& output_dir)
{
    if (images.empty()) {
        Logger::instance().info("GrabCutSegmentador", "No hay imagenes para procesar.");
        return;
    }

    Logger::instance().info("GrabCutSegmentador", "Modo automatico — ROI central al 70%");

    for (size_t i = 0; i < images.size(); ++i) {
        DishImage* img = images[i].get();
        Logger::instance().debug("GrabCutSegmentador",
            "[" + std::to_string(i + 1) + "/" + std::to_string(images.size()) + "] " + img->filename);

        // ROI: margen 15% en cada lado
        const int mx = static_cast<int>(img->original.cols * 0.15);
        const int my = static_cast<int>(img->original.rows * 0.15);
        const cv::Rect roi(mx, my,
                           img->original.cols - 2 * mx,
                           img->original.rows - 2 * my);

        cv::Mat bin_mask = aplicarGrabCut(img->original, roi);
        img->mask        = postprocesamientoMorfologico(bin_mask);
        cv::Mat masked   = aplicarMascaraAImagen(img->original, img->mask);
        img->segmented   = rellenarHuecos(masked, img->mask);
        img->processed   = true;

        saveSegmented(*img, output_dir);
    }

    const long done = std::count_if(images.begin(), images.end(),
        [](const std::unique_ptr<DishImage>& d){ return d->processed; });
    Logger::instance().info("GrabCutSegmentador",
        "Total: " + std::to_string(done) + " de " + std::to_string(images.size()) + " imagenes segmentadas.");
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char** argv)
{
    PathManager rutas(".");
    std::string input_dir  = rutas.imagenesCrudas().string();
    std::string output_dir = rutas.segmentadas().string();
    bool auto_mode = false;
    size_t tamanoLote = 30; // imagenes cargadas en memoria a la vez

    // Parsear argumentos: [--auto] [--lote N] [input_dir] [output_dir]
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--auto") { auto_mode = true; continue; }
        if (arg == "--lote" && i + 1 < argc) { tamanoLote = std::stoul(argv[++i]); continue; }
        if (pos == 0) { input_dir  = arg; ++pos; }
        else          { output_dir = arg; ++pos; }
    }

    Logger::instance().info("GrabCutSegmentador", "Modo: " + std::string(auto_mode ? "automatico (--auto)" : "interactivo"));
    Logger::instance().info("GrabCutSegmentador", "Entrada: " + input_dir);
    Logger::instance().info("GrabCutSegmentador", "Salida: " + output_dir);
    Logger::instance().info("GrabCutSegmentador", "Tamaño de lote: " + std::to_string(tamanoLote));

    std::vector<fs::path> todasLasRutas = recolectarRutas(input_dir);
    if (todasLasRutas.empty()) {
        Logger::instance().error("GrabCutSegmentador", "No se encontraron imagenes en: " + input_dir);
        return 1;
    }

    // Filtrar las que ya tienen _seg.png guardado — permite reanudar
    // sin repetir trabajo despues de una interrupcion.
    std::vector<fs::path> rutasPendientes;
    for (const auto& ruta : todasLasRutas) {
        if (yaProcesada(ruta, output_dir)) {
            Logger::instance().debug("GrabCutSegmentador", "Ya procesada, se omite: " + ruta.filename().string());
        } else {
            rutasPendientes.push_back(ruta);
        }
    }

    size_t saltadas = todasLasRutas.size() - rutasPendientes.size();
    if (saltadas > 0) {
        Logger::instance().info("GrabCutSegmentador",
            std::to_string(saltadas) + " imagen(es) ya procesadas se omiten. Pendientes: " + std::to_string(rutasPendientes.size()));
    }

    if (rutasPendientes.empty()) {
        Logger::instance().info("GrabCutSegmentador", "Nada pendiente por procesar.");
        return 0;
    }

    size_t totalLotes = (rutasPendientes.size() + tamanoLote - 1) / tamanoLote;

    for (size_t lote = 0; lote < totalLotes; ++lote) {
        size_t inicio = lote * tamanoLote;
        size_t fin = std::min(inicio + tamanoLote, rutasPendientes.size());
        std::vector<fs::path> rutasDeEsteLote(rutasPendientes.begin() + inicio, rutasPendientes.begin() + fin);

        Logger::instance().info("GrabCutSegmentador",
            "--- Lote " + std::to_string(lote + 1) + "/" + std::to_string(totalLotes) +
            " (" + std::to_string(rutasDeEsteLote.size()) + " imagenes) ---");

        try {
            // 'images' vive solo dentro de este bloque — al terminar
            // cada iteracion del for, su destructor libera toda la
            // memoria de este lote antes de cargar el siguiente.
            std::vector<std::unique_ptr<DishImage>> images = cargarLote(rutasDeEsteLote);

            if (images.empty()) {
                Logger::instance().warn("GrabCutSegmentador", "Lote vacio (fallo de carga en todas), se omite.");
                continue;
            }

            if (auto_mode)
                autoSegmentAll(images, output_dir);
            else
                interactiveSegmentation(images, output_dir);

        } catch (const std::exception& e) {
            Logger::instance().error("GrabCutSegmentador",
                "Error en lote " + std::to_string(lote + 1) + ": " + std::string(e.what()));
            // continua con el siguiente lote en vez de abortar todo
        }
    }

    Logger::instance().info("GrabCutSegmentador", "Proceso completo: " + std::to_string(totalLotes) + " lote(s).");

    return 0;
}

// Compilación:
//   g++ -std=c++17 -O2 -I../scripts_genericos/include \
//       ../scripts_genericos/src/Logger.cpp \
//       ../scripts_genericos/src/PathManager.cpp \
//       ../scripts_genericos/src/VideoMetadata.cpp \
//       GrabCutSegmentador.cpp -o grabcut_seg \
//       `pkg-config --cflags --libs opencv4`
//
// Uso:
//   ./grabcut_seg [--auto] [ruta_entrada] [ruta_salida]