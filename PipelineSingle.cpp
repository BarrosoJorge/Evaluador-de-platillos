/*  PipelineSingle.cpp
 *
 *  Prueba integral del pipeline completo sobre UNA sola imagen.
 *
 *  Pasos que ejecuta:
 *    1. Segmentación automática con GrabCut (ROI central al 70%)
 *       + postprocesamiento morfológico + inpainting Telea
 *    2. Redimensionamiento a 224×224
 *    3. Corrección de orientación (minAreaRect)
 *    4. Extracción de características:
 *         · Raw    — estadísticas de histograma de píxeles
 *         · SDH    — histogramas de suma y diferencia
 *         · GLCM   — matrices de co-ocurrencia (d=1,3,7 | θ=0,45,90,135°)
 *         · LBP    — Local Binary Pattern
 *         · HOG    — Histograma de gradientes orientados
 *         · Color  — estadísticas HSV + CIE L*a*b*
 *
 *  Salida:
 *    Data/Processed/Image/
 *      <stem>_seg.png    — imagen segmentada
 *      <stem>_mask.png   — máscara binaria
 *      <stem>_pre.png    — imagen preprocesada (224×224)
 *    Data/Processed/Features/
 *      features_raw.csv, features_sdh.csv, features_glcm.csv,
 *      features_lbp.csv, features_hog.csv, features_color.csv
 *      maps/<stem>/<extractor>_<feature>_w<N>.png
 *
 *  Compilación:
 *      g++ -std=c++17 -O2 -o PipelineSingle PipelineSingle.cpp \
 *          `pkg-config --cflags --libs opencv4`
 *
 *  Uso:
 *      ./PipelineSingle <ruta_imagen> [ruta_salida_imagen] [ruta_salida_features]
 *      ./PipelineSingle "Data/Raw/Imagenes/CDMX/Limon/Lateral/Chef/CDMX_Limon_Lateral_Chef_0.jpg"
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>

// Header compartido con todas las utilidades de extracción de características
#include "scripts de caracteristicas/Caracteristicas.h"

namespace fs = std::filesystem;

// Tamaños de ventana para la prueba (subconjunto para terminar en tiempo razonable)
// El extractor completo usa {3,5,7,9,11,13,15,17,19,21,23,25}
static const std::vector<int> TEST_WINDOWS = {3, 11, 25};
static constexpr int SDH_BINS_TEST = 64;   // bins para SDH (64 en vez de 256, más rápido)

// TIMER UTILITARIO

struct Timer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double elapsed() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
};

static void printStep(const std::string& msg) {
    std::cout << "\n[" << msg << "]" << std::endl;
}

// ETAPA 1 — SEGMENTACIÓN AUTOMÁTICA

// GrabCut con ROI generado automáticamente al 70% central de la imagen
cv::Mat segmentAuto(const cv::Mat& image,
                    cv::Mat& out_mask,
                    cv::Mat& out_segmented)
{
    // ROI: margen del 15% en cada lado
    const int mx = static_cast<int>(image.cols * 0.15);
    const int my = static_cast<int>(image.rows * 0.15);
    cv::Rect roi(mx, my, image.cols - 2*mx, image.rows - 2*my);

    // GrabCut
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_model, fgd_model;
    cv::grabCut(image, mask, roi, bgd_model, fgd_model, 5, cv::GC_INIT_WITH_RECT);
    cv::Mat binary = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);

    // Apertura morfológica — elimina ruido
    cv::Mat kernel_open  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,  5));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::Mat opened, closed;
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN,  kernel_open,  cv::Point(-1,-1), 2);
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, kernel_close, cv::Point(-1,-1), 3);
    out_mask = closed;

    // Aplicar máscara
    cv::Mat masked = cv::Mat::zeros(image.size(), image.type());
    image.copyTo(masked, out_mask);

    // Inpainting en huecos internos
    std::vector<cv::Point> fg_pts;
    cv::findNonZero(out_mask, fg_pts);
    if (!fg_pts.empty()) {
        cv::Rect bbox = cv::boundingRect(fg_pts);
        cv::Mat interior = cv::Mat::zeros(out_mask.size(), CV_8UC1);
        cv::rectangle(interior, bbox, cv::Scalar(255), cv::FILLED);
        cv::Mat holes;
        cv::bitwise_and(interior, ~out_mask, holes);
        if (cv::countNonZero(holes) > 0 &&
            cv::countNonZero(holes) < bbox.area() * 0.35) {
            cv::Mat inpainted;
            cv::inpaint(masked, holes, inpainted, 5, cv::INPAINT_TELEA);
            out_segmented = cv::Mat::zeros(inpainted.size(), inpainted.type());
            inpainted.copyTo(out_segmented, out_mask);
            return out_segmented;
        }
    }
    out_segmented = masked;
    return out_segmented;
}

// ETAPA 2 — PREPROCESAMIENTO (resize + orientación)

cv::Mat resizeUniform224(const cv::Mat& image)
{
    cv::Mat r;
    cv::resize(image, r, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
    return r;
}

cv::Mat correctOrientationLocal(const cv::Mat& image)
{
    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image.clone();

    cv::Mat binary;
    cv::threshold(gray, binary, 1, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return image.clone();

    auto it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b){ return cv::contourArea(a) < cv::contourArea(b); });
    if (cv::contourArea(*it) < 500.0) return image.clone();

    cv::RotatedRect rr = cv::minAreaRect(*it);
    double angle = rr.angle;
    if (rr.size.width < rr.size.height) angle += 90.0;

    // El PDF corrige inclinaciones pequeñas del platillo; ángulos grandes indican
    // detección errónea (máscara con bordes irregulares → minAreaRect no confiable).
    if (std::abs(angle) < 1.0) return image.clone();
    if (std::abs(angle) > 30.0) {
        std::cout << "  Ángulo " << angle << "° fuera de rango — no se corrige." << std::endl;
        return image.clone();
    }

    std::cout << "  Ángulo corregido: " << angle << "°" << std::endl;

    // Canvas expandido — usar |cos| y |sin| para que no sean negativos en ningún cuadrante
    const double rad   = angle * CV_PI / 180.0;
    const int    new_w = static_cast<int>(std::ceil(
        image.cols * std::abs(std::cos(rad)) + image.rows * std::abs(std::sin(rad))));
    const int    new_h = static_cast<int>(std::ceil(
        image.cols * std::abs(std::sin(rad)) + image.rows * std::abs(std::cos(rad))));

    // Rotar alrededor del centro de la imagen (no del contorno)
    const cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, -angle, 1.0);
    rot.at<double>(0,2) += new_w / 2.0 - center.x;
    rot.at<double>(1,2) += new_h / 2.0 - center.y;

    cv::Mat result;
    cv::warpAffine(image, result, rot, cv::Size(new_w, new_h),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    // Recortar al bounding box del contenido no-negro para no dejar márgenes vacíos
    cv::Mat g2;
    if (result.channels() == 3) cv::cvtColor(result, g2, cv::COLOR_BGR2GRAY);
    else g2 = result.clone();
    cv::Mat bin2;
    cv::threshold(g2, bin2, 1, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> nz_pts;
    cv::findNonZero(bin2, nz_pts);
    if (!nz_pts.empty()) result = result(cv::boundingRect(nz_pts)).clone();

    return result;
}

// ETAPA 3 — EXTRACTORES DE CARACTERÍSTICAS (versión test, ventanas reducidas)

// ── RAW ─────────────────────────────────────────────────────────────────────

std::vector<cv::Mat> rawFeatureMaps(const cv::Mat& gray, int w)
{
    const int out_r = gray.rows - w + 1, out_c = gray.cols - w + 1;
    std::vector<cv::Mat> maps(7, cv::Mat::zeros(out_r, out_c, CV_64F));
    const std::string names[] = {"Energia","Contraste","Correlacion",
                                  "Homogeneidad","IDF","Entropia","Varianza"};
    slideWindow(gray, w, [&](const cv::Mat& roi, int r, int c) {
        auto h = computeHist1D(roi);
        const double mu  = histMean(h);
        const double var = histVariance(h, mu);
        maps[0].at<double>(r,c) = histEnergy(h);
        maps[1].at<double>(r,c) = var;
        maps[2].at<double>(r,c) = histCorrelation(h);
        maps[3].at<double>(r,c) = histHomogeneity(h, mu);
        maps[4].at<double>(r,c) = histIDF(h, mu);
        maps[5].at<double>(r,c) = histEntropy(h);
        maps[6].at<double>(r,c) = var;
    });
    return maps;
}

// ── SDH ─────────────────────────────────────────────────────────────────────

std::vector<cv::Mat> sdhFeatureMaps(const cv::Mat& gray, int w)
{
    const int out_r = gray.rows - w + 1, out_c = gray.cols - w + 1;
    std::vector<cv::Mat> maps(7, cv::Mat::zeros(out_r, out_c, CV_64F));

    slideWindow(gray, w, [&](const cv::Mat& roi, int r, int c) {
        // Histograma conjunto P_sd con SDH_BINS_TEST bins
        const int B = SDH_BINS_TEST;
        std::vector<std::vector<double>> P(B, std::vector<double>(B, 0.0));
        int cnt = 0;
        for (int rr = 0; rr < roi.rows; ++rr)
            for (int cc = 0; cc < roi.cols; ++cc) {
                auto add = [&](int p1, int p2) {
                    int s = (p1 + p2) / 2 * (B-1) / 255;
                    int d = std::abs(p1-p2) * (B-1) / 255;
                    P[s][d] += 1.0; ++cnt;
                };
                int p1 = roi.at<uchar>(rr, cc);
                if (cc+1 < roi.cols) add(p1, roi.at<uchar>(rr, cc+1));
                if (rr+1 < roi.rows) add(p1, roi.at<uchar>(rr+1, cc));
            }
        if (cnt > 0) for (auto& row : P) for (auto& v : row) v /= cnt;

        std::vector<double> Ps(B,0.0), Pd(B,0.0);
        for (int s=0;s<B;++s) for (int d=0;d<B;++d) { Ps[s]+=P[s][d]; Pd[d]+=P[s][d]; }

        double mu_s=0, mu_d=0;
        for (int k=0;k<B;++k) { mu_s+=k*Ps[k]; mu_d+=k*Pd[k]; }

        double var_s=0, var_d=0;
        for (int k=0;k<B;++k) { var_s+=(k-mu_s)*(k-mu_s)*Ps[k]; var_d+=(k-mu_d)*(k-mu_d)*Pd[k]; }
        double sg_s=std::sqrt(var_s), sg_d=std::sqrt(var_d);

        double contraste=0, homog=0, cov=0, cs=0, cp=0;
        for (int d=0;d<B;++d) contraste += (double)d*d*Pd[d];
        for (int d=0;d<B;++d) homog    += Pd[d]/(1.0+d);
        for (int s=0;s<B;++s) for (int d=0;d<B;++d) cov += (s-mu_s)*(d-mu_d)*P[s][d];
        for (int s=0;s<B;++s) for (int d=0;d<B;++d) {
            double t = (s+d) - mu_s - mu_d;
            cs += t*t*t*P[s][d];
            cp += t*t*t*t*P[s][d];
        }
        double corr = (sg_s>1e-10 && sg_d>1e-10) ? cov/(sg_s*sg_d) : 0.0;

        maps[0].at<double>(r,c) = mu_s;
        maps[1].at<double>(r,c) = var_s;
        maps[2].at<double>(r,c) = corr;
        maps[3].at<double>(r,c) = contraste;
        maps[4].at<double>(r,c) = homog;
        maps[5].at<double>(r,c) = cs;
        maps[6].at<double>(r,c) = cp;
    });
    return maps;
}

// ── GLCM ────────────────────────────────────────────────────────────────────

// Genera 12 grupos de 7 mapas (12 configs GLCM × 7 Haralick)
std::vector<cv::Mat> glcmFeatureMaps(const cv::Mat& gray, int w)
{
    const int out_r = gray.rows - w + 1, out_c = gray.cols - w + 1;
    // 12 configs × 7 features = 84 mapas
    std::vector<cv::Mat> maps(84, cv::Mat::zeros(out_r, out_c, CV_64F));

    slideWindow(gray, w, [&](const cv::Mat& roi, int r, int c) {
        int idx = 0;
        for (int dist : GLCM_DISTANCES) {
            for (double angle : GLCM_ANGLES) {
                cv::Mat glcm = computeGLCM(roi, dist, angle, GLCM_LEVELS);
                GLCMFeatures f = extractGLCMFeatures(glcm);
                auto v = f.toVector();
                for (int k = 0; k < 7; ++k)
                    maps[idx*7 + k].at<double>(r,c) = v[k];
                ++idx;
            }
        }
    });
    return maps;
}

// ── LBP ─────────────────────────────────────────────────────────────────────

std::vector<cv::Mat> lbpFeatureMaps(const cv::Mat& lbp_img, int w)
{
    const int out_r = lbp_img.rows - w + 1, out_c = lbp_img.cols - w + 1;
    std::vector<cv::Mat> maps(5, cv::Mat::zeros(out_r, out_c, CV_64F));
    slideWindow(lbp_img, w, [&](const cv::Mat& roi, int r, int c) {
        auto h = computeHist1D(roi);
        const double mu  = histMean(h);
        const double var = histVariance(h, mu);
        maps[0].at<double>(r,c) = mu;
        maps[1].at<double>(r,c) = var;
        maps[2].at<double>(r,c) = histCorrelation(h);
        maps[3].at<double>(r,c) = var;
        maps[4].at<double>(r,c) = histHomogeneity(h, mu);
    });
    return maps;
}

// ── HOG ─────────────────────────────────────────────────────────────────────

static constexpr int HOG_CELL = 8, HOG_BINS = 9, HOG_BLOCK = 2;

std::vector<double> computeHOG(const cv::Mat& gray)
{
    cv::Mat mag, ang;
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 1);
    cv::magnitude(gx, gy, mag);
    cv::phase(gx, gy, ang, true);
    for (int r=0;r<ang.rows;++r) for (int c=0;c<ang.cols;++c) {
        float& a = ang.at<float>(r,c); if (a>=180.f) a-=180.f;
    }

    const int nx = gray.cols / HOG_CELL, ny = gray.rows / HOG_CELL;
    std::vector<std::vector<std::vector<double>>> ch(ny,
        std::vector<std::vector<double>>(nx, std::vector<double>(HOG_BINS, 0.0)));

    const double bw = 180.0 / HOG_BINS;
    for (int cy=0;cy<ny;++cy) for (int cx=0;cx<nx;++cx) {
        cv::Rect roi(cx*HOG_CELL, cy*HOG_CELL, HOG_CELL, HOG_CELL);
        for (int r=0;r<HOG_CELL;++r) for (int c=0;c<HOG_CELL;++c) {
            double m = mag(roi).at<float>(r,c);
            double a = ang(roi).at<float>(r,c);
            int b0 = static_cast<int>(a/bw) % HOG_BINS;
            int b1 = (b0+1) % HOG_BINS;
            double w1 = a/bw - std::floor(a/bw);
            ch[cy][cx][b0] += m*(1-w1);
            ch[cy][cx][b1] += m*w1;
        }
    }

    std::vector<double> desc;
    for (int by=0;by<ny-HOG_BLOCK+1;++by) for (int bx=0;bx<nx-HOG_BLOCK+1;++bx) {
        std::vector<double> bv;
        for (int dy=0;dy<HOG_BLOCK;++dy) for (int dx=0;dx<HOG_BLOCK;++dx)
            bv.insert(bv.end(), ch[by+dy][bx+dx].begin(), ch[by+dy][bx+dx].end());
        double norm = 0; for (double v:bv) norm+=v*v; norm=std::sqrt(norm+1e-8);
        for (double& v:bv) v/=norm;
        desc.insert(desc.end(), bv.begin(), bv.end());
    }
    return desc;
}

// ── COLOR ───────────────────────────────────────────────────────────────────

std::vector<cv::Mat> colorFeatureMaps(const cv::Mat& bgr_img, int w)
{
    // 6 canales × 4 stats + 3 global = 27 mapas
    std::vector<cv::Mat> channels;
    cv::Mat hsv;
    cv::cvtColor(bgr_img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvch(3);
    cv::split(hsv, hsvch);
    hsvch[0].convertTo(hsvch[0], CV_8U, 255.0/179.0);

    cv::Mat bgr_f, lab_f;
    bgr_img.convertTo(bgr_f, CV_32F, 1.0/255.0);
    cv::cvtColor(bgr_f, lab_f, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labch(3);
    cv::split(lab_f, labch);
    cv::Mat L8, a8, b8;
    labch[0].convertTo(L8, CV_8U, 255.0/100.0);
    labch[1].convertTo(a8, CV_8U, 1.0, 127.0);
    labch[2].convertTo(b8, CV_8U, 1.0, 127.0);

    channels = {hsvch[0], hsvch[1], hsvch[2], L8, a8, b8};

    const int out_r = bgr_img.rows - w + 1, out_c = bgr_img.cols - w + 1;
    std::vector<cv::Mat> maps(27, cv::Mat::zeros(out_r, out_c, CV_64F));

    for (int r=0;r<out_r;++r) for (int c=0;c<out_c;++c) {
        int k = 0;
        std::vector<double> vars;
        for (int ch=0;ch<6;++ch) {
            auto h = computeHist1D(channels[ch](cv::Rect(c,r,w,w)));
            double mu = histMean(h), var = histVariance(h, mu);
            maps[k++].at<double>(r,c) = mu;
            maps[k++].at<double>(r,c) = std::sqrt(var);
            maps[k++].at<double>(r,c) = histEntropy(h);
            maps[k++].at<double>(r,c) = histIDF(h, mu);
            if (ch<3) vars.push_back(var);
        }
        maps[k++].at<double>(r,c) = vars[0]+vars[1]+vars[2];
        {
            auto hs = computeHist1D(channels[1](cv::Rect(c,r,w,w)));
            maps[k++].at<double>(r,c) = histMean(hs);
        }
        {
            auto hl = computeHist1D(channels[3](cv::Rect(c,r,w,w)));
            double ml = histMean(hl);
            maps[k++].at<double>(r,c) = histVariance(hl, ml);
        }
    }
    return maps;
}

// GUARDADO DE RESULTADOS

void writeCSVRow(const std::string& csv_path,
                 const std::string& label,
                 const std::string& filename,
                 const std::vector<std::string>& feat_names,
                 const std::vector<double>& values)
{
    // Si el archivo no existe, escribir cabecera
    bool exists = fs::exists(csv_path);
    std::ofstream f(csv_path, std::ios::app);
    if (!exists) {
        f << "label,filename";
        for (const auto& n : feat_names) f << "," << n << "_mean," << n << "_std";
        f << "\n";
    }
    f << label << "," << filename;
    for (double v : values) f << "," << v;
    f << "\n";
}

void saveMapsAndRow(const std::vector<cv::Mat>& maps,
                    const std::vector<std::string>& feat_names,
                    int window_size,
                    const std::string& extractor,
                    const std::string& img_stem,
                    const std::string& maps_dir,
                    std::vector<double>& row_accum)
{
    const fs::path out_dir = fs::path(maps_dir) / img_stem;
    fs::create_directories(out_dir);

    for (size_t k = 0; k < maps.size(); ++k) {
        double mn, sd;
        mapStats(maps[k], mn, sd);
        row_accum.push_back(mn);
        row_accum.push_back(sd);

        // Guardar mapa normalizado como PNG
        const std::string fname = extractor + "_" + feat_names[k % feat_names.size()]
                                + "_w" + std::to_string(window_size) + ".png";
        cv::imwrite((out_dir / fname).string(), normalizeToImage(maps[k]));
    }
}

// FUNCIÓN PRINCIPAL

// Ejecuta una prueba integral del pipeline para una imagen.
// Devuelve 0 si termina correctamente.
int main(int argc, char** argv)
{
    // Rutas relativas a la raiz del proyecto.
    std::string img_path = argc >= 2 ? argv[1]
        : "Data/Raw/Imagenes/CDMX/Limon/Lateral/Chef/CDMX_Limon_Lateral_Chef_0.jpg";
    std::string out_image_dir = argc >= 3 ? argv[2]
        : "Data/Processed/Image";
    std::string out_feat_dir  = argc >= 4 ? argv[3]
        : "Data/Processed/Features";

    const std::string maps_dir = out_feat_dir + "/maps";

    std::cout << "Pipeline single" << std::endl;
    std::cout << "Imagen:   " << img_path << std::endl;
    std::cout << "Imágenes: " << out_image_dir << std::endl;
    std::cout << "Features: " << out_feat_dir  << std::endl;
    std::cout << "Ventanas: ";
    for (int w : TEST_WINDOWS) std::cout << w << " ";
    std::cout << std::endl;

    // Verifica que la imagen de entrada exista.
    if (!fs::exists(img_path)) {
        std::cerr << "Error: no se encontró la imagen: " << img_path << std::endl;
        return 1;
    }

    fs::create_directories(out_image_dir);
    fs::create_directories(maps_dir);

    const std::string stem = fs::path(img_path).stem().string();
    const std::string label = "CDMX_Limon_Chef";    // etiqueta manual para 1 imagen

    // Carga la imagen original.
    cv::Mat original = cv::imread(img_path);
    if (original.empty()) {
        std::cerr << "No se pudo cargar: " << img_path << std::endl;
        return 1;
    }
    std::cout << "\nImagen cargada: " << original.cols << "×" << original.rows << " px" << std::endl;

    // Etapa 1: segmentacion.
    Timer t1;
    printStep("ETAPA 1 — Segmentación automática GrabCut");
    cv::Mat mask, segmented;
    segmentAuto(original, mask, segmented);
    std::cout << "  Tiempo: " << t1.elapsed() << "s" << std::endl;

    cv::imwrite(out_image_dir + "/" + stem + "_mask.png", mask);
    cv::imwrite(out_image_dir + "/" + stem + "_seg.png",  segmented);
    std::cout << "  Guardado: _seg.png + _mask.png" << std::endl;

    // Etapa 2: preprocesamiento.
    Timer t2;
    printStep("ETAPA 2 — Preprocesamiento (crop → resize 224×224 → orientación)");

    // Paso 2a: recortar al bounding box del contenido segmentado con margen del 5%
    // Esto hace que el platillo llene el cuadro, igual que en el PDF.
    cv::Mat seg_crop;
    {
        cv::Mat seg_gray;
        cv::cvtColor(segmented, seg_gray, cv::COLOR_BGR2GRAY);
        cv::Mat bin;
        cv::threshold(seg_gray, bin, 1, 255, cv::THRESH_BINARY);
        std::vector<cv::Point> pts;
        cv::findNonZero(bin, pts);
        if (!pts.empty()) {
            cv::Rect bb = cv::boundingRect(pts);
            // Margen del 5% en cada lado (para no cortar bordes del plato)
            int pad_x = static_cast<int>(bb.width  * 0.05);
            int pad_y = static_cast<int>(bb.height * 0.05);
            bb.x      = std::max(0, bb.x - pad_x);
            bb.y      = std::max(0, bb.y - pad_y);
            bb.width  = std::min(segmented.cols - bb.x, bb.width  + 2 * pad_x);
            bb.height = std::min(segmented.rows - bb.y, bb.height + 2 * pad_y);
            seg_crop = segmented(bb).clone();
            std::cout << "  Crop: " << bb << std::endl;
        } else {
            seg_crop = segmented.clone();
        }
    }

    // Paso 2b: resize uniforme a 224×224
    cv::Mat resized = resizeUniform224(seg_crop);

    // Paso 2c: corrección de orientación (solo si ángulo < 30°)
    cv::Mat preprocessed = correctOrientationLocal(resized);

    // Resize final a exactamente 224×224 si la orientación expandió el canvas
    if (preprocessed.cols != 224 || preprocessed.rows != 224)
        cv::resize(preprocessed, preprocessed, cv::Size(224,224), 0,0, cv::INTER_AREA);

    cv::imwrite(out_image_dir + "/" + stem + "_pre.png", preprocessed);
    std::cout << "  Guardado: _pre.png (" << preprocessed.cols << "×" << preprocessed.rows << " px)" << std::endl;
    std::cout << "  Tiempo: " << t2.elapsed() << "s" << std::endl;

    // Imagen gris para extractores de textura
    cv::Mat gray;
    cv::cvtColor(preprocessed, gray, cv::COLOR_BGR2GRAY);

    // Etapa 3: extraccion de caracteristicas.

    // ─ 3A. RAW ───────────────────────────────────────────────────────────────
    {
        Timer t;
        printStep("ETAPA 3A — Características Raw");
        const std::vector<std::string> names = {
            "Energia","Contraste","Correlacion","Homogeneidad","IDF","Entropia","Varianza"};
        std::vector<double> row;
        for (int w : TEST_WINDOWS) {
            std::cout << "  w=" << w << "..." << std::flush;
            auto maps = rawFeatureMaps(gray, w);
            saveMapsAndRow(maps, names, w, "Raw", stem, maps_dir, row);
            std::cout << " OK" << std::endl;
        }
        std::vector<std::string> feat_names;
        for (int w : TEST_WINDOWS) for (auto& n : names) feat_names.push_back("Raw_w"+std::to_string(w)+"_"+n);
        writeCSVRow(out_feat_dir+"/features_raw.csv", label, stem, feat_names, row);
        std::cout << "  Tiempo: " << t.elapsed() << "s" << std::endl;
    }

    // ─ 3B. SDH ───────────────────────────────────────────────────────────────
    {
        Timer t;
        printStep("ETAPA 3B — Histogramas de Suma y Diferencia (SDH)");
        const std::vector<std::string> names = {
            "Media","Varianza","Correlacion","Contraste","Homogeneidad","ClusterSombra","ClusterProminencia"};
        std::vector<double> row;
        for (int w : TEST_WINDOWS) {
            std::cout << "  w=" << w << " (bins=" << SDH_BINS_TEST << ")..." << std::flush;
            auto maps = sdhFeatureMaps(gray, w);
            saveMapsAndRow(maps, names, w, "SDH", stem, maps_dir, row);
            std::cout << " OK" << std::endl;
        }
        std::vector<std::string> feat_names;
        for (int w : TEST_WINDOWS) for (auto& n : names) feat_names.push_back("SDH_w"+std::to_string(w)+"_"+n);
        writeCSVRow(out_feat_dir+"/features_sdh.csv", label, stem, feat_names, row);
        std::cout << "  Tiempo: " << t.elapsed() << "s" << std::endl;
    }

    // ─ 3C. GLCM ──────────────────────────────────────────────────────────────
    {
        Timer t;
        printStep("ETAPA 3C — GLCM (3 dist × 4 ángulos = 12 matrices)");
        std::vector<double> row;
        std::vector<std::string> feat_names;
        for (int w : TEST_WINDOWS) {
            std::cout << "  w=" << w << "..." << std::flush;
            auto maps = glcmFeatureMaps(gray, w);

            // Nombre de cada uno de los 84 mapas
            int idx = 0;
            for (int d : GLCM_DISTANCES) for (double a : GLCM_ANGLES) {
                auto hnames = GLCMFeatures::names();
                for (auto& hn : hnames) {
                    std::string fname = "GLCM_d"+std::to_string(d)+"_a"+std::to_string((int)a)+"_"+hn;
                    double mn, sd;
                    mapStats(maps[idx], mn, sd);
                    row.push_back(mn); row.push_back(sd);

                    // Guardar solo d=1,θ=0 para no saturar el disco
                    if (d == 1 && a == 0.0)
                        saveFeatureMap(maps[idx], maps_dir, stem,
                                       "GLCM_d1_a0_"+hn, w);

                    feat_names.push_back("GLCM_w"+std::to_string(w)+"_"+fname);
                    ++idx;
                }
            }
            std::cout << " OK" << std::endl;
        }
        writeCSVRow(out_feat_dir+"/features_glcm.csv", label, stem, feat_names, row);
        std::cout << "  Tiempo: " << t.elapsed() << "s" << std::endl;
    }

    // ─ 3D. LBP ───────────────────────────────────────────────────────────────
    {
        Timer t;
        printStep("ETAPA 3D — LBP (imagen LBP + ventaneo)");
        const cv::Mat lbp_img = computeLBPImage(gray);
        cv::imwrite(out_image_dir + "/" + stem + "_lbp.png", lbp_img);
        std::cout << "  Imagen LBP guardada: " << lbp_img.cols << "×" << lbp_img.rows << std::endl;

        const std::vector<std::string> names = {"Media","Varianza","Correlacion","Contraste","Homogeneidad"};
        std::vector<double> row;
        for (int w : TEST_WINDOWS) {
            if (lbp_img.rows < w || lbp_img.cols < w) continue;
            std::cout << "  w=" << w << "..." << std::flush;
            auto maps = lbpFeatureMaps(lbp_img, w);
            saveMapsAndRow(maps, names, w, "LBP", stem, maps_dir, row);
            std::cout << " OK" << std::endl;
        }
        std::vector<std::string> feat_names;
        for (int w : TEST_WINDOWS) for (auto& n : names) feat_names.push_back("LBP_w"+std::to_string(w)+"_"+n);
        writeCSVRow(out_feat_dir+"/features_lbp.csv", label, stem, feat_names, row);
        std::cout << "  Tiempo: " << t.elapsed() << "s" << std::endl;
    }

    // ─ 3E. HOG ───────────────────────────────────────────────────────────────
    {
        Timer t;
        printStep("ETAPA 3E — HOG (descriptor global)");
        auto desc = computeHOG(gray);
        std::cout << "  Dimensión descriptor: " << desc.size() << std::endl;

        // Gradiente magnitud como mapa visual
        cv::Mat gx, gy, mag;
        cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
        cv::Sobel(gray, gy, CV_32F, 0, 1, 1);
        cv::magnitude(gx, gy, mag);
        cv::imwrite(out_image_dir + "/" + stem + "_grad_mag.png", normalizeToImage(mag));

        // CSV
        std::vector<std::string> feat_names;
        for (size_t k = 0; k < desc.size(); ++k) feat_names.push_back("HOG_"+std::to_string(k));
        std::ofstream f(out_feat_dir+"/features_hog.csv");
        f << "label,filename";
        for (auto& n : feat_names) f << "," << n;
        f << "\n" << label << "," << stem;
        for (double v : desc) f << "," << v;
        f << "\n";
        std::cout << "  Tiempo: " << t.elapsed() << "s" << std::endl;
    }

    // ─ 3F. COLOR ─────────────────────────────────────────────────────────────
    {
        Timer t;
        printStep("ETAPA 3F — Color (HSV + CIE L*a*b*)");
        std::vector<std::string> ch_names = {"H","S","V","L","a","b"};
        std::vector<std::string> st_names = {"mean","std","entropy","idf"};
        const std::vector<std::string> global_names = {
            "var_total_color","saturacion_media","contraste_luminosidad"};

        std::vector<double> row;
        std::vector<std::string> feat_names;
        for (int w : TEST_WINDOWS) {
            std::cout << "  w=" << w << "..." << std::flush;
            auto maps = colorFeatureMaps(preprocessed, w);
            // Nombres de los 27 mapas (igual orden que colorFeatureMaps)
            std::vector<std::string> map_names;
            for (auto& ch : ch_names) for (auto& st : st_names) map_names.push_back(ch+"_"+st);
            for (auto& gn : global_names) map_names.push_back(gn);

            saveMapsAndRow(maps, map_names, w, "Color", stem, maps_dir, row);
            for (auto& n : map_names)
                feat_names.push_back("Color_w"+std::to_string(w)+"_"+n);
            std::cout << " OK" << std::endl;
        }
        writeCSVRow(out_feat_dir+"/features_color.csv", label, stem, feat_names, row);
        std::cout << "  Tiempo: " << t.elapsed() << "s" << std::endl;
    }

    // Resumen de salida.
    std::cout << std::endl;
    std::cout << "Pipeline completado" << std::endl;
    std::cout << "Imágenes guardadas en: " << out_image_dir << std::endl;
    std::cout << "  " << stem << "_seg.png   (segmentada)"  << std::endl;
    std::cout << "  " << stem << "_mask.png  (máscara)"     << std::endl;
    std::cout << "  " << stem << "_pre.png   (224×224)"     << std::endl;
    std::cout << "  " << stem << "_lbp.png   (imagen LBP)"  << std::endl;
    std::cout << "  " << stem << "_grad_mag.png (gradiente)" << std::endl;
    std::cout << "\nCSVs en: " << out_feat_dir << std::endl;
    std::cout << "  features_raw.csv | features_sdh.csv | features_glcm.csv" << std::endl;
    std::cout << "  features_lbp.csv | features_hog.csv | features_color.csv" << std::endl;
    std::cout << "\nMapas PNG en: " << maps_dir << "/" << stem << "/" << std::endl;

    return 0;
}
