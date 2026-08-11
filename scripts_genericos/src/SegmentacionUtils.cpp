#include "SegmentacionUtils.hpp"
#include "Logger.hpp"

#include <algorithm>

namespace evaluador {

cv::Mat aplicarGrabCut(const cv::Mat& imagen, const cv::Rect& roi, int iteraciones, FormaROI forma) {
    // GrabCut no es un recorrido simple de pixeles: por cada iteracion
    // ajusta mezclas gaussianas y corre un corte minimo sobre un grafo
    // con un nodo por pixel. En fotos de celular a resolucion completa
    // (facilmente 3000x4000+ = 12 millones de pixeles), esto es lento
    // sin importar el lenguaje — el costo es algoritmico, no de C++.
    //
    // Se corre sobre una copia reducida (lado mayor <= 800px) y la
    // mascara resultante se escala de vuelta al tamano original. El
    // costo de GrabCut no baja linealmente con los pixeles — baja mucho
    // mas — asi que esto da una mejora de velocidad grande a cambio de
    // una perdida de precision en el borde de la mascara que la
    // limpieza morfologica posterior (postprocesamientoMorfologico)
    // ya absorbe sin problema.
    constexpr int LADO_MAXIMO_TRABAJO = 800;

    int ladoMayor = std::max(imagen.cols, imagen.rows);
    double escala = (ladoMayor > LADO_MAXIMO_TRABAJO)
        ? static_cast<double>(LADO_MAXIMO_TRABAJO) / ladoMayor
        : 1.0;

    cv::Mat imagenTrabajo;
    cv::Rect roiTrabajo;

    if (escala == 1.0) {
        imagenTrabajo = imagen;
        roiTrabajo = roi;
    } else {
        cv::resize(imagen, imagenTrabajo, cv::Size(), escala, escala, cv::INTER_AREA);
        roiTrabajo = cv::Rect(
            static_cast<int>(roi.x * escala),
            static_cast<int>(roi.y * escala),
            static_cast<int>(roi.width * escala),
            static_cast<int>(roi.height * escala)
        );
        roiTrabajo &= cv::Rect(0, 0, imagenTrabajo.cols, imagenTrabajo.rows);
    }

    cv::Mat mascaraTrabajo;
    cv::Mat modeloFondo, modeloFrente;

    if (forma == FormaROI::Rectangulo) {
        mascaraTrabajo = cv::Mat(imagenTrabajo.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
        cv::grabCut(imagenTrabajo, mascaraTrabajo, roiTrabajo, modeloFondo, modeloFrente,
                    iteraciones, cv::GC_INIT_WITH_RECT);
    } else {
        // Elipse inscrita en el rectangulo delimitador: no todos los
        // platillos son rectangulares (la mayoria de los platos son
        // redondos), y un rectangulo completo como "probable frente"
        // incluye innecesariamente las esquinas — la elipse arranca
        // GrabCut mas cerca de la forma real del plato.
        mascaraTrabajo = cv::Mat(imagenTrabajo.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
        cv::Point centro(roiTrabajo.x + roiTrabajo.width / 2, roiTrabajo.y + roiTrabajo.height / 2);
        cv::Size ejes(std::max(1, roiTrabajo.width / 2), std::max(1, roiTrabajo.height / 2));
        cv::ellipse(mascaraTrabajo, centro, ejes, 0.0, 0.0, 360.0, cv::Scalar(cv::GC_PR_FGD), cv::FILLED);

        cv::grabCut(imagenTrabajo, mascaraTrabajo, cv::Rect(), modeloFondo, modeloFrente,
                    iteraciones, cv::GC_INIT_WITH_MASK);
    }

    cv::Mat binariaTrabajo = (mascaraTrabajo == cv::GC_FGD) | (mascaraTrabajo == cv::GC_PR_FGD);

    if (escala == 1.0) {
        return binariaTrabajo;
    }

    // INTER_NEAREST al escalar de vuelta: evita introducir grises
    // intermedios en una mascara que debe seguir siendo binaria.
    cv::Mat binariaOriginal;
    cv::resize(binariaTrabajo, binariaOriginal, imagen.size(), 0, 0, cv::INTER_NEAREST);
    return binariaOriginal;
}

cv::Mat postprocesamientoMorfologico(const cv::Mat& mascaraBinaria, bool mantenerSoloElMasGrande) {
    cv::Mat kernelApertura = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernelCierre = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));

    cv::Mat abierta, cerrada;
    cv::morphologyEx(mascaraBinaria, abierta, cv::MORPH_OPEN, kernelApertura, cv::Point(-1, -1), 2);
    cv::morphologyEx(abierta, cerrada, cv::MORPH_CLOSE, kernelCierre, cv::Point(-1, -1), 3);

    if (!mantenerSoloElMasGrande) {
        // Algunos usos (ej. fusionar manualmente varias manchas del
        // mismo color/objeto, como 2 puntos de salsa separados)
        // necesitan conservar varios componentes desconectados a
        // proposito — quedarse solo con el mas grande los destruiria.
        return cerrada;
    }

    // Quedarse solo con el componente conexo mas grande. GrabCut no
    // distingue "platillo" de "objeto secundario dentro del ROI"
    // (cubiertos, props que siempre aparecen en la misma posicion por
    // un montaje de camara fijo) — ambos contrastan igual contra el
    // fondo negro. El platillo casi siempre es, por mucho, el blob mas
    // grande de la imagen; cualquier otro objeto separado (aunque haya
    // sobrevivido la apertura/cierre) es descartado aqui.
    cv::Mat etiquetas, estadisticas, centroides;
    int numComponentes = cv::connectedComponentsWithStats(cerrada, etiquetas, estadisticas, centroides, 8);

    if (numComponentes <= 1) {
        return cerrada;
    }

    int indiceMasGrande = 1;
    int areaMasGrande = estadisticas.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < numComponentes; ++i) {
        int area = estadisticas.at<int>(i, cv::CC_STAT_AREA);
        if (area > areaMasGrande) {
            areaMasGrande = area;
            indiceMasGrande = i;
        }
    }

    cv::Mat soloElMasGrande = (etiquetas == indiceMasGrande);
    cv::Mat resultado;
    soloElMasGrande.convertTo(resultado, CV_8UC1, 255);
    return resultado;
}

cv::Mat aplicarMascaraAImagen(const cv::Mat& imagen, const cv::Mat& mascara) {
    cv::Mat resultado = cv::Mat::zeros(imagen.size(), imagen.type());
    imagen.copyTo(resultado, mascara);
    return resultado;
}

cv::Mat rellenarHuecos(const cv::Mat& segmentada, const cv::Mat& mascara) {
    std::vector<cv::Point> puntosFrente;
    cv::findNonZero(mascara, puntosFrente);
    if (puntosFrente.empty()) return segmentada.clone();

    cv::Rect bbox = cv::boundingRect(puntosFrente);

    cv::Mat interior = cv::Mat::zeros(mascara.size(), CV_8UC1);
    cv::rectangle(interior, bbox, cv::Scalar(255), cv::FILLED);

    cv::Mat huecos;
    cv::bitwise_and(interior, ~mascara, huecos);

    const int cantidadHuecos = cv::countNonZero(huecos);
    if (cantidadHuecos == 0 || cantidadHuecos > bbox.area() * 0.35)
        return segmentada.clone();

    cv::Mat rellenada;
    cv::inpaint(segmentada, huecos, rellenada, 5, cv::INPAINT_TELEA);

    cv::Mat resultado = cv::Mat::zeros(rellenada.size(), rellenada.type());
    rellenada.copyTo(resultado, mascara);
    return resultado;
}

cv::Mat escalarParaVista(const cv::Mat& imagen, int maxAncho, int maxAlto) {
    if (imagen.empty()) return imagen;

    const double escala = std::min({
        static_cast<double>(maxAncho) / imagen.cols,
        static_cast<double>(maxAlto) / imagen.rows,
        1.0
    });

    if (escala == 1.0) return imagen.clone();

    cv::Mat vista;
    cv::resize(imagen, vista, cv::Size(), escala, escala, cv::INTER_AREA);
    return vista;
}

cv::Mat mascaraDesdeNoNegro(const cv::Mat& imagen) {
    cv::Mat gris;
    cv::cvtColor(imagen, gris, cv::COLOR_BGR2GRAY);
    cv::Mat binaria;
    cv::threshold(gris, binaria, 1, 255, cv::THRESH_BINARY);
    return binaria;
}

cv::Vec3d colorPromedioEnMascara(const cv::Mat& imagen, const cv::Mat& mascara) {
    cv::Scalar media = cv::mean(imagen, mascara);
    return cv::Vec3d(media[0], media[1], media[2]);
}

cv::Scalar colorDePaleta(int indice) {
    static const std::vector<cv::Scalar> paleta = {
        {255, 80, 80}, {80, 200, 255}, {80, 255, 120}, {255, 220, 80},
        {220, 80, 255}, {80, 255, 220}, {255, 140, 80}, {160, 160, 255},
        {200, 255, 80}, {255, 80, 200}
    };
    return paleta[indice % static_cast<int>(paleta.size())];
}

std::vector<ClusterColor> proponerClusters(const cv::Mat& imagen, const cv::Mat& mascaraArea,
                                            int k, cv::Mat& etiquetasPorPixel) {
    std::vector<cv::Point> puntos;
    cv::findNonZero(mascaraArea, puntos);

    etiquetasPorPixel = cv::Mat(imagen.size(), CV_32S, cv::Scalar(-1));

    std::vector<ClusterColor> clusters(k);
    for (auto& c : clusters) c.mascara = cv::Mat::zeros(imagen.size(), CV_8UC1);

    if (puntos.empty()) return clusters;

    cv::Mat imagenLab;
    cv::cvtColor(imagen, imagenLab, cv::COLOR_BGR2Lab);

    cv::Mat muestras(static_cast<int>(puntos.size()), 3, CV_32F);
    for (size_t i = 0; i < puntos.size(); ++i) {
        cv::Vec3b lab = imagenLab.at<cv::Vec3b>(puntos[i]);
        muestras.at<float>(static_cast<int>(i), 0) = lab[0];
        muestras.at<float>(static_cast<int>(i), 1) = lab[1];
        muestras.at<float>(static_cast<int>(i), 2) = lab[2];
    }

    cv::Mat etiquetas, centros;
    cv::kmeans(muestras, k, etiquetas,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.5),
        3, cv::KMEANS_PP_CENTERS, centros);

    for (size_t i = 0; i < puntos.size(); ++i) {
        int idx = etiquetas.at<int>(static_cast<int>(i));
        clusters[idx].mascara.at<uchar>(puntos[i]) = 255;
        etiquetasPorPixel.at<int>(puntos[i]) = idx;
    }

    for (int i = 0; i < k; ++i) {
        clusters[i].colorPromedioBGR = colorPromedioEnMascara(imagen, clusters[i].mascara);
    }

    return clusters;
}

} // namespace evaluador