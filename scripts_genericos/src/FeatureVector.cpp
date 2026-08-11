#include "FeatureVector.hpp"
#include "Logger.hpp"

#include <fstream>
#include <sstream>

namespace evaluador {

bool guardarFeaturesPorRegion(const std::vector<RegionFeatureVector>& regiones,
                               const std::filesystem::path& ruta) {
    std::error_code ec;
    std::filesystem::create_directories(ruta.parent_path(), ec);

    std::ofstream archivo(ruta);
    if (!archivo.is_open()) {
        Logger::instance().error("FeatureVector", "No se pudo escribir: " + ruta.string());
        return false;
    }

    archivo << "# features_por_region\n";
    archivo << "# nombre;mediaB;mediaG;mediaR;stdB;stdG;stdR;"
            << "energia;contraste;correlacion;homogeneidad;idf;entropia_glcm;varianza_glcm;"
            << "lbp_energia;lbp_entropia;lbp_varianza;lbp_media\n";

    for (const auto& r : regiones) {
        archivo << r.nombreRegion << ";"
                << r.color.mediaB << ";" << r.color.mediaG << ";" << r.color.mediaR << ";"
                << r.color.stdB << ";" << r.color.stdG << ";" << r.color.stdR << ";"
                << r.textura.energia << ";" << r.textura.contraste << ";"
                << r.textura.correlacion << ";" << r.textura.homogeneidad << ";"
                << r.textura.idf << ";" << r.textura.entropia << ";" << r.textura.varianza << ";"
                << r.lbp.energia << ";" << r.lbp.entropia << ";"
                << r.lbp.varianza << ";" << r.lbp.media << "\n";
    }

    Logger::instance().info("FeatureVector",
        "Features por region guardados (" + std::to_string(regiones.size()) + " regiones): " + ruta.string());
    return true;
}

std::optional<std::vector<RegionFeatureVector>> cargarFeaturesPorRegion(const std::filesystem::path& ruta) {
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) return std::nullopt;

    std::vector<RegionFeatureVector> resultado;
    std::string linea;

    while (std::getline(archivo, linea)) {
        // Ignorar líneas vacías, comentarios y cabeceras
        if (linea.empty() || linea.find("#") != std::string::npos) continue;
        
        // Solo procesar líneas que empiezan con "zona_"
        if (linea.find("zona_") == std::string::npos) continue;

        std::vector<std::string> campos;
        std::stringstream ss(linea); 
        std::string item;
        while (std::getline(ss, item, ';')) campos.push_back(item);

        // Validar que tengamos los 18 campos
        if (campos.size() != 18) continue;

        try {
            RegionFeatureVector r;
            size_t i = 0;
            r.nombreRegion = campos[i++];
            r.color.mediaB = std::stod(campos[i++]); 
            r.color.mediaG = std::stod(campos[i++]); 
            r.color.mediaR = std::stod(campos[i++]);
            r.color.stdB = std::stod(campos[i++]); 
            r.color.stdG = std::stod(campos[i++]); 
            r.color.stdR = std::stod(campos[i++]);
            r.textura.energia = std::stod(campos[i++]);
            r.textura.contraste = std::stod(campos[i++]);
            r.textura.correlacion = std::stod(campos[i++]);
            r.textura.homogeneidad = std::stod(campos[i++]);
            r.textura.idf = std::stod(campos[i++]);
            r.textura.entropia = std::stod(campos[i++]);
            r.textura.varianza = std::stod(campos[i++]);
            r.lbp.energia = std::stod(campos[i++]);
            r.lbp.entropia = std::stod(campos[i++]);
            r.lbp.varianza = std::stod(campos[i++]);
            r.lbp.media = std::stod(campos[i++]);
            
            resultado.push_back(r);
        } catch (...) {
            continue; // Si algo falla en el parseo, saltamos la línea
        }
    }
    return resultado.empty() ? std::nullopt : std::optional(resultado);
}

bool guardarFeaturesGlobal(const GlobalFeatureVector& global, const std::filesystem::path& ruta) {
    std::error_code ec;
    std::filesystem::create_directories(ruta.parent_path(), ec);

    std::ofstream archivo(ruta);
    if (!archivo.is_open()) {
        Logger::instance().error("FeatureVector", "No se pudo escribir: " + ruta.string());
        return false;
    }

    archivo << "# features_globales\n";
    archivo << "# simetria;limpieza;enfoque;volumen_general\n";
    archivo << global.simetria << ";" 
            << global.limpieza << ";" 
            << global.enfoque << ";" 
            << global.volumenGeneral << "\n";

    return true;
}

std::optional<GlobalFeatureVector> cargarFeaturesGlobal(const std::filesystem::path& ruta) {
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) return std::nullopt;

    std::string linea;
    while (std::getline(archivo, linea)) {
        // Buscamos la línea que tiene exactamente los 4 valores globales 
        // (excluyendo cabeceras que tienen letras)
        if (linea.find(";") == std::string::npos || std::isalpha(linea.front())) continue;

        std::vector<std::string> campos;
        std::stringstream ss(linea); std::string item;
        while (std::getline(ss, item, ';')) campos.push_back(item);

        if (campos.size() == 4) {
            GlobalFeatureVector global;
            try {
                global.simetria = std::stod(campos[0]);
                global.limpieza = std::stod(campos[1]);
                global.enfoque = std::stod(campos[2]);
                global.volumenGeneral = std::stod(campos[3]);
                return global;
            } catch (...) { continue; }
        }
    }
    return std::nullopt;
}

cv::Mat regionAVector(const RegionFeatureVector& r) {
    cv::Mat v(1, 17, CV_64F);
    int i = 0;
    v.at<double>(0, i++) = r.color.mediaB;
    v.at<double>(0, i++) = r.color.mediaG;
    v.at<double>(0, i++) = r.color.mediaR;
    v.at<double>(0, i++) = r.color.stdB;
    v.at<double>(0, i++) = r.color.stdG;
    v.at<double>(0, i++) = r.color.stdR;
    v.at<double>(0, i++) = r.textura.energia;
    v.at<double>(0, i++) = r.textura.contraste;
    v.at<double>(0, i++) = r.textura.correlacion;
    v.at<double>(0, i++) = r.textura.homogeneidad;
    v.at<double>(0, i++) = r.textura.idf;
    v.at<double>(0, i++) = r.textura.entropia;
    v.at<double>(0, i++) = r.textura.varianza;
    v.at<double>(0, i++) = r.lbp.energia;
    v.at<double>(0, i++) = r.lbp.entropia;
    v.at<double>(0, i++) = r.lbp.varianza;
    v.at<double>(0, i++) = r.lbp.media;
    return v;
}

cv::Mat globalAVector(const GlobalFeatureVector& g, const std::vector<std::string>& /*ordenRegiones*/) {
    // Creamos un vector de 1 fila x 4 columnas
    cv::Mat v(1, 4, CV_64F);
    v.at<double>(0, 0) = g.simetria;
    v.at<double>(0, 1) = g.limpieza;
    v.at<double>(0, 2) = g.enfoque;
    v.at<double>(0, 3) = g.volumenGeneral;
    return v;
}

cv::Mat colorAVector(const EstadisticosColor& c) {
    cv::Mat v(1, 6, CV_64F);
    v.at<double>(0, 0) = c.mediaB; v.at<double>(0, 1) = c.mediaG; v.at<double>(0, 2) = c.mediaR;
    v.at<double>(0, 3) = c.stdB;   v.at<double>(0, 4) = c.stdG;   v.at<double>(0, 5) = c.stdR;
    return v;
}

cv::Mat texturaAVector(const FeaturesGLCM& t) {
    cv::Mat v(1, 7, CV_64F);
    v.at<double>(0, 0) = t.energia;
    v.at<double>(0, 1) = t.contraste;
    v.at<double>(0, 2) = t.correlacion;
    v.at<double>(0, 3) = t.homogeneidad;
    v.at<double>(0, 4) = t.idf;
    v.at<double>(0, 5) = t.entropia;
    v.at<double>(0, 6) = t.varianza;
    return v;
}

cv::Mat lbpAVector(const EstadisticosHistograma& l) {
    cv::Mat v(1, 4, CV_64F);
    v.at<double>(0, 0) = l.energia;
    v.at<double>(0, 1) = l.entropia;
    v.at<double>(0, 2) = l.varianza;
    v.at<double>(0, 3) = l.media;
    return v;
}

} // namespace evaluador
