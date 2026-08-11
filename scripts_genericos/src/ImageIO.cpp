#include "ImageIO.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <cctype>

namespace evaluador {

std::optional<ImagenCargada> ImageIO::cargar(const std::filesystem::path& ruta) {
    if (!std::filesystem::exists(ruta)) {
        Logger::instance().warn("ImageIO", "No existe la imagen: " + ruta.string());
        return std::nullopt;
    }

    cv::Mat mat = cv::imread(ruta.string(), cv::IMREAD_COLOR);
    if (mat.empty()) {
        Logger::instance().warn("ImageIO", "No se pudo decodificar: " + ruta.string());
        return std::nullopt;
    }

    return ImagenCargada{ruta, std::move(mat)};
}

std::vector<ImagenCargada> ImageIO::cargarCarpeta(
    const std::filesystem::path& carpeta,
    const std::vector<std::string>& extensiones
) {
    std::vector<ImagenCargada> resultado;

    if (!std::filesystem::exists(carpeta) || !std::filesystem::is_directory(carpeta)) {
        Logger::instance().error("ImageIO", "Carpeta invalida: " + carpeta.string());
        return resultado;
    }

    for (const auto& entrada : std::filesystem::directory_iterator(carpeta)) {
        if (!entrada.is_regular_file()) continue;
        if (!tieneExtensionValida(entrada.path(), extensiones)) continue;

        auto imagen = cargar(entrada.path());
        if (imagen.has_value()) {
            resultado.push_back(std::move(imagen.value()));
        }
    }

    Logger::instance().info("ImageIO",
        "Cargadas " + std::to_string(resultado.size()) + " imagenes desde " + carpeta.string());

    return resultado;
}

bool ImageIO::guardar(const cv::Mat& imagen, const std::filesystem::path& rutaDestino) {
    if (imagen.empty()) {
        Logger::instance().warn("ImageIO", "Intento de guardar imagen vacia: " + rutaDestino.string());
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(rutaDestino.parent_path(), ec);
    if (ec) {
        Logger::instance().error("ImageIO",
            "No se pudo crear carpeta destino " + rutaDestino.parent_path().string() + ": " + ec.message());
        return false;
    }

    bool exito = cv::imwrite(rutaDestino.string(), imagen);
    if (!exito) {
        Logger::instance().error("ImageIO", "Fallo al guardar: " + rutaDestino.string());
    }
    return exito;
}

std::vector<cv::Mat> ImageIO::extraerFramesUniformes(
    const std::filesystem::path& rutaVideo,
    int numeroFrames
) {
    std::vector<cv::Mat> frames;

    if (numeroFrames <= 0) {
        Logger::instance().warn("ImageIO", "numeroFrames <= 0, no se extrae nada.");
        return frames;
    }

    cv::VideoCapture captura(rutaVideo.string());
    if (!captura.isOpened()) {
        Logger::instance().error("ImageIO", "No se pudo abrir el video: " + rutaVideo.string());
        return frames;
    }

    int totalFrames = static_cast<int>(captura.get(cv::CAP_PROP_FRAME_COUNT));
    if (totalFrames <= 0) {
        Logger::instance().error("ImageIO", "Video sin frames legibles: " + rutaVideo.string());
        return frames;
    }

    int frameEfectivos = std::min(numeroFrames, totalFrames);
    frames.reserve(static_cast<size_t>(frameEfectivos));

    for (int i = 0; i < frameEfectivos; ++i) {
        // Distribucion uniforme evitando el primer y ultimo frame exactos
        // (suelen tener transiciones/desenfoque de camara).
        double proporcion = (frameEfectivos == 1)
            ? 0.5
            : static_cast<double>(i) / (frameEfectivos - 1);
        int indiceFrame = static_cast<int>(proporcion * (totalFrames - 1));

        captura.set(cv::CAP_PROP_POS_FRAMES, indiceFrame);

        cv::Mat frame;
        captura >> frame;

        if (frame.empty()) {
            Logger::instance().warn("ImageIO",
                "Frame vacio en indice " + std::to_string(indiceFrame) + " de " + rutaVideo.string());
            continue;
        }

        frames.push_back(frame.clone());
    }

    Logger::instance().info("ImageIO",
        "Extraidos " + std::to_string(frames.size()) + " frames de " + rutaVideo.string());

    return frames;
}

bool ImageIO::tieneExtensionValida(
    const std::filesystem::path& ruta,
    const std::vector<std::string>& extensiones
) {
    std::string ext = ruta.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return std::tolower(c); });

    return std::find(extensiones.begin(), extensiones.end(), ext) != extensiones.end();
}

} // namespace evaluador
