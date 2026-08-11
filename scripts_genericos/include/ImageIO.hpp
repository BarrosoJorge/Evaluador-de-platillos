#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <memory>
#include <optional>
#include <string>

namespace evaluador {

// Representa una imagen cargada junto con su ruta de origen, para no perder
// la trazabilidad (nombre uniforme -> region/feature/score) a lo largo del
// pipeline.
struct ImagenCargada {
    std::filesystem::path ruta;
    cv::Mat mat;

    bool valida() const { return !mat.empty(); }
};

// Envoltura sobre OpenCV para centralizar lectura/escritura de imagenes.
// No usa punteros crudos: cv::Mat ya maneja su propia memoria por
// referencia contada internamente, y las colecciones se devuelven en
// std::vector.
class ImageIO {
public:
    // Carga una sola imagen. Devuelve std::nullopt si no existe o no se
    // pudo decodificar.
    static std::optional<ImagenCargada> cargar(const std::filesystem::path& ruta);

    // Carga todas las imagenes de una carpeta (no recursivo) con las
    // extensiones dadas. Las que fallan al cargar se omiten y se registran
    // como warning en el Logger, no detienen el proceso.
    static std::vector<ImagenCargada> cargarCarpeta(
        const std::filesystem::path& carpeta,
        const std::vector<std::string>& extensiones = {".jpg", ".jpeg", ".png"}
    );

    // Guarda una imagen en disco, creando la carpeta destino si no existe.
    static bool guardar(const cv::Mat& imagen, const std::filesystem::path& rutaDestino);

    // Extrae N frames distribuidos uniformemente a lo largo de un video.
    // Devuelve las imagenes en memoria; quien llame decide si las guarda
    // (util para VideoToImage).
    static std::vector<cv::Mat> extraerFramesUniformes(
        const std::filesystem::path& rutaVideo,
        int numeroFrames
    );

private:
    static bool tieneExtensionValida(
        const std::filesystem::path& ruta,
        const std::vector<std::string>& extensiones
    );
};

} // namespace evaluador
