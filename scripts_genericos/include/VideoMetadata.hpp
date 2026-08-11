#pragma once

#include <string>
#include <optional>

namespace evaluador {

// Metadatos codificados en el nombre de archivo segun la convencion del
// proyecto: Ciudad_Platillo_Angulo_Autor_Calidad (Calidad es opcional,
// solo aplica cuando autor == "Estudiante").
//
// Antes esta misma estructura y su logica de parseo/formato estaban
// duplicadas en Labeler.cpp y VideoToImage.cpp. Vive aqui una sola vez
// para que ambos scripts (y cualquier otro futuro, como el localizador
// de la etapa 2) usen exactamente la misma convencion sin poder
// desincronizarse entre si.
struct VideoMetadata {
    std::string ciudad;
    std::string platillo;
    std::string angulo;
    std::string autor;
    std::string calidad; // vacio si no aplica (ej. autor == "Chef")
    std::string variante; // contador anti-colision de Delabeler (ej. "1", "2"), vacio si no aplica

    // Reconstruye el nombre base (sin extension), ej:
    // "Queretaro_Milanesa_Superior_Estudiante_B"
    std::string formatear() const;

    // Un VideoMetadata se considera "vacio" (parseo fallido) si le
    // faltan los 4 campos obligatorios.
    bool vacio() const {
        return ciudad.empty() || platillo.empty() || angulo.empty() || autor.empty();
    }

    bool esChef() const { return autor == "Chef"; }
    bool esEstudiante() const { return autor == "Estudiante"; }
};

// Parsea un nombre de archivo (con o sin extension), separando por '_'.
// Si no hay al menos 4 partes, devuelve un VideoMetadata vacio — quien
// llame debe revisar metadata.vacio() antes de usarlo.
//
// La 5a parte SOLO se toma como calidad si es exactamente "B", "R" o
// "M". Esto es necesario porque VideoToImage agrega un indice de frame
// al final del nombre (ej. "..._Chef_3.jpg"): sin este guard, ese "3"
// se leeria como si fuera la calidad. Si tu nombre trae un indice de
// frame despues de la calidad (ej. "..._Estudiante_B_3.jpg"), esa
// sexta parte simplemente se ignora — no forma parte del metadato.
VideoMetadata parsearNombreArchivo(const std::string& nombreArchivo);

// Valida que los metadatos cumplan la convencion esperada del proyecto:
//   - angulo en {"Superior", "Lateral"}
//   - autor en {"Chef", "Estudiante"}
//   - calidad en {"B", "R", "M"} solo cuando autor == "Estudiante"
//   - Chef no debe traer calidad
//
// Devuelve std::nullopt si es valido, o un mensaje describiendo el
// primer problema encontrado si no lo es. Ajusta aqui los valores
// permitidos si tu convencion cambia — es el unico lugar donde se
// definen.
std::optional<std::string> validarMetadata(const VideoMetadata& metadata);

} // namespace evaluador
