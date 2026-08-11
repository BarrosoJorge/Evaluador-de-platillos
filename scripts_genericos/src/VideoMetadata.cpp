#include "VideoMetadata.hpp"

#include <vector>
#include <cctype>
#include <algorithm>

namespace evaluador {

std::string VideoMetadata::formatear() const {
    std::string resultado;
        if (!variante.empty()) resultado += variante + "_";
        resultado += ciudad + "_" + platillo + "_" + angulo + "_" + autor;
        if (!calidad.empty()) resultado += "_" + calidad;
        return resultado;
}

VideoMetadata parsearNombreArchivo(const std::string& nombreArchivo) {
    VideoMetadata metadata;

    std::string nombre = nombreArchivo;
    size_t punto = nombre.find_last_of('.');
    if (punto != std::string::npos) nombre = nombre.substr(0, punto);

    // Detectar y separar el prefijo "N_" que agrega Delabeler contra
    // colisiones (ej. "1_CDMX_..." -> variante="1", resto="CDMX_...").
    // Solo cuenta si TODO lo que precede al primer '_' son digitos.
    size_t primerGuion = nombre.find('_');
    if (primerGuion != std::string::npos) {
        std::string posibleContador = nombre.substr(0, primerGuion);
        bool esNumerico = !posibleContador.empty() &&
            std::all_of(posibleContador.begin(), posibleContador.end(), ::isdigit);
        if (esNumerico) {
            metadata.variante = posibleContador;
            nombre = nombre.substr(primerGuion + 1);
        }
    }

    // Separar por '_'
    std::vector<std::string> partes;
    size_t inicio = 0, fin = 0;
    while ((fin = nombre.find('_', inicio)) != std::string::npos) {
        partes.push_back(nombre.substr(inicio, fin - inicio));
        inicio = fin + 1;
    }
    partes.push_back(nombre.substr(inicio));

    if (partes.size() >= 4) {
        metadata.ciudad = partes[0];
        metadata.platillo = partes[1];
        metadata.angulo = partes[2];
        metadata.autor = partes[3];
        // Guard: la 5a parte solo es calidad si es literalmente B/R/M.
        // Sin esto, un indice de frame como "..._Chef_3" se leeria
        // como calidad="3", que es incorrecto (Chef no tiene calidad).
        if (partes.size() >= 5 &&
            (partes[4] == "B" || partes[4] == "R" || partes[4] == "M")) {
            metadata.calidad = partes[4];
        }
    }
    // Si partes.size() < 4, metadata queda vacio (todos los campos "")
    // y metadata.vacio() lo reflejara correctamente.

    return metadata;
}

std::optional<std::string> validarMetadata(const VideoMetadata& metadata) {
    if (metadata.vacio()) {
        return "Metadatos incompletos: se esperaban al menos 4 partes "
               "(Ciudad_Platillo_Angulo_Autor)";
    }

    if (metadata.angulo != "Superior" && metadata.angulo != "Lateral") {
        return "Angulo invalido: '" + metadata.angulo +
               "' (debe ser 'Superior' o 'Lateral')";
    }

    if (metadata.autor != "Chef" && metadata.autor != "Estudiante") {
        return "Autor invalido: '" + metadata.autor +
               "' (debe ser 'Chef' o 'Estudiante')";
    }

    if (metadata.autor == "Estudiante") {
        if (metadata.calidad != "B" && metadata.calidad != "R" && metadata.calidad != "M") {
            return "Calidad invalida: '" + metadata.calidad +
                   "' (debe ser 'B', 'R' o 'M' para autor Estudiante)";
        }
    } else if (!metadata.calidad.empty()) {
        return "Autor 'Chef' no debe traer calidad especificada (trae: '" +
               metadata.calidad + "')";
    }

    return std::nullopt;
}

} // namespace evaluador
