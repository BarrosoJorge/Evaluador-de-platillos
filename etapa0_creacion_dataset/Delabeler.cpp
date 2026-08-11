/*
Delabeler: quita especificamente el token de CALIDAD (B/R/M) del nombre
de un video de alumno, cuando ya existe (ej. "Queretaro_Huevo_Superior_
Estudiante_B.mp4" -> "Queretaro_Huevo_Superior_Estudiante.mp4").

Cambio respecto a la version anterior de este archivo: antes era un
"quita lo que sea que siga al ultimo guion bajo" completamente
generico. Eso funcionaba por accidente para nombres de Estudiante (el
ultimo token SI es la calidad), pero rompia los nombres de Chef —
"..._Chef.MP4" no tiene calidad, y el stripper generico le habria
quitado "Chef" completo, dejando el video sin autor. Ahora usa
VideoMetadata::parsearNombreArchivo() (el mismo parser que ya usan
Labeler/VideoToImage) para saber con certeza si el ultimo token es
realmente una calidad valida (B/R/M) antes de tocar nada — a los
videos de Chef, o a nombres que no siguen la convencion todavia
(fuera de formato, con typos, etc.), NO se les quita nada.
*/

#include "Logger.hpp"
#include "PathManager.hpp"
#include "VideoMetadata.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;
using namespace evaluador;

const std::vector<std::string> EXTENSIONES_SOPORTADAS = {".mp4", ".mov", ".avi", ".mkv"};

bool esExtensionSoportada(const std::string& ext) {
    std::string minuscula = ext;
    std::transform(minuscula.begin(), minuscula.end(), minuscula.begin(), ::tolower);
    return std::find(EXTENSIONES_SOPORTADAS.begin(), EXTENSIONES_SOPORTADAS.end(), minuscula)
           != EXTENSIONES_SOPORTADAS.end();
}

// Quita el token de calidad SOLO si parsearNombreArchivo lo reconoce
// como una calidad valida (B, R o M). Devuelve "" si no hay nada que
// quitar (Chef, ya sin calidad, o nombre fuera de formato).
std::string quitarCalidad(const std::string& nombreBase) {
    VideoMetadata metadata = parsearNombreArchivo(nombreBase);

    if (metadata.calidad.empty()) return "";

    VideoMetadata sinCalidad = metadata;
    sinCalidad.calidad.clear();
    return sinCalidad.formatear();
}

int main(int argc, char* argv[]) {
    // PathManager centraliza donde viven los videos crudos. Si prefieres
    // apuntar a otra carpeta puntual, sigue funcionando el argumento por
    // linea de comandos como antes.
    PathManager rutas(".");
    std::string directorio = rutas.videosCrudos().string();
    if (argc >= 2) {
        directorio = argv[1];
    }

    Logger::instance().info("Delabeler", "Directorio: " + directorio);

    if (!fs::exists(directorio) || !fs::is_directory(directorio)) {
        Logger::instance().error("Delabeler", "Directorio no existe o no es valido: " + directorio);
        return 1;
    }

    std::vector<std::pair<fs::path, fs::path>> pendientes; // (viejo, nuevo)

    std::set<std::string> nombresReservadosEnEsteLote;
    for (const auto& entrada : fs::recursive_directory_iterator(directorio)) {
        if (!entrada.is_regular_file()) continue;
        
        std::string ext = entrada.path().extension().string();
        if (!esExtensionSoportada(ext)) continue;

        std::string nombreBase = entrada.path().stem().string();
        std::string nuevoNombreBase = quitarCalidad(nombreBase);

        if (nuevoNombreBase.empty()) {
            Logger::instance().debug("Delabeler", "[SIN CALIDAD] " + entrada.path().filename().string());
            continue;
        }

        fs::path candidato = entrada.path().parent_path() / (nuevoNombreBase + ext);

    
        //Si otro archivo ya existe con el mismo nombre, Agregamos un contador al inicio ej. CDMX_Limon_Lateral_Estudiante.MP4 -> 1_CDMX_Limon_Lateral_Estudiante.MP4
        if (fs::exists(candidato) || nombresReservadosEnEsteLote.count(candidato.string())) {
        int contador = 1;
        std::string nombreConContador;
        do {
            nombreConContador = std::to_string(contador) + "_" + nuevoNombreBase;
            candidato = entrada.path().parent_path() / (nombreConContador + ext);
            contador++;
            } while (fs::exists(candidato) || nombresReservadosEnEsteLote.count(candidato.string()));

        nuevoNombreBase = nombreConContador;
        }

        nombresReservadosEnEsteLote.insert(candidato.string());
        fs::path nuevaRuta = entrada.path().parent_path() / (nuevoNombreBase + ext);
        pendientes.push_back({entrada.path(), nuevaRuta});
        Logger::instance().info("Delabeler",
            "[DETECTADO] " + entrada.path().filename().string() + " -> " + nuevaRuta.filename().string());
    }

    if (pendientes.empty()) {
        Logger::instance().info("Delabeler", "No se encontraron archivos con calidad detectada en el nombre.");
        return 0;
    }

    std::cout << "\n" << pendientes.size() << " archivo(s) seran renombrados." << std::endl;
    std::cout << "Continuar? (s/N): ";
    std::string confirmacion;
    std::getline(std::cin, confirmacion);

    if (confirmacion != "s" && confirmacion != "S") {
        Logger::instance().info("Delabeler", "Cancelado por el usuario.");
        return 0;
    }

    int ok = 0, fallidos = 0;
    for (const auto& [rutaVieja, rutaNueva] : pendientes) {
        if (fs::exists(rutaNueva)) {
            Logger::instance().warn("Delabeler",
                "Ya existe " + rutaNueva.filename().string() + " — se omite " + rutaVieja.filename().string());
            fallidos++;
            continue;
        }
        try {
            fs::rename(rutaVieja, rutaNueva);
            Logger::instance().info("Delabeler",
                "[OK] " + rutaVieja.filename().string() + " -> " + rutaNueva.filename().string());
            ok++;
        } catch (const std::exception& e) {
            Logger::instance().error("Delabeler", rutaVieja.filename().string() + ": " + e.what());
            fallidos++;
        }
    }

    Logger::instance().info("Delabeler",
        "Completado: " + std::to_string(ok) + " renombrados, " + std::to_string(fallidos) + " fallidos.");
    return (fallidos > 0) ? 1 : 0;
}

// Compilacion (ejemplo):
// g++ -std=c++17 -I../scripts_genericos/include \
//   ../scripts_genericos/src/Logger.cpp ../scripts_genericos/src/PathManager.cpp \
//   ../scripts_genericos/src/VideoMetadata.cpp \
//   Delabeler.cpp -o delabeler
//
// Uso:
// ./delabeler [directorio opcional]
