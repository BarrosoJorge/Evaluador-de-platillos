#include <iostream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

void ejecutarComando(const std::string& comando) {
    std::cout << ">> " << comando << std::endl;
    std::system(comando.c_str());
}

int main() {
    std::string dirPreprocesadas = "Data/Preprocesadas";
    
    for (const auto& entry : fs::recursive_directory_iterator(dirPreprocesadas)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            std::string imgPre = entry.path().string();
            
            // Solo procesar si es archivo _pre.png
            if (imgPre.find("_pre.png") == std::string::npos) continue;

            std::cout << "\n==================================================" << std::endl;
            std::cout << "Procesando: " << entry.path().filename().string() << std::endl;
            std::cout << "==================================================" << std::endl;

            // 1. Etapa 2: Generar Mascara Global y Grid (Zonas)
            ejecutarComando("./generador_grid \"" + imgPre + "\"");

            // Construir ruta de la carpeta de máscaras recién creada
            std::string rutaMascaraBase = imgPre;
            size_t pos = rutaMascaraBase.find("Preprocesadas");
            if (pos != std::string::npos) rutaMascaraBase.replace(pos, 13, "Mascaras");
            size_t p = rutaMascaraBase.find("_pre.png");
            if (p != std::string::npos) rutaMascaraBase = rutaMascaraBase.substr(0, p);

            std::string mascaraGlobal = rutaMascaraBase + "/mascara_global.png";

            // 2. Etapa 3: Vectorizadores
            if (fs::exists(mascaraGlobal)) {
                ejecutarComando("./vectorizador_global \"" + imgPre + "\" \"" + mascaraGlobal + "\"");
                ejecutarComando("./vectorizador_parches \"" + imgPre + "\" \"" + mascaraGlobal + "\"");
                ejecutarComando("./ensamblador \"" + imgPre + "\"");
            } else {
                std::cerr << "ERROR FATAL: La Etapa 2 no genero mascara_global.png" << std::endl;
            }
        }
    }
    std::cout << "\n¡Pipeline completo!" << std::endl;
    return 0;
}