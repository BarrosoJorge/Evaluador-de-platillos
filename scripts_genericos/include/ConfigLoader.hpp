#pragma once

#include <string>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <sstream>

namespace evaluador {

// Carga archivos de configuracion simples tipo clave=valor, uno por linea.
// Lineas vacias o que empiezan con '#' se ignoran.
//
// Ejemplo de archivo config.ini:
//   ruta_data = ./Data
//   imagenes_por_video = 15
//   umbral_confianza_matching = 0.75
//
// Uso:
//   ConfigLoader config("config.ini");
//   int n = config.get<int>("imagenes_por_video").value_or(10);
class ConfigLoader {
public:
    explicit ConfigLoader(const std::filesystem::path& rutaArchivo);

    bool cargado() const noexcept { return cargadoExitosamente_; }

    template <typename T>
    std::optional<T> get(const std::string& clave) const {
        auto it = valores_.find(clave);
        if (it == valores_.end()) {
            return std::nullopt;
        }
        std::istringstream iss(it->second);
        T resultado;
        if (!(iss >> resultado)) {
            return std::nullopt;
        }
        return resultado;
    }

    std::optional<std::string> getString(const std::string& clave) const;

    const std::unordered_map<std::string, std::string>& valores() const noexcept {
        return valores_;
    }

private:
    void parsear(const std::filesystem::path& rutaArchivo);
    static std::string trim(const std::string& s);

    std::unordered_map<std::string, std::string> valores_;
    bool cargadoExitosamente_ = false;
};

// Especializacion para string: get<std::string>("clave") funciona directo
// sin pasar por istringstream (evita cortar en el primer espacio).
template <>
inline std::optional<std::string> ConfigLoader::get<std::string>(const std::string& clave) const {
    return getString(clave);
}

} // namespace evaluador
