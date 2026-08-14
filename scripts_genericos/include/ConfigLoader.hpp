#pragma once

#include <string>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <sstream>

namespace evaluador {

// Carga configuracion simple en formato clave=valor por linea.
// Ignora lineas vacias y comentarios iniciados con '#'.
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

// Especializacion para string sin parseo por stream.
template <>
inline std::optional<std::string> ConfigLoader::get<std::string>(const std::string& clave) const {
    return getString(clave);
}

} // namespace evaluador
