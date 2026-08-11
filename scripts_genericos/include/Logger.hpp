#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <filesystem>

namespace evaluador {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

// Logger singleton, thread-safe. Escribe a consola y opcionalmente a archivo.
// Uso:
//   Logger::instance().setLogFile("data/logs/pipeline.log");
//   Logger::instance().info("Segmentacion", "Imagen procesada correctamente");
class Logger {
public:
    static Logger& instance();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void setMinLevel(LogLevel level);
    void setLogFile(const std::filesystem::path& path);

    void debug(const std::string& modulo, const std::string& mensaje);
    void info(const std::string& modulo, const std::string& mensaje);
    void warn(const std::string& modulo, const std::string& mensaje);
    void error(const std::string& modulo, const std::string& mensaje);

private:
    Logger() = default;

    void log(LogLevel level, const std::string& modulo, const std::string& mensaje);
    static std::string levelToString(LogLevel level);
    static std::string timestamp();

    std::mutex mutex_;
    LogLevel minLevel_ = LogLevel::DEBUG;
    std::unique_ptr<std::ofstream> archivo_;
};

} // namespace evaluador
