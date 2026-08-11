#include "Logger.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>

namespace evaluador {

Logger& Logger::instance() {
    static Logger instancia;
    return instancia;
}

void Logger::setMinLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    minLevel_ = level;
}

void Logger::setLogFile(const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::filesystem::create_directories(path.parent_path());
    archivo_ = std::make_unique<std::ofstream>(path, std::ios::app);
    if (!archivo_->is_open()) {
        std::cerr << "[Logger] No se pudo abrir el archivo de log: " << path << "\n";
        archivo_.reset();
    }
}

void Logger::debug(const std::string& modulo, const std::string& mensaje) {
    log(LogLevel::DEBUG, modulo, mensaje);
}

void Logger::info(const std::string& modulo, const std::string& mensaje) {
    log(LogLevel::INFO, modulo, mensaje);
}

void Logger::warn(const std::string& modulo, const std::string& mensaje) {
    log(LogLevel::WARN, modulo, mensaje);
}

void Logger::error(const std::string& modulo, const std::string& mensaje) {
    log(LogLevel::ERROR, modulo, mensaje);
}

void Logger::log(LogLevel level, const std::string& modulo, const std::string& mensaje) {
    if (level < minLevel_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream linea;
    linea << "[" << timestamp() << "] "
          << "[" << levelToString(level) << "] "
          << "[" << modulo << "] "
          << mensaje;

    if (level >= LogLevel::WARN) {
        std::cerr << linea.str() << "\n";
    } else {
        std::cout << linea.str() << "\n";
    }

    if (archivo_ && archivo_->is_open()) {
        (*archivo_) << linea.str() << "\n";
        archivo_->flush();
    }
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
    }
    return "?????";
}

std::string Logger::timestamp() {
    auto ahora = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(ahora);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

} // namespace evaluador
