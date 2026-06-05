#!/bin/bash

# Script de compilación para VideoLabeler
# Soporta: Linux, macOS, Windows (con WSL2)

set -e  # Salir en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Variables
BUILD_TYPE=${1:-Release}  # Release por defecto, Debug si se especifica
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}VideoLabeler - Sistema de Compilación${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# Detectar sistema operativo
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == "Linux" ]]; then
    echo -e "${YELLOW}Sistema detectado: Linux${NC}"
    NUM_CORES=$(nproc)
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    echo -e "${YELLOW}Sistema detectado: macOS${NC}"
    NUM_CORES=$(sysctl -n hw.ncpu)
elif [[ "$OS_TYPE" =~ "MINGW" ]] || [[ "$OS_TYPE" =~ "MSYS" ]]; then
    echo -e "${YELLOW}Sistema detectado: Windows (MSYS/MinGW)${NC}"
    NUM_CORES=$NUMBER_OF_PROCESSORS
else
    echo -e "${RED}Sistema operativo no soportado${NC}"
    exit 1
fi

echo "Núcleos de CPU disponibles: $NUM_CORES"
echo "Tipo de compilación: $BUILD_TYPE"
echo ""

# Verificar dependencias
echo -e "${YELLOW}Verificando dependencias...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake no está instalado${NC}"
    exit 1
fi

if ! command -v make &> /dev/null && ! command -v ninja &> /dev/null; then
    echo -e "${RED}Error: Make o Ninja no están instalados${NC}"
    exit 1
fi

# Detectar OpenCV
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    echo -e "${RED}Error: OpenCV no está instalado${NC}"
    echo -e "${YELLOW}Intenta instalarlo con:${NC}"
    if [[ "$OS_TYPE" == "Linux" ]]; then
        echo "  sudo apt-get install libopencv-dev"
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        echo "  brew install opencv"
    fi
    exit 1
fi

echo -e "${GREEN}Todas las dependencias encontradas${NC}"
echo ""

# Crear directorio de build
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Limpiando directorio de build anterior...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Ejecutar CMake
echo -e "${YELLOW}Ejecutando CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
       "$SCRIPT_DIR"

# Compilar
echo -e "${YELLOW}Compilando (usando $NUM_CORES núcleos)...${NC}"
if command -v ninja &> /dev/null; then
    ninja -j "$NUM_CORES"
else
    make -j "$NUM_CORES"
fi

# Verificar compilación
if [ -f "$BUILD_DIR/video_labeler" ]; then
    echo ""
    echo -e "${GREEN}✓ Compilación exitosa${NC}"
    echo -e "${GREEN}Ejecutable: ${NC}$BUILD_DIR/video_labeler"
    echo ""
    
    # Información del binario
    echo -e "${YELLOW}Información del ejecutable:${NC}"
    if command -v ldd &> /dev/null; then
        file "$BUILD_DIR/video_labeler"
    fi
else
    echo -e "${RED}✗ Error en compilación${NC}"
    exit 1
fi

# Sugerencias de uso
echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Próximos pasos:${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "Para ejecutar:"
echo "  $BUILD_DIR/video_labeler"
echo ""
echo "Compilar en modo Debug (con símbolos de depuración):"
echo "  $0 Debug"
echo ""
echo "Compilar con Docker:"
echo "  docker build -t video_labeler ."
echo "  docker run -v /ruta/videos:/home/video_user/data video_labeler"
echo ""