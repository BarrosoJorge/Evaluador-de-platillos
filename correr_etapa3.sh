#!/bin/bash

# Limpiamos features anteriores para evitar basura
rm -rf Data/Features

# Iteramos sobre todas las imágenes listas
find Data/Preprocesadas/ -type f -name "*_pre.png" | while read -r img_pre; do
    
    # Deducimos dónde está la carpeta Grids de esta imagen específica
    dir_base=$(dirname "$img_pre")
    mascara_global="$dir_base/Grids/mascara_global.png"
    
    if [ -f "$mascara_global" ]; then
        echo "--> Extrayendo features de: $(basename "$img_pre")"
        
        # 1. Extraer features globales (Equilibrio, Limpieza, Enfoque, Volumen)
        ./vectorizador_global "$img_pre" "$mascara_global"
        
        # 2. Extraer features por zonas (Color, GLCM, LBP)
        ./vectorizador_parches "$img_pre" "$mascara_global"
        
        # 3. Juntar ambos en el vector final
        ./ensamblador "$img_pre"
    else
        echo "! ERROR: No se encontró la máscara global en $mascara_global"
    fi
done

echo "¡Extracción de Features completada con éxito!"