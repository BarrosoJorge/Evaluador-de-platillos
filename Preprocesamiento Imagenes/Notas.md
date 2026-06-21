# Preprocesamiento de Imágenes — Notas

Referencia metodológica: sección 3.1 y 3.2 del ReporteFinal (Peñaran Prieto, 2024).

---

## Pipeline de la Etapa 1

```
Data/Raw/Imagenes/
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │  GrabCutSegmentador.cpp  (interactivo)       │
 │  · Usuario selecciona ROI del platillo       │
 │  · GrabCut (5 iteraciones, GMM)              │
 │  · Apertura morfológica  (kernel 5×5, ×2)   │
 │  · Cierre morfológico    (kernel 15×15, ×3)  │
 │  · Inpainting Telea sobre huecos internos    │
 │  · Guarda: *_mask.png + *_seg.png            │
 └──────────────────────────────────────────────┘
        │
        ▼
Data/Segmentadas/
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │  Preprocesador.cpp  (automático)             │
 │  · Resize uniforme 224×224 (para VGG16)      │
 │  · Corrección de orientación (minAreaRect)   │
 │    → rotación segura centrada en el objeto   │
 │  · Alineación SSIM:                          │
 │    original vs. 180° contra referencia Chef  │
 │  · Guarda: *_pre.png (224×224)               │
 └──────────────────────────────────────────────┘
        │
        ▼
Data/Preprocesadas/
```

---

## Archivos

| Archivo | Función | Estado |
|---|---|---|
| `GrabCutSegmentador.cpp` | Segmentación interactiva GrabCut + morfología + Telea | ✅ Listo |
| `Preprocesador.cpp` | Resize + orientación + alineación SSIM | ✅ Listo |

---

## Compilación

```bash
# GrabCut Segmentador
g++ -std=c++17 -O2 -o grabcut_seg GrabCutSegmentador.cpp \
    `pkg-config --cflags --libs opencv4`

# Preprocesador
g++ -std=c++17 -O2 -o preprocesador Preprocesador.cpp \
    `pkg-config --cflags --libs opencv4`
```

> Nota: `GrabCutSegmentador.cpp` usa `opencv2/photo.hpp` (inpainting).
> `pkg-config --libs opencv4` ya incluye ese módulo automáticamente.

---

## Uso

```bash
# Paso 1: Segmentar
./grabcut_seg /mnt/d/Data/Raw/Imagenes /mnt/d/Data/Segmentadas

# Paso 2: Preprocess
./preprocesador /mnt/d/Data/Segmentadas /mnt/d/Data/Preprocesadas
```

---

## Convención de nombres de archivos

| Sufijo | Descripción |
|---|---|
| `*_seg.png` | Imagen segmentada (fondo negro, platillo foreground) |
| `*_mask.png` | Máscara binaria: 255=platillo, 0=fondo |
| `*_pre.png` | Imagen preprocesada final (224×224, orientada, alineada) |

---

## Nota sobre el MLP pixel-a-pixel (🆕 en Flujo_trabajo.md)

El flujo de trabajo menciona un MLP para clasificación pixel-a-pixel
("plato" vs "comida"). Este módulo **no está implementado aún** porque:

1. Requiere imágenes etiquetadas **a nivel de píxel** (plato vs. comida),
   que todavía no existen en el dataset.
2. Es un paso experimental — el ReporteFinal no lo describe en detalle.
3. GrabCut ya segmenta el platillo completo (plato + comida) del fondo.
   El MLP distinguiría la comida del borde del plato dentro de la ROI.

**Cuándo implementarlo:** después de generar máscaras de entrenamiento
pixel-a-pixel con una herramienta de anotación (e.g., CVAT, LabelMe).
Entonces se puede entrenar un MLP sencillo con parches 3×3 sobre los
canales BGR o HSV y las etiquetas {0=plato, 1=comida}.
