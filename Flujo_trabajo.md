# 🍽️ EVALUADOR DE PLATILLOS — DASHBOARD DE PROYECTO
### Proyecto de Servicio Social · Visión por Computadora + Sistema Difuso

---

## 🗺️ MAPA VISUAL DEL PROYECTO (Pipeline completo)

```
 🎥 VIDEOS .mp4 (D:\Data\Raw\Videos)
        │
        ▼
 ┌───────────────────────────┐
 │  Labeler.cpp        ✅     │  Preview con OpenCV + renombrado interactivo
 │  (renombrado/etiquetado)   │  Formato: Ciudad_Platillo_Angulo_Autor_Calidad
 └───────────────────────────┘
        │
        ▼
 ┌───────────────────────────┐
 │  VideoToImage.cpp   ✅     │  Extrae N frames distribuidos por video
 │  (video → imágenes)        │  Organiza en Data/Raw/Imagenes/...
 └───────────────────────────┘
        │
        ▼
   🟡 ETAPA 0 · DATOS  (en proceso)
        │
        ▼
   🔴 ETAPA 1 · PREPROCESAMIENTO
        ├─ GrabCut → segmentar platillo / fondo
        ├─ Resize uniforme
        ├─ Corrección de orientación
        ├─ Alineación de imágenes
        └─ 🆕 MLP pixel-a-pixel (clase "plato" vs "comida")
        │
        ▼
   🔴 ETAPA 2 · EXTRACCIÓN DE CARACTERÍSTICAS
        ├─ Ventaneo de la imagen (ej. 3×3, pixel = centro)
        ├─ Energía · Correlación · Contraste · Homogeneidad · IDF · Entropía · Varianza
        ├─ Bases → Raw | Hist. suma-diferencia | Co-ocurrencia (3 dist × 4 áng) | LBP
        └─ Salida → CSVs con doble etiqueta (platillo, chef/estudiante)
        │
        ▼
   🔴 ETAPA 3 · MODELO DE EVALUACIÓN (Sistema Difuso)
        ├─ Genetic Fuzzy System  → evoluciona reglas automáticamente
        └─ Neural Fuzzy System   → aprende parámetros de la red difusa
        │
        ▼
   🎯 CALIFICACIÓN AUTOMÁTICA DEL PLATILLO (Estudiante vs Chef)
```

### 📊 Tablero de estado por etapa

| Etapa | Descripción | Estado | Avance |
|---|---|---|---|
| 0️⃣ Datos | Videos → Etiquetado → Imágenes | 🟡 EN PROCESO | ▓▓▓▓▓▓░░░░ ~60% |
| 1️⃣ Preprocesamiento | Segmentación, resize, normalización | 🔴 PENDIENTE | ░░░░░░░░░░ 0% |
| 2️⃣ Características | Ventaneo + texturas → CSVs | 🔴 PENDIENTE | ░░░░░░░░░░ 0% |
| 3️⃣ Modelo difuso | Genetic/Neural Fuzzy evaluador | 🔴 PENDIENTE | ░░░░░░░░░░ 0% |

---

## 🧰 ¿QUÉ TENGO?

### Herramientas de código (C++/OpenCV)

| Script | Función | Estado |
|---|---|---|
| `VideoToImage.cpp` | Convierte videos en imágenes, extracción distribuida de N frames, organización automática en jerarquía de carpetas, parseo de metadatos del nombre del archivo | ✅ Completo y funcional |
| `Labeler.cpp` | Etiquetado interactivo de videos: preview de primer frame, renombrado con hilos para que la GUI no se congele | ✅ Completo y funcional |

### Estructura de datos definida

```
Data/
 └── Raw/
      ├── Videos/
      └── Imagenes/
           └── Ciudad/
                └── Platillo/
                     └── Angulo/ (Superior | Lateral)
                          └── Autor/ (Chef | Estudiante)
```

**Convención de nombres:**
```
Ciudad_Platillo_Angulo_Autor_Calidad.mp4
   │       │       │      │      └─ B/R/M (solo Estudiante)
   │       │       │      └──────── Chef | Estudiante
   │       │       └─────────────── Superior | Lateral
   │       └─────────────────────── choux, pizza, paste, etc.
   └─────────────────────────────── Querétaro, CDMX, etc.
```
> 📝 Nota: los videos "COCINERO" se renombraron como Estudiante calidad B (Bueno).

### Catálogo de platillos disponibles

| Ciudad | Platillo | Estado |
|---|---|---|
| Querétaro | Choux | ✅ Disponible |
| Querétaro | Pizza | ✅ Disponible |
| Querétaro | Pollo Rostizado | ✅ Disponible |
| Querétaro | Suprema | ✅ Disponible |
| Querétaro | Huevo | ⚠️ Solo Chef |
| Querétaro | Paste | ⚠️ Por verificar |
| Querétaro | Pollo Relleno | ❌ Descartado |
| Querétaro | Tecs | ❌ Descartado |
| CDMX | Omelette | ⚠️ Por verificar (Chef + Estudiante) |

---

## ✅ ¿QUÉ YA HICE?

- 🛠️ **Dos herramientas C++ terminadas y funcionando:** pipeline de adquisición de datos (etiquetado → conversión a imágenes) completamente operativo, sin depender de pasos manuales repetitivos.
- 🧵 **Etiquetado robusto:** `Labeler.cpp` resuelve el problema de UI congelada usando hilos, lo cual demuestra manejo de concurrencia aplicado a una herramienta real.
- 🗂️ **Convención de datos consolidada:** jerarquía de carpetas y nomenclatura de archivos ya definida y aplicada, lo que da una base sólida y escalable para todo el resto del proyecto.
- 🍳 **Catálogo de platillos mapeado:** ya sabes qué platillos están listos, cuáles faltan verificar y cuáles se descartaron — evitas trabajar sobre datos incompletos.
- 🧠 **Diseño conceptual de las etapas 1-3:** ya tienes decidido el enfoque técnico (segmentación, ventaneo, texturas, sistema difuso híbrido) aunque falte implementarlo. Esto no es trivial: ya resolviste el "qué hacer", ahora falta el "cómo construirlo".

---

## 🛣️ ROADMAP / PASOS SIGUIENTES

| # | Paso | Depende de | Complejidad |
|---|---|---|---|
| 1 | Verificar Paste (Querétaro) y Omelette (CDMX) — ¿tienen Chef y Estudiante? | — | 🟢 Fácil |
| 2 | Etiquetar videos restantes con `Labeler.cpp` | Paso 1 | 🟢 Fácil |
| 3 | Correr `VideoToImage.cpp` sobre todo el set ya etiquetado | Paso 2 | 🟢 Fácil |
| 4 | Re-probar GrabCut, resize, corrección de orientación y alineación sobre las imágenes nuevas | Paso 3 | 🟡 Medio |
| 5 | Decidir si se intenta el MLP de segmentación pixel-a-pixel (plato vs comida) o se sigue solo con GrabCut | Paso 4 | 🔴 Difícil |
| 6 | Implementar ventaneo de imagen (empezar con 3×3) | Paso 4/5 | 🟡 Medio |
| 7 | Implementar extracción de características por ventana (energía, contraste, homogeneidad, IDF, entropía, varianza) sobre las 4 bases (Raw, histogramas, co-ocurrencia, LBP) | Paso 6 | 🔴 Difícil |
| 8 | Generar CSVs etiquetados (platillo + chef/estudiante) | Paso 7 | 🟡 Medio |
| 9 | Decidir arquitectura del sistema difuso (Genetic, Neural, o ambos) | Paso 8 | 🟡 Medio |
| 10 | Implementar y entrenar el sistema difuso evaluador | Paso 9 | 🔴 Difícil |
| 11 | Validar resultados contra evaluaciones humanas (Chef vs Estudiante B/R/M) | Paso 10 | 🟡 Medio |

> 💡 Los pasos 1-3 son "limpieza de pendientes" de la Etapa 0 y se pueden hacer en paralelo mientras decides los puntos abiertos de Etapa 1.

---

## ❓ DECISIONES PENDIENTES

| Pregunta | Por qué importa |
|---|---|
| ¿Es viable el MLP de segmentación pixel-a-pixel con pocas imágenes? | Si no hay suficientes datos etiquetados a nivel pixel, el modelo puede no generalizar — quizás convenga quedarse con GrabCut por ahora |
| ¿Qué tamaños de ventana probar además de 3×3? | Ventanas más grandes capturan más contexto pero aumentan el costo computacional y pueden diluir detalles finos |
| ¿Padding con ceros o con promedio? | Afecta directamente los valores de textura en los bordes del platillo, que suelen ser zonas de evaluación visual importantes |
| ¿Genetic Fuzzy, Neural Fuzzy, o ambos para comparar? | Define la arquitectura final del modelo de evaluación — comparar ambos da más rigor pero duplica el trabajo de implementación |
| ¿Omelette (CDMX) tiene tanto Chef como Estudiante? | Si no, ese platillo no sirve para entrenar/evaluar comparativamente y habría que descartarlo como los otros |

---

## 🪪 TARJETA DE ESTADO DEL PROYECTO

```
┌─────────────────────────────────────────────────────────┐
│ 🍽️  EVALUADOR DE PLATILLOS — Servicio Social             │
│ 🟡  Etapa actual: 0 (Datos) — Etapas 1-3 sin iniciar      │
│ ✅  2 herramientas C++ funcionales (Labeler + ToImage)    │
│ 🔴  Pendiente crítico: extracción de características      │
│ ❓  5 decisiones técnicas abiertas por resolver            │
└─────────────────────────────────────────────────────────┘
```
