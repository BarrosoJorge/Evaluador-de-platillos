# Pipeline de evaluación automatizada de platillos (visual)

## Contexto y supuestos confirmados

- El sistema **no es open-set**: un chef prepara un platillo de referencia y todos los alumnos de una sesión replican ese mismo platillo. El chef fija la plantilla de regiones **una vez por sesión**, no por platillo genérico del mundo.
- La foto del chef es el "true" (calificación implícita de 10). Las fotos de alumnos no están etiquetadas ni globalmente ni por categoría.
- No hay calificaciones humanas por dimensión (color, textura, etc.) disponibles para entrenamiento supervisado. El aprendizaje debe derivarse de la comparación estadística contra la población de alumnos de la sesión (aprox. 20+ por sesión), no de labels.
- Rúbrica final objetivo (6 categorías, escala 1-10 cada una):
  1. Color y contraste
  2. Equilibrio y simetría
  3. Altura y volumen
  4. Texturas y formas
  5. Limpieza y vajilla
  6. Proporción y enfoque


## Decisiones de diseño clave

- **Whole-image pixel-a-pixel se descarta** como método principal: no hay garantía de alineación entre foto de chef y de alumno (ángulo, distancia, iluminación), y no da interpretabilidad por componente.
- **Evaluación por áreas/elementos** es el enfoque correcto, coherente con el pipeline existente de segmentación (GrabCut) y extracción de features por ventaneo.
- Las propiedades a evaluar se dividen en dos naturalezas distintas y **no deben tratarse con el mismo mecanismo**:
  - **Locales (por región)**: color, textura/forma.
  - **Globales (sobre imagen completa o relación entre regiones)**: equilibrio/simetría, limpieza/vajilla, proporción, enfoque.
  - **No medible de forma confiable con una sola foto RGB 2D**: altura/volumen (requiere profundidad; solo estimable con proxies débiles como sombras u oclusión).
- No se fuzzifica el valor crudo de una feature, sino el **delta relativo entre alumno y chef**.
- La ponderación entre features **no se define a mano**: se deriva de la estructura estadística de la población de alumnos de la sesión (matriz de covarianza), evitando decisiones arbitrarias de "cuánto pesa cada feature".
- Con ~20 alumnos por sesión y features de alta dimensionalidad (GLCM + LBP + histogramas de color), hay riesgo real de *curse of dimensionality* al estimar la covarianza. Mitigación: (1) reducir cada bloque de features a estadísticos resumen (media, varianza, entropía) antes de comparar, y (2) usar estimación de covarianza con shrinkage (Ledoit-Wolf) en vez de covarianza empírica cruda.

## Pipeline completo

### 0. Creación de dataset
- **Entrada**: videos de alumnos y chef, sin formato uniforme.
- **Proceso**: renombrado de videos a formato uniforme (convención `Ciudad_Platillo_Angulo_Autor_Calidad`); extracción de N imágenes por video.
- **Salida**: imágenes con nombre uniforme, identificando explícitamente cuál pertenece al chef (referencia) y a qué sesión/platillo pertenece cada imagen de alumno. Sin ponderación 1-10.

### 1. Preprocesamiento
- **Entrada**: imágenes crudas.
- **Proceso**: mejoras de imagen únicamente (corrección de balance de blancos, reducción de ruido, normalización de tamaño/resolución, corrección de iluminación). **No incluye segmentación** — eso es exclusivo de la etapa 2.
- **Salida**: imágenes mejoradas.

### 2. Aislamiento y Subdivisión Espacial (Antes Segmentación)
    Ya no buscaremos ingredientes. Solo separaremos la comida del plato y la dividiremos en zonas medibles.

    Entrada: Imágenes mejoradas (chef + alumnos).

    Proceso:

    Aislamiento Comida-Fondo (GrabCut o Umbralización): Extraer una máscara global que diferencie qué es "materia comestible" y qué es "plato/fondo vacío".

    Subdivisión (Parches o Superpíxeles): Dividir el área de la comida en regiones agnósticas usando un Grid espacial (ej. cuadrícula de 8x8 sobre la imagen) o Superpíxeles. Cada zona recibe un ID (Zona 1, Zona 2... Zona N), de modo que la Zona 1 del chef se comparará espacialmente con la Zona 1 del alumno.

    Salida:

    Máscara binaria global (1 = comida, 0 = fondo/plato).

Mapa de subdivisiones (patch_id).

### 3. Vectores de Features Espaciales y Globales
    Calcularemos las propiedades visuales de cada zona generada en el paso anterior y propiedades generales del emplatado.

    Entrada: Imágenes transformadas a espacio de color HSV/Lab + Máscara global + Mapa de subdivisiones de la Etapa 2.

    Proceso (dos ramas en paralelo):

    Por Parche/Zona (Espacial): Iterar sobre cada patch_id. Se calculan las features solo sobre los píxeles válidos (comida) dentro de esa zona:

    Color: Media y desviación estándar de los canales H y S (o L, a, b).

    Textura: GLCM (energía, contraste, homogeneidad) o LBP de esa zona específica.

    Global (Estructura del platillo):

    Simetría/Equilibrio: Centro de masa de la máscara global de comida (si está muy desviado del centro del plato, el platillo está desequilibrado).

    Limpieza: Conteo de pequeñas "islas" de píxeles fuera de la masa principal de comida (gotas de salsa mal puestas o manchas).

    Volumen/Proporción: Área total de píxeles de comida respecto al tamaño total del plato.

    Salida: Vector de features final por imagen. Contendrá un bloque de características globales y un bloque secuencial de características por zona (Ej: [Global_Features, Zona1_Features, Zona2_Features...]).

### 4. Comparación estadística
- **Entrada**: matriz de vectores de todos los alumnos de la sesión + vector del chef.
- **Proceso**: estimar covarianza con Ledoit-Wolf sobre la población de alumnos; calcular distancia de Mahalanobis de cada alumno contra el chef (por región y/o global).
- **Salida**: score de distancia por dimensión evaluada, por alumno.

### 5. Fuzzificación
- **Entrada**: distancias de Mahalanobis de la etapa 4.
- **Proceso**: calibrar funciones de membresía usando los percentiles de la distribución de distancias de esa sesión (sin necesidad de labels humanos).
- **Salida**: grados de pertenencia a términos lingüísticos ("similar", "algo distinto", "muy distinto") por dimensión.

### 6. Sistema difuso
- **Entrada**: grados de pertenencia de la etapa 5.
- **Proceso**: aplicar reglas difusas y defuzzificar. Reglas ajustables en el futuro vía ANFIS o algoritmo genético si se llegan a tener calificaciones reales del chef por dimensión.
- **Salida**: score 1-10 por **dimensión elemental** (ej. `textura_carne`, `color_salsa`, `simetria_global`) — la salida más fina que existe en el pipeline.

### 7. Rúbrica final (agregación)
- **Entrada**: scores elementales de la etapa 6.
- **Proceso**: agregación de los scores elementales que pertenecen a cada una de las 6 categorías de la rúbrica (ej. "Color y contraste" = combinación de `color_carne` + `color_salsa` + `color_puré`). Pendiente de definir: promedio simple, ponderado, o una capa adicional de reglas.
- **Salida**: reporte final con las 6 categorías de la rúbrica, cada una en escala 1-10.

## Diferencia clave etapa 6 vs. etapa 7

- Etapa 6 responde: *¿qué tan distinta es cada pieza individual (región + feature, o feature global)?*
- Etapa 7 responde: *¿qué tan distinto es el platillo en cada una de las 6 dimensiones que le importan al alumno?*
- La etapa 7 es una capa de **agregación** sobre la etapa 6, no una repetición: varias categorías de la rúbrica final combinan más de una salida elemental de la etapa 6 (ej. color y textura combinan las 3 regiones; equilibrio y limpieza son 1:1 porque ya son dimensiones globales).

## Preguntas abiertas / pendientes de definir

- Método de agregación de la etapa 7 (promedio simple vs. ponderado vs. reglas adicionales).
- Si Mahalanobis (etapa 4) se calcula por región por separado o concatenando todas las regiones en un vector único por alumno — implica trade-off entre interpretabilidad por categoría y estabilidad estadística con ~20 muestras.
- Cómo estimar "altura y volumen" dado que no es medible directamente con una sola foto RGB (definir si se usa proxy o se excluye/simplifica esa categoría).
- Implementación concreta de la etapa 2 (segmentación con plantilla fija del chef) y de la etapa 0/1 (renombrado y extracción), que son las más avanzadas actualmente según el dashboard del proyecto (`evaluador_platillos_dashboard.md`).



1. Carpetas

    Data
        Videos
            Crudos
        Imagenes
            Crudas
        Preprocesadas
        Segmentadas
            Mascaras
            Confianza (revisar)
        Features
            Raw
                HOG
                GLCM
                LBP
                Color
                SDH
                ...
            Estadisticos
                PorRegion
                Global
        Comparacion
            Covarianzas
            Distancias
        Fuzzy
            Percentiles
            Pertenencias
        Rubrica
            Reportes
        Sesiones (revisar)
            manifest_sesion.json

**Scripts genericos**
    Logger
    ConfigLoader
    PathManager
    ImageIO
    ...

* 0. Creacion de dataset
    Labeler
    Delabeler
    VideoToImage
    RenombradorUniforme
    IndexadorSesion (revisar)

1. Preprocesamiento
    AislamientoFondo (grabcut, plato vs. fondo — no regiones semanticas)
    CorreccionIluminacion (revisar)
    Redimensionado
    CorreccionOrientacion
    AlineacionImagenes

2. Segmentacion/correspondencia
    Segmentador (aplica plantilla del chef)
    Localizador (busca regiones en imagen de alumno)
    Correspondencia (revisar)
    ValidadorConfianza (revisar)

3. Vectorizacion
    ** Carpeta ** (caracteristicas)
        HOG
        GLCM
        LBP
        Color
        SDH
        ...
    **** Carpeta ** (Estadisticos)
        1. Energía
        2. Contraste
        3. Correlación
        4. Homogeneidad
        5. IDF
        6. Entropía
        7. Varianza
        ...

    ReductorEstadisticos
    VectorizadorPorRegion
    VectorizadorGlobal
    Ensamblador

4. Comparacion
    EstimadorLedoitWolf
    CalculadorMahalanobis
    (revisar si agregar otro)

5. Fuzzyfication
    CalculadorPercentiles
    Fuzzificador

6. Sistema difuso
    MotorReglas
    Defuzzificador
    AjustadorReglas (revisar — ANFIS/genetico, futuro)

7. Rubrica
    Agregador
    GeneradorReporte