# Prompt para Claude — Presentación Visual del Proyecto

---

Copia y pega todo lo que está debajo de esta línea en una conversación nueva con Claude.

---

Eres un asistente que genera presentaciones visuales en Markdown usando bloques de texto, tablas, diagramas ASCII y emojis para que sea lo más claro y fácil de leer posible.

Quiero que me hagas una **presentación completa y visual** sobre mi proyecto de Servicio Social. Es un sistema de evaluación automática de platillos de cocina usando visión por computadora y un sistema difuso. Léelo todo con cuidado y preséntamelo como si fuera una guía personal de estado del proyecto: qué hay, qué ya se hizo, qué falta, y cuál es el plan.

---

## CONTEXTO DEL PROYECTO

**Nombre:** Evaluador de Platillos (Proyecto de Servicio Social)
**Objetivo:** Sistema que evalúa automáticamente la calidad visual de un platillo de cocina comparando la presentación de un estudiante contra la de un chef, usando visión por computadora + sistema difuso inteligente.
**Lenguaje principal:** C++ con OpenCV
**Datos:** Videos .mp4 organizados por ciudad, platillo, ángulo (Superior/Lateral) y autor (Chef/Estudiante con calidad B/R/M)

---

## LO QUE YA EXISTE (código ya escrito)

### Script 1 — VideoToImage.cpp (COMPLETO Y FUNCIONAL)
- Recibe una carpeta de videos .mp4
- Los convierte en imágenes extrayendo frames de forma distribuida (N imágenes por video)
- Organiza automáticamente las imágenes en carpetas según la jerarquía:
```
Data/
 └── Raw/
      ├── Videos/
      └── Imagenes/
           └── Ciudad/
                └── Platillo/
                     └── Angulo/ (Superior o Lateral)
                          └── Autor/ (Chef o Estudiante)
```
- Parsea el nombre del archivo para extraer metadatos: Ciudad_Platillo_Angulo_Autor[_Calidad]

### Script 2 — Labeler.cpp (COMPLETO Y FUNCIONAL)
- Herramienta interactiva de etiquetado de videos
- Abre cada video, muestra un preview del primer frame con OpenCV
- Pide al usuario que escriba el nuevo nombre con el formato correcto
- Renombra el archivo automáticamente manteniendo metadatos correctos
- Usa hilos para que la GUI no se congele mientras el usuario escribe

---

## ESTRUCTURA DE DATOS (naming convention de videos)

```
Ciudad_Platillo_Angulo_Autor_Calidad.mp4
   |       |       |      |      |
   |       |       |      |      +-- B (Bueno), R (Regular), M (Malo) — solo para Estudiante
   |       |       |      +--------- Chef | Estudiante
   |       |       +---------------- Superior | Lateral
   |       +------------------------ choux, pizza, paste, etc.
   +-------------------------------- Queretaro, CDMX, etc.
```

**Nota importante:** Los videos etiquetados como "COCINERO" se renombraron como "Estudiante" con calidad B (Bueno).

---

## PLATILLOS DISPONIBLES (por ciudad)

### Querétaro:
- Choux ✅
- Pizza ✅
- Pollo Rostizado ✅
- Suprema ✅
- Huevo (solo Chef)
- Paste (por verificar)
- Pollo Relleno ❌ (descartado)
- Tecs ❌ (descartado)

### CDMX:
- Omelette ✅ (solo CDMX — por verificar si tiene Chef y Estudiante)

---

## FLUJO COMPLETO DEL PROYECTO (4 etapas)

### ETAPA 0 — Datos (EN PROCESO)
1. Videos en D:\Data\Raw\Videos
2. Labeler.cpp → renombrar videos correctamente
3. VideoToImage.cpp → extraer imágenes a Data/Raw/Imagenes/

### ETAPA 1 — Preprocesamiento de Imágenes (PENDIENTE)
Lo que YA se usó antes (probar de nuevo):
- GrabCut → segmentar el platillo del fondo
- Redimensionamiento a tamaño uniforme
- Corrección de orientación
- Alineación de imágenes

Propuesta NUEVA:
- Entrenar un MLP para clasificar cada pixel como "plato" o "comida"
- Razonamiento: plato = fondo, comida = región de interés

### ETAPA 2 — Extracción de Características (PENDIENTE)
**Enfoque: Ventaneo de la imagen**
- En lugar de calcular 1 escalar por imagen completa, se divide en ventanas (ej. 3x3)
- Cada pixel es el centro de su ventana → genera vector de características por pixel
- Se rellena el borde (padding) con ceros o promedio

**Características a extraer (por cada tipo de base):**
- Energía, Correlación, Contraste, Homogeneidad, IDF, Entropía, Varianza

**Bases a usar:**
| Base | Descripción |
|------|-------------|
| Raw | Imagen original con ventaneo |
| Suma y diferencia de histogramas | Histogramas combinados |
| Matrices de co-ocurrencia | 3 distancias (d=1,3,7) x 4 ángulos (0°,45°,90°,135°) |
| Imágenes LBP | Local Binary Pattern, ventana 3x3 |

**Salida:** CSVs con características por pixel, con doble etiqueta (platillo, chef/estudiante)

### ETAPA 3 — Modelo de Evaluación (PENDIENTE)
**Producto final:** Sistema Difuso que asigna calificaciones basadas en características visuales

**Problema con sistema difuso manual:** Crear reglas a mano con características no-humanas es muy complejo.

**Solución propuesta:** Usar un sistema híbrido automático:
- Genetic Fuzzy System → evoluciona las reglas automáticamente
- Neural Fuzzy System → aprende los parámetros de la red difusa

**Lo que se optimiza automáticamente:**
1. Rangos de las funciones de membresía
2. Número de conjuntos difusos
3. Tipo de función de membresía por conjunto
4. Reglas difusas completas

---

## ESTADO ACTUAL RESUMIDO

| Etapa | Descripción | Estado |
|-------|-------------|--------|
| 0. Datos | Videos → Etiquetado → Imágenes | 🟡 EN PROCESO |
| 1. Preprocesamiento | Segmentación, resize, normalización | 🔴 PENDIENTE |
| 2. Características | Ventaneo + HOG/LBP/coocurrencia → CSVs | 🔴 PENDIENTE |
| 3. Modelo difuso | Genetic/Neural Fuzzy evaluador | 🔴 PENDIENTE |

---

## LO QUE ME FALTA DEFINIR / DUDAS ABIERTAS

1. ¿El MLP de segmentación pixel a pixel es viable con pocas imágenes?
2. ¿Qué tamaños de ventana probar (además de 3x3)?
3. ¿Cuál es mejor estrategia de padding: ceros o promedio?
4. ¿Genetic Fuzzy o Neural Fuzzy? ¿O ambos para comparar?
5. ¿Los platillos de CDMX (Omelette) tienen tanto Chef como Estudiante?

---

Ahora, con toda esa información, genera una **presentación visual completa** con:

1. Un **mapa visual del proyecto** (diagrama ASCII o tabla visual) mostrando todas las etapas de inicio a fin
2. Una sección de **"¿Qué tengo?"** con todo lo que ya existe listado claramente
3. Una sección **"¿Qué ya hice?"** con los logros concretos
4. Una sección **"Roadmap / Pasos siguientes"** con los pasos ordenados, qué depende de qué, y estimación de complejidad (fácil/medio/difícil)
5. Una sección de **"Decisiones pendientes"** destacando las preguntas abiertas que necesito resolver
6. Un **resumen de 5 líneas** al final tipo "tarjeta de estado del proyecto"

Hazlo visual, con emojis, tablas, diagramas ASCII y secciones bien delimitadas. Que se vea como un dashboard de estado de proyecto, no como un documento de texto plano.
