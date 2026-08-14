import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética de publicación científica estricta
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False
})

os.makedirs("Data/Graficas", exist_ok=True)
SESION_EJEMPLO = "Queretaro_Choux_Superior"

print("==================================================")
print(f"Generando gráficos de alta calidad para: {SESION_EJEMPLO}")
print("==================================================")

# =============================================================================
# 1. MAPA DE CALOR LIMPIO (16 Zonas x 3 Descriptores)
# =============================================================================
path_distancias = f"Data/Comparacion/Distancias/{SESION_EJEMPLO}.csv"

if os.path.exists(path_distancias):
    df_dist = pd.read_csv(path_distancias, sep=';')
    df_zonas = df_dist[df_dist['dimension'].str.startswith('zona_')].copy()
    
    if not df_zonas.empty:
        # Acortar nombres de alumnos y dimensiones para limpieza visual
        df_zonas['alumno_corto'] = df_zonas['alumno'].str.replace(f"{SESION_EJEMPLO}_", "").str.replace("Estudiante", "Est.")
        
        def clean_dim_name(dim):
            match = re.match(r"zona_(\d+)_(\d+)_(.*)", dim)
            if match:
                return f"Z({match.group(1)},{match.group(2)}) {match.group(3)}"
            return dim
            
        df_zonas['dimension_corta'] = df_zonas['dimension'].apply(clean_dim_name)
        pivot_dist = df_zonas.pivot(index='alumno_corto', columns='dimension_corta', values='distancia')
        
        plt.figure(figsize=(15, 4))
        # cmap 'flare' es muy elegante para publicaciones
        ax = sns.heatmap(pivot_dist, cmap='flare', annot=False, linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Distancia de Mahalanobis ($D_M$)', 'shrink': 0.8})
        
        plt.title(f"Discrepancia Visual Localizada ($D_M$)\nSesión: {SESION_EJEMPLO}", pad=15)
        plt.xlabel("Dimensión Espacial y Descriptor")
        plt.ylabel("Estudiante")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig("Data/Graficas/1_heatmap_descriptores.png", dpi=300, bbox_inches='tight')
        plt.close()

        # =============================================================================
        # 2. BOXPLOT DE DISTRIBUCIÓN DE ERRORES
        # =============================================================================
        plt.figure(figsize=(8, 5))
        
        # Boxplot sin los outliers por defecto (para no duplicar puntos con stripplot)
        sns.boxplot(data=df_zonas, x='alumno_corto', y='distancia', palette='Pastel2', 
                    width=0.5, showfliers=False, boxprops=dict(edgecolor='black', linewidth=1.2))
        
        # Puntos superpuestos para ver la densidad real
        sns.stripplot(data=df_zonas, x='alumno_corto', y='distancia', color='#2c3e50', 
                      alpha=0.6, jitter=0.15, size=5)
        
        mediana_global = df_zonas['distancia'].median()
        plt.axhline(y=mediana_global, color='#c0392b', linestyle='--', linewidth=1.5, 
                    label=f'Mediana Global ($D_M={mediana_global:.2f}$)')
        
        plt.title(f"Variabilidad de Errores por Estudiante\nSesión: {SESION_EJEMPLO}", pad=15)
        plt.xlabel("Estudiante")
        plt.ylabel("Distancia de Mahalanobis ($D_M$)")
        plt.ylim(0, df_zonas['distancia'].max() * 1.1) # Empezar estrictamente en 0
        
        # Leyenda fuera del gráfico
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        
        plt.tight_layout()
        plt.savefig("Data/Graficas/2_boxplot_distancias.png", dpi=300, bbox_inches='tight')
        plt.close()

        # =============================================================================
        # 3. MATRIZ 4x4 DEL PLATO FÍSICO (PROMEDIO ESPACIAL)
        # =============================================================================
        grid_matrix = np.zeros((4, 4))
        counts = np.zeros((4, 4))
        
        for _, row in df_zonas.iterrows():
            match = re.match(r"zona_(\d+)_(\d+)", row['dimension'])
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                grid_matrix[i, j] += row['distancia']
                counts[i, j] += 1
                
        grid_matrix = np.divide(grid_matrix, counts, out=np.zeros_like(grid_matrix), where=counts!=0)
        
        plt.figure(figsize=(6, 5))
        # square=True garantiza que las celdas sean cuadradas como un grid real
        sns.heatmap(grid_matrix, annot=True, fmt=".2f", cmap='Reds', square=True,
                    annot_kws={"size": 12, "weight": "bold"}, linewidths=1, linecolor='white',
                    cbar_kws={'label': 'Error Promedio de la Clase ($D_M$)', 'shrink': 0.8},
                    xticklabels=['Col 0', 'Col 1', 'Col 2', 'Col 3'],
                    yticklabels=['Fila 0', 'Fila 1', 'Fila 2', 'Fila 3'])
        
        plt.title(f"Mapeo de Error Espacial en el Plato\nSesión: {SESION_EJEMPLO}", pad=15)
        plt.xlabel("Posición Horizontal")
        plt.ylabel("Posición Vertical")
        
        plt.tight_layout()
        plt.savefig("Data/Graficas/3_matriz_fisica_plato.png", dpi=300, bbox_inches='tight')
        plt.close()

# =============================================================================
# 4. BARRAS DE RÚBRICA CON LÍNEAS RECTAS DE UMBRAL (1-10)
# =============================================================================
path_rubrica = f"Data/Rubrica/Reportes/{SESION_EJEMPLO}.csv"

if os.path.exists(path_rubrica):
    df_rub = pd.read_csv(path_rubrica, sep=';')
    df_rub['score_num'] = pd.to_numeric(df_rub['score'], errors='coerce')
    df_rub['alumno_corto'] = df_rub['alumno'].str.replace(f"{SESION_EJEMPLO}_", "").str.replace("Estudiante", "Est.")
    
    df_rub_valid = df_rub.dropna(subset=['score_num']).copy()
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_rub_valid, x='categoria', y='score_num', hue='alumno_corto', 
                     palette='crest', edgecolor='black', linewidth=0.8)
    
    # Líneas de referencia rectas
    plt.axhline(y=6.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Aprobatorio (6.0)')
    plt.axhline(y=10.0, color='#27ae60', linestyle='-', linewidth=1.5, label='Referencia Chef (10.0)')
    
    plt.title(f"Calificación Final por Categoría de Rúbrica\nSesión: {SESION_EJEMPLO}", pad=15)
    plt.xlabel("Categoría Analizada")
    plt.ylabel("Puntuación (1-10)")
    plt.ylim(0, 10.5) # Dejar un poco de margen superior visual, pero el tope logico es 10
    plt.xticks(rotation=0)
    
    # Textos sobre las barras
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.1f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=9, color='black', fontweight='bold',
                        xytext=(0, 3), textcoords='offset points')

    # Leyenda limpia afuera
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, title="Estudiantes y Umbrales")
    
    plt.tight_layout()
    plt.savefig("Data/Graficas/4_rubrica_barras.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 5. RADAR CHART MEJORADO
# =============================================================================
if os.path.exists(path_rubrica):
    df_rub = pd.read_csv(path_rubrica, sep=';')
    df_rub['score_num'] = pd.to_numeric(df_rub['score'], errors='coerce').fillna(0)
    
    categorias = df_rub['categoria'].unique().tolist()
    alumnos = df_rub['alumno'].unique().tolist()
    
    N = len(categorias)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    colores = sns.color_palette("Set2", len(alumnos))
    
    for i, alumno in enumerate(alumnos):
        df_al = df_rub[df_rub['alumno'] == alumno]
        values = df_al['score_num'].tolist()
        values += values[:1]
        
        nombre_corto = alumno.replace(f"{SESION_EJEMPLO}_", "").replace("Estudiante", "Est.")
        ax.plot(angles, values, linewidth=2.5, label=nombre_corto, color=colores[i])
        ax.fill(angles, values, alpha=0.15, color=colores[i])
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias, size=10, weight='bold')
    ax.tick_params(pad=20) 
    
    # Quitar la línea del borde circular exterior para un look más limpio
    ax.spines['polar'].set_visible(False)
    
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=9)
    plt.ylim(0, 10)
    
    plt.title(f"Perfil de Desempeño Multidimensional\nSesión: {SESION_EJEMPLO}", pad=30)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig("Data/Graficas/5_rubrica_radar.png", dpi=300, bbox_inches='tight')
    plt.close()

print("==================================================")
print("¡Gráficas de alta calidad exportadas exitosamente!")
print("==================================================")