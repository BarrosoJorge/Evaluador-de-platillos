

#archivo que genera un archivo md con todas las rutas de archivos de la carpeta de trabajo
#y el nombre de los archivos presentes en cada carpeta
import os

def generar_md_rutas(ruta_padre):
    with open("rutas_archivos.md", "w") as archivo_md:
        for ruta, directorios, archivos in os.walk(ruta_padre):
            archivo_md.write(f"## Ruta: {ruta}\n\n")
            if archivos:
                archivo_md.write("### Archivos:\n")
                for archivo in archivos:
                    archivo_md.write(f"- {archivo}\n")
            else:
                archivo_md.write("No hay archivos en esta carpeta.\n")
            archivo_md.write("\n")
if __name__ == "__main__":
    ruta_padre = os.getcwd()  # Obtiene la ruta de la carpeta de trabajo actual
    generar_md_rutas(ruta_padre)