import pandas as pd
import glob
import os

def compilar_dataset(ruta_caidas, ruta_normales, archivo_salida="dataset_compilado_custom.csv"):
    """
    Lee archivos CSV de dos directorios diferentes, les asigna una etiqueta 
    y los compila en un solo DataFrame.
    
    Etiquetas:
    - 1: Caída (caidas)
    - 0: Normal (normales)
    """
    datos_completos = []
    
    # 1. Procesar archivos de "caídas"
    print(f"Buscando archivos en: {ruta_caidas}")
    archivos_caidas = glob.glob(os.path.join(ruta_caidas, "*.csv"))
    
    for archivo in archivos_caidas:
        try:
            df = pd.read_csv(archivo)
            # Agregar la columna de etiqueta (Label)
            df['label'] = 1
            # Opcional: Agregar el nombre del archivo fuente por si necesitas rastrearlo
            df['source_file'] = os.path.basename(archivo)
            datos_completos.append(df)
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")

    # 2. Procesar archivos "normales"
    print(f"Buscando archivos en: {ruta_normales}")
    archivos_normales = glob.glob(os.path.join(ruta_normales, "*.csv"))
    
    for archivo in archivos_normales:
        try:
            df = pd.read_csv(archivo)
            # Agregar la columna de etiqueta (Label)
            df['label'] = 0
            df['source_file'] = os.path.basename(archivo)
            datos_completos.append(df)
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")

    # 3. Combinar todos los DataFrames
    if datos_completos:
        dataset_final = pd.concat(datos_completos, ignore_index=True)
        
        # Guardar el dataset compilado
        dataset_final.to_csv(archivo_salida, index=False)
        print(f"\n¡Proceso completado! Dataset guardado en: {archivo_salida}")
        print(f"Total de filas: {len(dataset_final)}")
        print(f"Distribución de clases:\n{dataset_final['label'].value_counts()}")
        
        return dataset_final
    else:
        print("No se encontraron archivos CSV o hubo un error al leerlos.")
        return None

# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # Define las rutas donde tienes tus archivos CSV (cambia estas rutas según tu computadora)
    CARPETA_CAIDAS = "D:/Firefox Downloads/archive/Fall/Keypoints_CSV"
    CARPETA_NORMALES = "D:/Firefox Downloads/archive/No_Fall/Keypoints_CSV"
    ARCHIVO_DESTINO = "dataset_entrenamiento_custom.csv"
    
    # Crear carpetas de ejemplo si no existen (solo para que no dé error si lo corres directo)
    os.makedirs(CARPETA_CAIDAS, exist_ok=True)
    os.makedirs(CARPETA_NORMALES, exist_ok=True)
    
    # Llamar a la función
    df_final = compilar_dataset(
        ruta_caidas=CARPETA_CAIDAS, 
        ruta_normales=CARPETA_NORMALES, 
        archivo_salida=ARCHIVO_DESTINO
    )