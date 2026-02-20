import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# 1. Configuración de parámetros
CSV_PATH = 'dataset_caidas.csv'
TIME_STEPS = 30  # Cuántos frames consecutivos forman una secuencia (1 segundo a 30fps)
FEATURES = 132   # 33 landmarks * 4 valores (x, y, z, visibilidad)

def cargar_y_crear_secuencias(csv_path, time_steps):
    print("Cargando datos del CSV...")
    df = pd.read_csv(csv_path)
    
    secuencias = []
    etiquetas = []
    
    # Agrupar por video_id para no mezclar frames de diferentes videos
    videos = df.groupby('video_id')
    
    print("Creando ventanas de tiempo...")
    for video_id, data in videos:
        # Extraer solo las columnas de coordenadas (desde 'x0' hasta 'v32')
        valores_coordenadas = data.iloc[:, 3:].values
        etiqueta_video = data.iloc[0]['label'] # La etiqueta es la misma para todo el video
        
        # Crear secuencias deslizantes (Sliding Window)
        num_frames = len(valores_coordenadas)
        if num_frames >= time_steps:
            for i in range(num_frames - time_steps + 1):
                ventana = valores_coordenadas[i : i + time_steps]
                secuencias.append(ventana)
                etiquetas.append(etiqueta_video)
                
    return np.array(secuencias), np.array(etiquetas)

# 2. Preparar los datos
X, y = cargar_y_crear_secuencias(CSV_PATH, TIME_STEPS)

print(f"Total de secuencias generadas: {X.shape[0]}")
print(f"Forma de X (Entrada): {X.shape} -> (muestras, time_steps, features)")
print(f"Forma de y (Etiquetas): {y.shape}")

# Dividir en datos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Construir la arquitectura del modelo LSTM
print("\nConstruyendo el modelo...")
model = Sequential([
    # Primera capa LSTM que lee la secuencia
    LSTM(64, return_sequences=True, activation='relu', input_shape=(TIME_STEPS, FEATURES)),
    Dropout(0.2), # Previene el sobreajuste (overfitting)
    
    # Segunda capa LSTM que consolida la información
    LSTM(32, return_sequences=False, activation='relu'),
    Dropout(0.2),
    
    # Capas densas de clasificación
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Salida binaria: 0 (Normal) o 1 (Caída)
])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 4. Entrenar el modelo
print("\nIniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    epochs=50,             # Número de pasadas completas por los datos
    batch_size=32,         # Cuántas secuencias procesar a la vez
    validation_data=(X_test, y_test)
)

# 5. Evaluar y guardar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en datos de prueba: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
model_path = 'modelo_caidas.keras'
model.save(model_path)
print(f"Modelo guardado exitosamente como '{model_path}'")