import tensorflow as tf

TIME_STEPS = 30
FEATURES = 132

print("1. Cargando el modelo original...")
modelo_original = tf.keras.models.load_model('modelo_caidas.keras')

print("2. Creando un modelo clon con tamaño de lote (batch) estrictamente fijo en 1...")
nuevo_modelo = tf.keras.Sequential([
    # En Keras 3, declaramos la forma de entrada con una capa Input dedicada
    tf.keras.layers.Input(batch_shape=(1, TIME_STEPS, FEATURES)),
    
    # Las capas LSTM ya no necesitan recibir el shape directamente
    tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.LSTM(32, return_sequences=False, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("3. Transfiriendo el conocimiento (pesos) al nuevo modelo...")
nuevo_modelo.set_weights(modelo_original.get_weights())

print("4. Convirtiendo a TFLite nativo...")
converter = tf.lite.TFLiteConverter.from_keras_model(nuevo_modelo)
tflite_model = converter.convert()

nuevo_nombre = 'modelo_caidas_nativo.tflite'
with open(nuevo_nombre, 'wb') as f:
    f.write(tflite_model)

print(f"¡Éxito! Modelo guardado como {nuevo_nombre}.")