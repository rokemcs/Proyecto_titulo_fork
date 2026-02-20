import tensorflow as tf

modelo_path = 'modelo_caidas.keras'
model = tf.keras.models.load_model(modelo_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Añadir las configuraciones que pide el error:
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Operaciones estándar de TFLite
    tf.lite.OpsSet.SELECT_TF_OPS    # Permitir operaciones completas de TF
]
converter._experimental_lower_tensor_list_ops = False

print("Convirtiendo a TensorFlow Lite con Select TF Ops...")
tflite_model = converter.convert()

with open('modelo_caidas.tflite', 'wb') as f:
    f.write(tflite_model)

print("¡Conversión exitosa!")