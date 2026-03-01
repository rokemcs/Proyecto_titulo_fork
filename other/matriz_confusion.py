import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# pip install matplotlib seaborn scikit-learn

print("\n--- INICIANDO EVALUACIÓN DEL MODELO ---")

# 1. Hacer predicciones usando los datos de prueba (X_test) que la red no ha visto durante el entrenamiento
# El modelo devuelve probabilidades (ej. [0.1, 0.8, 0.05, 0.05])
y_pred_probs = model.predict(X_test)

# Convertir las probabilidades al índice de la clase ganadora (ej. 1 para 'Caida')
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Nombres de tus 4 categorías en el orden exacto de tu diccionario (0, 1, 2, 3)
nombres_clases = ['Normal', 'Caida', 'Sentado', 'Caminando']

# 2. Generar el Reporte de Clasificación (Calcula Precision, Recall y F1-Score)
print("\nReporte de Clasificación:")
reporte = classification_report(y_test, y_pred_classes, target_names=nombres_clases)
print(reporte)

# 3. Calcular la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_classes)

# 4. Dibujar la Matriz de Confusión usando Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nombres_clases, 
            yticklabels=nombres_clases)

plt.title('Matriz de Confusión - Detección de Estados Multiclase', fontsize=14)
plt.ylabel('Estado Real (Lo que realmente pasó)', fontsize=12)
plt.xlabel('Predicción de la IA (Lo que el modelo creyó)', fontsize=12)

# Guardar la imagen para usarla en tu documento de tesis
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada como 'matriz_confusion.png'")

# Mostrar la gráfica en pantalla
plt.show()