import cv2
import mediapipe as mp
import threading
import numpy as np
import time
#import requests # Para enviar alertas web (HTTP)
from collections import deque
#import tensorflow as tf # O 'import tflite_runtime.interpreter as tflite' en Raspberry Pi
import ai_edge_litert.interpreter as tflite

#pip uninstall tensorflow protobuf -y
#pip install protobuf==4.25.3

#configuración
feed = './videos/50wtf.mp4'
MODEL_PATH = 'modelo_nativo.tflite'
API_ALERTA_URL = 'http://tu-servidor-central.com/api/alertas'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class CameraHandler:
    def __init__(self, camera_index, window_name, camera_id):
        self.cap = cv2.VideoCapture(camera_index)
        self.window_name = window_name
        self.camera_id = camera_id
        
        #configurar mediapipe
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
        self.running = True
        
        #búfer de secuencia (30 frames)
        self.sequence_buffer = deque(maxlen=30)
        
        #control de alertas para no saturar el servidor enviando alertas cada frame
        self.ultima_alerta_tiempo = 0
        self.cooldown_alerta_segundos = 10 
        
        #cargar el intérprete de TFLite
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.diccionario_clases = {0: 'Normal', 1: 'Caida', 2: 'Sentado', 3: 'Caminando'}

        #filtro de estabilización
        self.estado_confirmado = 'Normal'
        self.frames_consecutivos = 0
        self.clase_candidata = None
        self.FRAMES_REQUERIDOS = 10  # requiere ~1/3 de segundo de predicción continua a 30fps

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def extract_keypoints(self, results):

        #aplanar los 33 puntos x 4 valores = 132 características (Igual que en el entrenamiento)
        if results.pose_landmarks:
            return np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        else:
            return np.zeros(132)

    def enviar_alerta(self, confianza):

        #Envía una alerta HTTP asíncrona al servidor central
        tiempo_actual = time.time()
        
        #Solo enviar alerta si ha pasado el tiempo de cooldown
        if (tiempo_actual - self.ultima_alerta_tiempo) > self.cooldown_alerta_segundos:
            self.ultima_alerta_tiempo = tiempo_actual
            
            payload = {
                "camara_id": self.camera_id,
                "evento": "CAIDA_DETECTADA",
                "confianza": float(confianza),
                "timestamp": tiempo_actual
            }
            
            print(f"\n[!] ENVIANDO ALERTA AL SERVIDOR: {payload}")
            
            # Ejecutar la petición en un hilo separado para no congelar el video
            def post_request():
                try:
                    # requests.post(API_ALERTA_URL, json=payload, timeout=3)
                    print(f"[{self.camera_id}] Alerta enviada con éxito.")
                except Exception as e:
                    print(f"Error enviando alerta: {e}")
            
            threading.Thread(target=post_request, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: 
                # Si es un video grabado y se acaba, reiniciar (loop)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 1. Extraer puntos clave y actualizar búfer
                keypoints = self.extract_keypoints(results)
                self.sequence_buffer.append(keypoints)
                
                # 2. Si tenemos una secuencia completa de 30 frames, hacemos la predicción
                if len(self.sequence_buffer) == 30:
                    # Preparar los datos de entrada para TFLite (Shape: 1, 30, 132)
                    input_data = np.expand_dims(self.sequence_buffer, axis=0).astype(np.float32)
                    
                    # Ejecutar inferencia
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()

                    # 2. Obtener el arreglo de 4 probabilidades
                    predicciones = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                    # 3. Encontrar el índice con la probabilidad más alta (0, 1, 2 o 3)
                    indice_clase = np.argmax(predicciones)
                    confianza = predicciones[indice_clase]
                    #estado_actual = self.diccionario_clases[indice_clase]
                    estado_predicho = self.diccionario_clases[indice_clase]

                    # 4. Lógica de visualización y alertas
                    if confianza > 0.85:
                        #Si el modelo predice lo mismo que en el frame anterior
                        if estado_predicho == self.clase_candidata:
                            self.frames_consecutivos += 1
                        else:
                            # Si cambia la predicción, reiniciamos el contador
                            self.clase_candidata = estado_predicho
                            self.frames_consecutivos = 1
                            
                        #Si la predicción se mantiene estable por N frames, confirmamos el estado
                        if self.frames_consecutivos >= self.FRAMES_REQUERIDOS:
                            if self.estado_confirmado != estado_predicho:
                                self.estado_confirmado = estado_predicho
                                print(f"NUEVO ESTADO CONFIRMADO: {self.estado_confirmado}")
                                
                                # SOLO AQUI disparas la alerta MQTT si es una caída
                                if self.estado_confirmado == 'Caida':
                                    self.enviar_alerta(confianza)
                    else:
                        #Si la confianza es baja, reiniciamos el contador por seguridad
                        self.frames_consecutivos = 0

                    #ibujar en pantalla usando el estado CONFIRMADO, no el predicho instantáneamente
                    if self.estado_confirmado == 'Caida':
                        cv2.putText(frame, f"ALERTA: {self.estado_confirmado}!", (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        cv2.putText(frame, f"Estado: {self.estado_confirmado}", (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cap.release()
        cv2.destroyWindow(self.window_name)

# Iniciar el sistema
print("Iniciando sistema de monitoreo inteligente...")
# Pasamos un ID único a la cámara para identificarla en el backend
cam1 = CameraHandler(feed, "Monitoreo Principal", camera_id="CAM_HABITACION_01")
cam1.start()

# Para mantener el programa principal corriendo mientras los hilos trabajan
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Apagando sistema...")
    cam1.running = False