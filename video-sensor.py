import cv2
import mediapipe as mp
import threading
import numpy as np
import time
import requests 
from collections import deque
#import paho.mqtt.client as mqtt # <--- NUEVA LIBRERÍA

# --- IMPORTANTE: Usamos la librería estándar de TensorFlow Lite ---
import tensorflow.lite as tflite 

# --- CONFIGURACIÓN DEL SISTEMA ---
MODEL_PATH = 'modelo_caidas_nativo.tflite'
#API_ALERTA_URL = 'http://tu-servidor-central.com/api/alertas' 

# --- CONFIGURACIÓN MQTT (COMUNICACIÓN CON ESP32) ---
#MQTT_BROKER = "192.168.1.66"  # O la IP de tu compu (ej. 192.168.1.50)
#MQTT_TOPIC = "casa/habitacion1/presencia"
#SISTEMA_ACTIVO = False     # Empieza apagado hasta que el sensor diga lo contrario

# Función que se ejecuta cuando llega un mensaje del ESP32
"""def on_message(client, userdata, msg):
    global SISTEMA_ACTIVO
    mensaje = msg.payload.decode()
    print(f"[MQTT] Mensaje recibido: {mensaje}")
    
    if mensaje == "1":
        SISTEMA_ACTIVO = True
    else:
        SISTEMA_ACTIVO = False"""

# Configurar Cliente MQTT
#client = mqtt.Client()
#client.on_message = on_message
#try:
"""    client.connect(MQTT_BROKER, 1883, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_start() # Escuchar en segundo plano
    print(f"Conectado al Broker MQTT en {MQTT_BROKER}")
except:
    print("ADVERTENCIA: No se encontró servidor MQTT. El sistema iniciará en modo MANUAL (Siempre activo).")
    SISTEMA_ACTIVO = True
"""
SISTEMA_ACTIVO = True
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class CameraHandler:
    def __init__(self, camera_index, window_name, camera_id):
        self.cap = cv2.VideoCapture(camera_index)
        self.window_name = window_name
        self.camera_id = camera_id
        
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
        self.running = True
        self.sequence_buffer = deque(maxlen=30)
        
        # IA
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Alertas
        self.ultima_alerta_tiempo = 0
        self.cooldown_alerta_segundos = 10 

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def extract_keypoints(self, results):
        if results.pose_landmarks:
            return np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        else:
            return np.zeros(132)

    def enviar_alerta(self, confianza):
        # (Misma lógica de alerta que tenías antes...)
        pass # Resumido para ahorrar espacio, deja tu código original aquí

    def update(self):
        global SISTEMA_ACTIVO
        while self.running:
            # 1. LEER CÁMARA SIEMPRE (Para vaciar el buffer)
            ret, frame = self.cap.read()
            if not ret: 
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # --- INTERRUPTOR INTELIGENTE ---
            # Si el ESP32 dice que no hay nadie, NO procesamos IA (Ahorro CPU/GPU)
            if not SISTEMA_ACTIVO:
                # Mostrar pantalla de espera (Ahorro de recursos)
                black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_screen, "MODO ESPERA...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                cv2.putText(black_screen, "Sensor: Sin Presencia", (210, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.imshow(self.window_name, black_screen)
                
                # Pausa pequeña para no saturar el ciclo while
                time.sleep(0.1)
                
                # Limpiamos el buffer de memoria para que no tenga datos viejos al despertar
                self.sequence_buffer.clear()
                
                if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False
                continue # Saltamos todo el código de abajo y volvemos al inicio del while

            # --- SI HAY PRESENCIA, EJECUTAMOS TODO EL CÓDIGO PESADO ---
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                keypoints = self.extract_keypoints(results)
                self.sequence_buffer.append(keypoints)
                
                if len(self.sequence_buffer) == 30:
                    input_data = np.expand_dims(self.sequence_buffer, axis=0).astype(np.float32)
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()
                    prediccion = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                    
                    if prediccion > 0.85: 
                        cv2.putText(frame, f"ALERTA IA: CAIDA! ({prediccion*100:.1f}%)", (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        # self.enviar_alerta(prediccion) # Descomenta cuando uses alertas
                    else:
                        cv2.putText(frame, f"Estado: Normal", (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cap.release()
        cv2.destroyWindow(self.window_name)

if __name__ == "__main__":
    print("Iniciando sistema IoT...")
    cam1 = CameraHandler(1, "Camara 1", "CAM_01")
    cam1.start()
    
    cam2 = CameraHandler(2, "Camara 2", "CAM_02") # Descomenta para 2 cámaras
    cam2.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        cam1.running = False