import cv2
import mediapipe as mp
import threading
import time

# --- CONFIGURACIÓN ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Variables compartidas (Para que los sliders funcionen bien)
umbral_suelo = 30
umbral_sentado = 65
running = True  # Interruptor maestro

class CameraWorker:
    """
    Este trabajador solo se dedica a procesar datos en segundo plano.
    NO toca ninguna ventana para evitar crashes.
    """
    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1)
        self.frame_listo = None  # Aquí guardaremos la imagen procesada para que el main la tome
        self.estado_actual = "INICIANDO"
        self.altura_pct = 0
        self.stopped = False

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        global umbral_suelo, umbral_sentado
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret: continue

            # Redimensionar para rendimiento constante
            frame = cv2.resize(frame, (640, 480))
            
            # --- PROCESAMIENTO IA (MediaPipe) ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            self.estado_actual = "BUSCANDO..."
            
            if results.pose_landmarks:
                # Dibujar esqueleto sobre el frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Matemáticas de postura
                landmarks = results.pose_landmarks.landmark
                y_hombro = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                y_tobillo = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2
                
                # Calcular altura visual (%)
                self.altura_pct = abs(y_tobillo - y_hombro) * 100

                # Decisión usando las variables globales (que controlas con sliders)
                if self.altura_pct < umbral_suelo:
                    self.estado_actual = "SUELO/CAIDA"
                elif self.altura_pct < umbral_sentado:
                    self.estado_actual = "SENTADO"
                else:
                    self.estado_actual = "PARADO"

            # Guardamos el frame procesado para que el Hilo Principal lo muestre
            self.frame_listo = frame
            
        self.cap.release()

    def stop(self):
        self.stopped = True

# Función vacía para los trackbars
def nada(x): pass

# --- BLOQUE PRINCIPAL (GUI) ---
# Todo lo visual ocurre aquí para que no se trabe
if __name__ == "__main__":
    print("Iniciando cámaras...")
    
    # 1. Iniciar los trabajadores en segundo plano
    cam1 = CameraWorker(0) # Cámara 1
    cam2 = CameraWorker(2) # Cámara 2 (Descomenta si la tienes)
    
    cam1.start()
    cam2.start()

    # 2. Configurar Ventana de Control
    cv2.namedWindow("Panel de Control")
    cv2.createTrackbar("Umbral Suelo %", "Panel de Control", 30, 100, nada)
    cv2.createTrackbar("Umbral Sentado %", "Panel de Control", 65, 100, nada)

    print("SISTEMA OPERATIVO: Presiona ESC para salir.")

    # 3. Bucle visual (Main Loop)
    while True:
        # A. Leer Sliders (Actualizamos las variables globales)
        umbral_suelo = cv2.getTrackbarPos("Umbral Suelo %", "Panel de Control")
        umbral_sentado = cv2.getTrackbarPos("Umbral Sentado %", "Panel de Control")

        # B. Obtener imagen del trabajador 1
        if cam1.frame_listo is not None:
            frame_show = cam1.frame_listo.copy()
            
            # Dibujar Interfaz (HUD)
            color = (0, 255, 0)
            if cam1.estado_actual == "SENTADO": color = (0, 255, 255)
            if cam1.estado_actual == "SUELO/CAIDA": color = (0, 0, 255)

            # Barra superior
            cv2.rectangle(frame_show, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(frame_show, f"ESTADO: {cam1.estado_actual}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame_show, f"Altura: {int(cam1.altura_pct)}%", (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Camara 1 - Principal", frame_show)

        # C. (Opcional) Obtener imagen del trabajador 2
            if cam2.frame_listo is not None:
             cv2.imshow("Camara 2", cam2.frame_listo)

        # D. CONTROL DE SALIDA (Aquí es donde funciona el ESC)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): # 27 es ESC
            print("Cerrando sistema...")
            break
    
    # Limpieza final
    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente.")