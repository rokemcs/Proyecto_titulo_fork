import cv2
import mediapipe as mp
import threading

#Python version 3.11.9

feed = 'C:/Users/ROKEM/Downloads/Programming/Python\Proyecto_titulo_fork/videos/50wtf.mp4'
#feed = "rtsp://rokemusic:matrix2004@192.168.1.67:554/11"

cap = cv2.VideoCapture(feed)

# Configuración de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class CameraHandler:
    def __init__(self, camera_index, window_name):
        self.cap = cv2.VideoCapture(camera_index)
        self.window_name = window_name
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
        self.running = True

    def start(self):
        threading.Thread(target=self.update, args=()).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break

            # Procesamiento de imagen
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Dibujar esqueleto
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Lógica de detección de caída (Umbral vertical)
                # En coordenadas normalizadas, y=0 es arriba y y=1 es abajo.
                y_hombro = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                
                if y_hombro > 0.8: # Si el hombro está muy cerca del borde inferior
                    cv2.putText(frame, "ALERTA: POSIBLE CAIDA", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cap.release()
        cv2.destroyWindow(self.window_name)

# Iniciar cámaras (0 es la integrada, 1 y 2 suelen ser las USB Salandens)
cam1 = CameraHandler(feed, "Camara Esquina A")
#cam2 = CameraHandler(2, "Camara Esquina B")

cam1.start()
#cam2.start()