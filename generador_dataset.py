import cv2
import mediapipe as mp
import os
import csv

# Configuración de rutas
BASE_DIR = 'dataset'
CATEGORIES = {'normales': 0, 'caidas': 1}
OUTPUT_CSV = 'dataset_caidas.csv'

# Configuración de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def procesar_videos():
    # Preparar el archivo CSV y escribir los encabezados
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Crear encabezados: video_id, frame_num, label, x0, y0, z0, v0... x32, y32, z32, v32
        headers = ['video_id', 'frame_num', 'label']
        for i in range(33): # MediaPipe detecta 33 landmarks
            headers.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
        writer.writerow(headers)

        video_id_counter = 0

        # Iterar sobre las carpetas 'normales' (0) y 'caidas' (1)
        for category, label in CATEGORIES.items():
            folder_path = os.path.join(BASE_DIR, category)
            
            if not os.path.exists(folder_path):
                print(f"Advertencia: La carpeta {folder_path} no existe.")
                continue

            for video_name in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_name)
                
                # Ignorar archivos que no sean videos
                if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
                    continue
                
                print(f"Procesando: {video_name} (Etiqueta: {label})")
                cap = cv2.VideoCapture(video_path)
                frame_num = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break # Fin del video
                    
                    # Convertir a RGB para MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    
                    # Si detecta a una persona, extraer y guardar las coordenadas
                    if results.pose_landmarks:
                        row = [video_id_counter, frame_num, label]
                        
                        for landmark in results.pose_landmarks.landmark:
                            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                            
                        writer.writerow(row)
                    
                    frame_num += 1
                
                cap.release()
                video_id_counter += 1

    print(f"\n¡Procesamiento completo! Datos guardados en {OUTPUT_CSV}")

if __name__ == "__main__":
    procesar_videos()