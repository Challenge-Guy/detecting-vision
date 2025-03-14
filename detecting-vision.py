import cv2
import numpy as np
from datetime import datetime,timedelta

# Función para detectar y contar varillas circulares en una región específica
def contar_varillas_circulares(frame, roi):
    # Recortar la región de interés (ROI) del frame
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
       
    # Convertir a escala de grises
    gris = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    desenfoque = cv2.GaussianBlur(gris, (15, 15), 0)
    
    # Usar la transformada de Hough para detectar círculos
    circulos = cv2.HoughCircles(desenfoque, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                param1=50, param2=30, minRadius=5, maxRadius=18)
    
    num_varillas = 0
    # Si se detectaron círculos, dibujarlos en el frame original
    if circulos is not None:
        circulos = np.round(circulos[0, :]).astype("int")
        num_varillas = len(circulos)
        for (cx, cy, r) in circulos:
            # Dibujar el círculo en el frame original
            cv2.circle(frame, (x + cx, y + cy), r, (0, 255, 0), 2)
            cv2.rectangle(frame, (x + cx - r, y + cy - r), (x + cx + r, y + cy + r), (0, 128, 255), 2)
    
    # Devolver el número de varillas detectadas y el frame con los círculos dibujados
    return num_varillas, frame

# Cargar video desde archivo
video_path = './videos/video1.mp4'
cap = cv2.VideoCapture(video_path)

# Obtener dimensiones del video
ret, frame = cap.read()
altura, ancho, _ = frame.shape

# Definir el tamaño de la ROI en píxeles (ajusta según la resolución del video)
cm_to_pixels = 20  # Número de píxeles que representan 1 cm (esto depende de tu cámara y configuración)
roi_size_w = 7 * cm_to_pixels  # Tamaño de la ROI en píxeles (7 cm)
roi_size_h = 7 * cm_to_pixels  # Tamaño de la ROI en píxeles (7 cm)

# Calcular las coordenadas de la ROI (cuadro de 7 x 7 cm en el centro)
roi_x = ancho // 2 - roi_size_w // 2
roi_y = 40
roi = (roi_x, roi_y, roi_size_w, roi_size_h)

# Inicializar el contador de varillas
total_varillas = 0

# Inicializar el sustractor de fondo
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

last_update_time = datetime.now()
current_time = timedelta(seconds=1)
cuenta=False

varillaActual=-1
varillaAnterior=0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aplicar la sustracción de fondo para detectar movimiento
    fgmask = fgbg.apply(frame)
    
    # Filtrar las áreas en movimiento horizontal en la ROI
    movement_mask = fgmask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    contours, _ = cv2.findContours(movement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar solo los contornos con movimiento horizontal significativo
    moving_objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > h:  # Asegurar que el movimiento sea mayormente horizontal
            moving_objects.append(contour)
    
    # Procesar solo si hay objetos en movimiento horizontal en la ROI
    if len(moving_objects) > 0:
        num_varillas, frame_con_varillas = contar_varillas_circulares(frame, roi)
        total_varillas += num_varillas
    
        # Dibujar la ROI en el frame
        cv2.rectangle(frame_con_varillas, (roi_x, roi_y), (roi_x + roi_size_w, roi_y + roi_size_h), (255, 0, 0), 2)
    
        # Mostrar el número de varillas detectadas
        cv2.putText(frame_con_varillas, f'Varillas en este frame: {num_varillas}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_con_varillas, f'Varillas totales: {total_varillas}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Mostrar el frame procesado
        cv2.imshow('Varillas', frame_con_varillas)
        
        
    else:
        # Dibujar la ROI en el frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size_w, roi_y + roi_size_h), (255, 0, 0), 2)
        # Mostrar el número de varillas detectadas
        cv2.putText(frame, f'Varillas en este frame: 0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Varillas totales: {total_varillas}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Mostrar el frame procesado
        cv2.imshow('Varillas', frame)
        
       
    varillaActual=total_varillas
    current_time = datetime.now()
    
    if (varillaActual!=varillaAnterior):
        cuenta=True
        last_update_time = current_time
        varillaAnterior=varillaActual
    
    
    if (current_time - last_update_time > timedelta(seconds=1)) and cuenta:
        
        # Actualizar los datos en Firebase en tiempo real
        fecha_actual = datetime.now().strftime('%Y-%m-%d')
        print('cantidad', str(total_varillas),'fecha', fecha_actual)
        
        cuenta=False
    
    
    
    
    
    
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

