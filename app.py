from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import urllib.request

app = Flask(__name__)

# 1. Cargar el modelo
print("Cargando modelo...")
model = tf.keras.models.load_model('modelo_digitos.keras')
print("Modelo cargado correctamente.")

nombres_clases = ["cero", "uno", "dos", "tres", "cuatro", 
                  "cinco", "seis", "siete", "ocho", "nueve"]

# ⚠️ CAMBIA ESTA IP por la que te dio IP Webcam
IP_WEBCAM = "http://192.168.1.50:8080"

# Variables globales
ultima_prediccion = {
    'detections': [],
    'total_numbers': 0
}

def preprocesar_digito(roi):
    """Preprocesa un dígito para que sea compatible con MNIST"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar a 20x20 (como MNIST) manteniendo aspect ratio
    height, width = gray.shape
    if height > width:
        new_height = 20
        new_width = int(width * 20 / height)
    else:
        new_width = 20
        new_height = int(height * 20 / width)
    
    resized = cv2.resize(gray, (new_width, new_height))
    
    # Crear canvas de 28x28 con el dígito centrado
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_height) // 2
    x_offset = (28 - new_width) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Normalizar
    normalized = canvas / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    return reshaped

def detectar_numeros(frame):
    """Detecta múltiples números en un frame"""
    detections = []
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detección de bordes adaptativa
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filtrar por área (tamaño del dígito)
        area = cv2.contourArea(contour)
        if area > 100:  # Ajusta este valor según la distancia
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar por aspect ratio (los dígitos no son muy alargados)
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 5.0:
                # Extraer el ROI (Región de Interés)
                roi = frame[y:y+h, x:x+w]
                
                # Preprocesar y predecir
                try:
                    processed = preprocesar_digito(roi)
                    predictions = model.predict(processed, verbose=0)
                    predicted_class = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                    
                    # Solo agregar si la confianza es alta
                    if confidence > 60:
                        detections.append({
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'prediction': nombres_clases[predicted_class],
                            'number': int(predicted_class),
                            'confidence': round(confidence, 2)
                        })
                except Exception as e:
                    print(f"Error procesando dígito: {e}")
    
    return detections

def generar_frames():
    """Captura video y dibuja las detecciones"""
    global ultima_prediccion
    
    url = f"{IP_WEBCAM}/video"
    cap = cv2.VideoCapture(url)
    
    contador = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame, reintentando...")
            cap.release()
            cap = cv2.VideoCapture(url)
            continue
        
        # Cada 15 frames, detectar números
        if contador % 15 == 0:
            detections = detectar_numeros(frame)
            ultima_prediccion = {
                'detections': detections,
                'total_numbers': len(detections)
            }
        
        # Dibujar las detecciones en el frame
        for det in ultima_prediccion['detections']:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Texto con predicción
            texto = f"{det['number']} ({det['confidence']}%)"
            cv2.putText(frame, texto, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar contador de números detectados
        cv2.putText(frame, f"Numeros: {ultima_prediccion['total_numbers']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Convertir frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        contador += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detections')
def get_detections():
    return jsonify(ultima_prediccion)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)