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
IP_WEBCAM = "http://10.217.124.17:8080"

# Variable global para almacenar la última predicción
ultima_prediccion = {
    'prediction': 'Esperando...',
    'number': -1,
    'confidence': 0
}

def predecir_frame(frame):
    """Preprocesa el frame y devuelve la predicción"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    predictions = model.predict(reshaped, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    return {
        'prediction': nombres_clases[predicted_class],
        'number': int(predicted_class),
        'confidence': round(confidence, 2)
    }

def generar_frames():
    """Captura video de IP Webcam y devuelve frames procesados"""
    global ultima_prediccion
    
    # URL del stream de video de IP Webcam
    url = f"{IP_WEBCAM}/video"
    
    # Abrir el stream con OpenCV
    cap = cv2.VideoCapture(url)
    
    contador = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame, reintentando...")
            cap.release()
            cap = cv2.VideoCapture(url)
            continue
        
        # Cada 10 frames, hacer predicción (para no saturar)
        if contador % 10 == 0:
            ultima_prediccion = predecir_frame(frame)
        
        # Dibujar la predicción en el frame
        texto = f"{ultima_prediccion['prediction'].upper()} ({ultima_prediccion['confidence']}%)"
        cv2.putText(frame, texto, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
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
    """Stream de video con predicciones"""
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """Devuelve la última predicción en JSON"""
    return jsonify(ultima_prediccion)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)