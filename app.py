from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# 1. Cargar el modelo
print("Cargando modelo...")
model = tf.keras.models.load_model('modelo_digitos.keras')
print("Modelo cargado correctamente.")

nombres_clases = ["cero", "uno", "dos", "tres", "cuatro", 
                  "cinco", "seis", "siete", "ocho", "nueve"]

# ⚠️ CAMBIA ESTA IP por la que te dio IP Webcam
IP_WEBCAM = "http://10.217.124.17:8080"

# Variables globales
estado = {
    'activo': False,
    'prediccion': 'Esperando...',
    'numero': -1,
    'confianza': 0,
    'historial': []
}

cap = None

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
    global estado, cap
    
    url = f"{IP_WEBCAM}/video"
    
    while estado['activo']:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print("Error al conectar con la cámara, reintentando...")
                import time
                time.sleep(2)
                continue
        
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame, reconectando...")
            cap.release()
            cap = None
            continue
        
        # Hacer predicción
        resultado = predecir_frame(frame)
        estado['prediccion'] = resultado['prediction']
        estado['numero'] = resultado['number']
        estado['confianza'] = resultado['confidence']
        
        # Agregar al historial (últimos 10)
        estado['historial'].append({
            'numero': resultado['number'],
            'confianza': resultado['confidence']
        })
        if len(estado['historial']) > 10:
            estado['historial'].pop(0)
        
        # Dibujar predicción en el frame
        color = (0, 255, 136) if resultado['confidence'] > 80 else (0, 200, 255)
        texto = f"{resultado['prediction'].upper()} ({resultado['confidence']}%)"
        
        # Fondo semitransparente para el texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, texto, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Convertir frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Liberar cámara cuando se detiene
    if cap is not None:
        cap.release()
        cap = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream de video con predicciones"""
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Iniciar detección"""
    estado['activo'] = True
    return jsonify({'status': 'started', 'message': 'Detección iniciada'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Detener detección"""
    estado['activo'] = False
    return jsonify({'status': 'stopped', 'message': 'Detección detenida'})

@app.route('/api/status')
def get_status():
    """Devuelve el estado actual"""
    return jsonify({
        'activo': estado['activo'],
        'prediccion': estado['prediccion'],
        'numero': estado['numero'],
        'confianza': estado['confianza'],
        'historial': estado['historial']
    })

@app.route('/api/reset', methods=['POST'])
def reset_historial():
    """Reiniciar historial"""
    estado['historial'] = []
    return jsonify({'status': 'reset', 'message': 'Historial reiniciado'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)