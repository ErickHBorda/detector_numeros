Aquí tienes el contenido listo para tu archivo `README.md`:
# 🔢 Detector Múltiple de Dígitos en Tiempo Real

Aplicación web que detecta múltiples números simultáneamente usando la cámara de un celular y una CNN entrenada con MNIST.

![Demo](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

## 🚀 Características

- ✅ Detección múltiple de dígitos en tiempo real
- ✅ Interfaz web moderna con diseño responsivo
- ✅ Visualización de confianza para cada detección
- ✅ Estadísticas en vivo

## 📋 Requisitos

- Python 3.8+
- App "IP Webcam" en tu celular (Android)
- PC y celular en la misma red WiFi

## 🚀 Instalación

```bash
git clone https://github.com/TU_USUARIO/detector-digitos-web.git
cd detector-digitos-web
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 📱 Configuración

1. Instala "IP Webcam" en tu celular
2. Edita `app.py` y cambia `IP_WEBCAM` con tu IP

## ▶️ Ejecutar

```bash
python app.py
```

Abre `http://localhost:5000` en tu navegador.

## 📂 Estructura

```
detector_digitos_web/
├── app.py              # Backend Flask
├── templates/
│   └── index.html      # Frontend
├── modelo_digitos.keras  # Modelo (no incluido en Git)
├── requirements.txt    # Dependencias
└── README.md
```

## 🎯 Uso

1. Inicia IP Webcam en tu celular
2. Ejecuta `app.py` en tu PC
3. Muestra números a la cámara (usa marcador grueso para mejores resultados)

## 🧠 Modelo

CNN entrenada con MNIST:
- Conv2D (32) → Pool → Conv2D (64) → Pool → Dense (64) → Dense (10)
- Precisión: ~98%


