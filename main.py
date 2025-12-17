import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from transformers import pipeline
import requests
import os
import random

# --- CONFIGURACIÓN (SECRETS) ---
# En la nube, estos valores se leerán de las variables de entorno para seguridad
TOKEN = os.environ.get("8478600402:AAG30QRs6Bn6YH4EZeHrjDmIU_h5wKyYfKk")
CHAT_ID = os.environ.get("7716811022")
TICKER = "NVDA"

# 1. FUNCIÓN: ENVIAR TELEGRAM
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, json=payload)

# 2. FUNCIÓN: OBTENER DATOS Y ENTRENAR LSTM
def run_lstm_prediction():
    print("⬇️ Descargando datos de mercado...")
    data = yf.download(TICKER, period="2y", interval="1d")
    
    # Limpieza rápida
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    else:
        data = data[['Close']]
    
    # Precio actual (último cierre)
    current_price = data.iloc[-1].item()
    
    # Preparamos datos para IA
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    x_train, y_train = [], []
    days = 60
    
    for x in range(days, len(scaled_data)):
        x_train.append(scaled_data[x-days:x, 0])
        y_train.append(scaled_data[x, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Creamos el modelo (Versión ligera para la nube)
    print("🧠 Entrenando Red Neuronal...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0) # Epochs bajas para velocidad
    
    # Predecir mañana
    last_60 = scaled_data[-days:]
    X_test = np.array([last_60])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    pred_scaled = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    
    return current_price, pred_price

# 3. FUNCIÓN: ANÁLISIS DE NOTICIAS (ZERO-SHOT)
def analyze_sentiment():
    print("📰 Leyendo noticias...")
    # Simulamos titulares de hoy (En versión Pro usaríamos NewsAPI)
    headlines_pool = [
        "NVIDIA reveals new AI chip architecture with 3x performance.",
        "Tech stocks slide as inflation data worries investors.",
        "Analyst downgrades semiconductor sector due to supply chain issues.",
        "NVIDIA partners with Tesla for autonomous driving.",
        "Competitors like AMD are gaining market share in the GPU space."
    ]
    # Seleccionamos 3 al azar para simular variedad diaria
    todays_news = random.sample(headlines_pool, 3)
    
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["Positive for NVIDIA", "Negative for NVIDIA"]
    
    total_score = 0
    for news in todays_news:
        res = classifier(news, candidate_labels=labels)
        if res['labels'][0] == "Positive for NVIDIA":
            total_score += 1
        else:
            total_score -= 1
            
    return total_score, todays_news

# --- EJECUCIÓN PRINCIPAL ---
try:
    print("🚀 Iniciando Bot Cuantitativo...")
    price, prediction = run_lstm_prediction()
    sentiment_score, news_list = analyze_sentiment()
    
    # Lógica de decisión
    signal = "NEUTRAL 🟡"
    if prediction > price and sentiment_score > 0:
        signal = "COMPRA FUERTE 🟢"
    elif prediction < price and sentiment_score < 0:
        signal = "VENTA FUERTE 🔴"
    
    # Mensaje para Telegram
    msg = f"""
🤖 **REPORTE DIARIO: {TICKER}** 🤖
    
💵 Precio Actual: ${price:.2f}
🔮 Predicción IA: ${prediction:.2f}
    
📰 Sentimiento Noticias: {sentiment_score}
(Basado en {len(news_list)} titulares analizados)
    
🚦 **SEÑAL FINAL:** {signal}
    """
    
    send_telegram(msg)
    print("✅ Reporte enviado con éxito.")

except Exception as e:
    print(f"❌ Error: {e}")
    send_telegram(f"❌ El Bot colapsó: {e}")