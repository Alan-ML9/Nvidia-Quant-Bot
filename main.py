import os
# PARCHE DE COMPATIBILIDAD: Debe ir antes de cualquier otro import
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from transformers import pipeline
import requests
import random

# --- CONFIGURACIÓN DE SEGURIDAD ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TICKER = "NVDA"

# 1. FUNCIÓN: COMUNICACIÓN CON TELEGRAM (CON REPORTE DE ERRORES)
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"❌ Error de API Telegram: {response.text}")
    else:
        print("✅ Mensaje entregado a Telegram.")

# 2. FUNCIÓN: CEREBRO NUMÉRICO (LSTM - DEEP LEARNING)
def run_lstm_prediction():
    print("⬇️ Descargando datos de Yahoo Finance...")
    data = yf.download(TICKER, period="2y", interval="1d")
    
    # Limpieza de datos (Módulo I: Ciencia de Datos)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    else:
        data = data[['Close']]
    
    current_price = float(data.iloc[-1].iloc[0] if hasattr(data.iloc[-1], 'iloc') else data.iloc[-1])
    
    # Normalización (Módulo III: Redes Neuronales)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    x_train, y_train = [], []
    prediction_days = 60
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Arquitectura de la Red (Deep Learning for Finance)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("🏋️ Entrenando modelo de predicción...")
    model.fit(x_train, y_train, epochs=12, batch_size=32, verbose=0)
    
    # Predicción para el cierre de mañana
    last_60 = scaled_data[-prediction_days:]
    real_df = np.array([last_60])
    real_df = np.reshape(real_df, (real_df.shape[0], real_df.shape[1], 1))
    
    prediction = model.predict(real_df)
    final_pred = scaler.inverse_transform(prediction)[0][0]
    
    return current_price, final_pred

# 3. FUNCIÓN: CEREBRO LÓGICO (NLP - ZERO-SHOT CLASSIFICATION)
def analyze_sentiment():
    print("🧠 Analizando contexto de noticias con BART...")
    # Simulamos el flujo de noticias (Módulo III: Modelos Multi-modales)
    news_samples = [
        f"{TICKER} breaks revenue records driven by AI data center demand.",
        "Regulatory pressure increases on semiconductor exports to Asia.",
        f"Competitors are launching new chips to challenge {TICKER}'s dominance.",
        "Investment firms upgrade price targets for AI sector."
    ]
    todays_news = random.sample(news_samples, 2)
    
    # Clasificador de contexto (Corrige el error de FinBERT con competidores)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = [f"Positive for {TICKER}", f"Negative for {TICKER}"]
    
    score = 0
    for text in todays_news:
        res = classifier(text, candidate_labels=labels)
        if res['labels'][0] == f"Positive for {TICKER}":
            score += 1
        else:
            score -= 1
            
    return score, todays_news

# --- FLUJO PRINCIPAL DE TOMA DE DECISIONES ---
try:
    print("🚀 Iniciando sistema cuantitativo...")
    
    curr_price, pred_price = run_lstm_prediction()
    sentiment, headlines = analyze_sentiment()
    
    # Lógica de Trading (Módulo I: Toma de decisiones)
    # Definimos si la predicción es alcista o bajista
    is_bullish_tech = pred_price > curr_price
    
    status = "ESPERAR 🟡"
    if is_bullish_tech and sentiment > 0:
        status = "COMPRA FUERTE 🟢"
    elif not is_bullish_tech and sentiment < 0:
        status = "VENTA/ALERTA 🔴"
    elif is_bullish_tech and sentiment <= 0:
        status = "DIVERGENCIA (RIESGO ALTO) 🟠"

    # Construcción del reporte para Telegram
    reporte = f"""
📈 **REPORTE CUANTITATIVO: {TICKER}**
---
💵 **Precio Actual:** ${curr_price:.2f}
🔮 **Predicción IA (Mañana):** ${pred_price:.2f}
📊 **Diferencia:** {((pred_price/curr_price)-1)*100:+.2f}%

📰 **Sentimiento (NLP):** {sentiment}
_{headlines[0]}_

🚦 **ACCIÓN:** {status}
    """
    
    send_telegram(reporte)
    print("✅ Proceso completado exitosamente.")

except Exception as e:
    error_msg = f"❌ **Falla en el Sistema:** {str(e)}"
    print(error_msg)
    send_telegram(error_msg)
