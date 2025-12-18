import os
# PARCHE DE COMPATIBILIDAD
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from transformers import pipeline
import requests
import random

# --- CONFIGURACIÓN ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TICKER = "NVDA"

# 1. FUNCIÓN: CALCULADORA DE INDICADORES (Feature Engineering)
def add_technical_indicators(df):
    # Evitamos advertencias de pandas copiando el dataframe
    df = df.copy()
    
    # A) RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # B) MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Limpiamos los NaNs generados por los cálculos (los primeros días)
    df = df.dropna()
    return df

# 2. FUNCIÓN: COMUNICACIÓN TELEGRAM
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# 3. FUNCIÓN: CEREBRO MULTIVARIABLE (LSTM AVANZADA)
def run_lstm_prediction():
    print(f"⬇️ Descargando datos de {TICKER}...")
    # Descargamos más historia para que los indicadores se calculen bien
    data = yf.download(TICKER, period="2y", interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    else:
        data = data[['Close']]
        
    # Agregamos los "Lentes" (Indicadores)
    data = add_technical_indicators(data)
    
    # Guardamos el precio actual para comparar después
    current_price = float(data['Close'].iloc[-1])
    
    # --- PREPROCESAMIENTO COMPLEJO ---
    # Escalamos TODO (Precio, RSI, MACD, Signal)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Definimos X (Entradas) e y (Objetivo)
    x_train, y_train = [], []
    prediction_days = 60
    
    # El objetivo ('y') es solo la columna 0 (Precio de Cierre)
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x]) # Toma las 4 columnas
        y_train.append(scaled_data[x, 0]) # Predice solo la columna 0 (Precio)
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # --- ARQUITECTURA NEURONAL V2 ---
    print(f"🧠 Entrenando IA con {x_train.shape[2]} variables (Precio + Indicadores)...")
    model = Sequential()
    # input_shape ahora se adapta automáticamente a (60, 4)
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2)) # Evita memorización
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Salida: 1 solo número (Precio)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=0)
    
    # --- PREDICCIÓN FUTURA ---
    # Tomamos los últimos 60 días (con sus 4 indicadores)
    last_60 = scaled_data[-prediction_days:]
    real_df = np.array([last_60])
    
    pred_scaled = model.predict(real_df)
    
    # TRUCO MATEMÁTICO: Inversión de Escala
    # El scaler espera 4 columnas para des-escalar, pero el modelo solo escupe 1 (precio).
    # Creamos una matriz fantasma con ceros y ponemos la predicción en la columna 0.
    dummy_matrix = np.zeros((1, scaled_data.shape[1]))
    dummy_matrix[0, 0] = pred_scaled[0][0]
    
    final_pred = scaler.inverse_transform(dummy_matrix)[0][0]
    
    return current_price, final_pred

# 4. FUNCIÓN: ANÁLISIS DE SENTIMIENTO (ZERO-SHOT)
def analyze_sentiment():
    print("📰 Analizando noticias...")
    news_samples = [
        f"{TICKER} shares surge as AI demand continues to grow.",
        "Market volatility increases ahead of Federal Reserve meeting.",
        f"Analysts question {TICKER}'s valuation after recent rally.",
        "New trade restrictions might impact semiconductor sector revenues."
    ]
    todays_news = random.sample(news_samples, 2)
    
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

# --- EJECUCIÓN ---
try:
    print("🚀 Iniciando Bot V2.0 (Multivariable)...")
    curr_price, pred_price = run_lstm_prediction()
    sentiment, headlines = analyze_sentiment()
    
    diff_percent = ((pred_price/curr_price)-1)*100
    
    # Lógica de Decisión Mejorada
    decision = "NEUTRAL 🟡"
    if diff_percent > 1 and sentiment >= 0:
        decision = "COMPRA (ALCISTA) 🟢"
    elif diff_percent < -1 and sentiment <= 0:
        decision = "VENTA (BAJISTA) 🔴"
    
    msg = f"""
🤖 **BOT QUANT V2.0: {TICKER}**
_Modelo Híbrido: LSTM + RSI + MACD_
---
💵 **Precio Hoy:** ${curr_price:.2f}
🔮 **Predicción IA:** ${pred_price:.2f}
📊 **Variación:** {diff_percent:+.2f}%

📰 **Sentimiento:** {sentiment} pts
_{headlines[0]}_

🚦 **ESTRATEGIA:** {decision}
    """
    
    send_telegram(msg)
    print("✅ Reporte V2 enviado.")

except Exception as e:
    print(f"❌ Error crítico: {e}")
    send_telegram(f"❌ Bot V2 Falló: {e}")
