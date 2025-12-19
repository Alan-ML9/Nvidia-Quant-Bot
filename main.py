import os
# PARCHE DE COMPATIBILIDAD KERAS
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

# TU SATÉLITE (20% DEL PORTAFOLIO) + EL MERCADO (MACRO)
TICKERS = ["NVDA", "MSFT", "AAPL", "BTC-USD"] # Activos Satélite
MARKET_TICKER = "SPY" # Referencia Macro (Dalio)

# 1. FUNCIÓN: INDICADORES TÉCNICOS (Simons)
def add_technical_indicators(df):
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df.dropna()

# 2. FUNCIÓN: ENTRENAMIENTO Y PREDICCIÓN (El Motor)
def predict_asset(ticker):
    print(f"🔄 Procesando {ticker}...")
    
    # Descarga Robusta
    raw_data = yf.download(ticker, period="2y", interval="1d")
    data = raw_data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)
    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data = data.rename(columns={'Adj Close': 'Close'})
    data = data[['Close']].copy()

    # Agregar Indicadores
    data = add_technical_indicators(data)
    current_price = float(data['Close'].iloc[-1])

    # Escalado
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Preparar X, y
    x_train, y_train = [], []
    prediction_days = 60
    
    # Si no hay suficientes datos (ej. error de descarga), regresamos None
    if len(scaled_data) < prediction_days + 10:
        return None

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Modelo Ligero (Optimizado para Loop)
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units=30, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Menos epochs para que no tarde mucho en procesar 4 activos
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predicción
    last_60 = scaled_data[-prediction_days:]
    real_df = np.array([last_60])
    pred_scaled = model.predict(real_df)
    
    dummy_matrix = np.zeros((1, scaled_data.shape[1]))
    dummy_matrix[0, 0] = pred_scaled[0][0]
    final_pred = scaler.inverse_transform(dummy_matrix)[0][0]
    
    return current_price, final_pred

# 3. FUNCIÓN: SENTIMIENTO DE MERCADO (NLP)
def get_market_mood():
    print("🌍 Analizando Contexto Macro...")
    # Usamos titulares genéricos de mercado para simular "Dalio"
    headlines = [
        "Fed signals interest rates might stay steady.",
        "Tech sector shows resilience despite inflation fears.",
        "Global markets await economic data reports."
    ]
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    res = classifier(headlines[0], candidate_labels=["Bullish Market", "Bearish Market"])
    return res['labels'][0]

# 4. ENVIAR REPORTE UNIFICADO
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# --- EJECUCIÓN MAESTRA ---
try:
    print("🚀 Iniciando Escaneo de Portafolio...")
    
    # A. Contexto Macro (Dalio)
    market_mood = get_market_mood()
    
    reporte = f"🏛 **Estrategia Quant: {market_mood}**\n"
    reporte += f"_Satélite (20%) - Análisis Diario_\n\n"
    
    # B. Loop de Activos (Simons)
    for ticker in TICKERS:
        try:
            curr, pred = predict_asset(ticker)
            if curr is None: continue
            
            diff = ((pred / curr) - 1) * 100
            
            # Iconografía
            icon = "⚪️"
            if diff > 1.5: icon = "🟢 COMPRA"
            elif diff < -1.5: icon = "🔴 VENTA"
            else: icon = "🟡 HOLD"
            
            reporte += f"*{ticker}*: ${curr:.2f} ➡️ ${pred:.2f} ({diff:+.2f}%)\n"
            reporte += f"└ {icon}\n\n"
            
        except Exception as e:
            print(f"Error en {ticker}: {e}")
            reporte += f"*{ticker}*: ⚠️ Error de Datos\n\n"

    reporte += "💡 _Recuerda: Mantén tu 30% en Reserva (Cetes/Gold)_"
    
    send_telegram(reporte)
    print("✅ Reporte de Portafolio Enviado.")

except Exception as e:
    print(f"❌ Error Global: {e}")
    send_telegram(f"❌ Falla Crítica: {e}")
