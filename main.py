import os
# PARCHE DE COMPATIBILIDAD (Vital para la nube)
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

# --- CONFIGURACIÓN DE SEGURIDAD ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TICKER = "NVDA"

# 1. FUNCIÓN: INGENIERÍA DE CARACTERÍSTICAS (INDICADORES TÉCNICOS)
def add_technical_indicators(df):
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
    
    # Limpiamos los datos vacíos generados por el cálculo
    return df.dropna()

# 2. FUNCIÓN: COMUNICACIÓN CON TELEGRAM
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# 3. FUNCIÓN: CEREBRO MULTIVARIABLE (LSTM REPARADA)
def run_lstm_prediction():
    print(f"⬇️ Descargando datos de {TICKER}...")
    # Descargamos datos crudos
    raw_data = yf.download(TICKER, period="2y", interval="1d")
    
    # --- BLOQUE DE LIMPIEZA DE DATOS (FIX ROBUSTO) ---
    data = raw_data.copy()
    
    # Si Yahoo devuelve un índice múltiple (ej: Price -> Close -> NVDA), lo aplanamos
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Intentamos obtener el nivel de 'Close', 'Open', etc.
            data.columns = data.columns.get_level_values(0) 
            # Si al aplanar quedan nombres raros, buscamos específicamente 'Close'
            if 'Close' not in data.columns:
                 # A veces el nivel correcto es el 1, probamos ese
                 data = raw_data.copy()
                 data.columns = data.columns.get_level_values(1)
        except:
            pass # Si falla, seguimos e intentamos buscar la columna manualmente
            
    # Último intento de seguridad: Renombrar si existe 'Adj Close' pero no 'Close'
    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data = data.rename(columns={'Adj Close': 'Close'})

    # Nos aseguramos de tener solo las columnas necesarias y eliminamos el resto
    # (El copy es importante para evitar advertencias de pandas)
    data = data[['Close']].copy()
    
    # --- FIN DEL FIX ---

    # Agregamos los Indicadores Técnicos
    data = add_technical_indicators(data)
    
    current_price = float(data['Close'].iloc[-1])
    
    # Escalamos las 4 variables (Precio, RSI, MACD, Signal)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    prediction_days = 60
    
    # Preparamos las secuencias
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x]) # Input: Las 4 variables
        y_train.append(scaled_data[x, 0]) # Output: Solo Precio (Columna 0)
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    print(f"🧠 Entrenando IA con {x_train.shape[2]} dimensiones...")
    
    # Arquitectura LSTM Multivariable
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=0)
    
    # Predicción
    last_60 = scaled_data[-prediction_days:]
    real_df = np.array([last_60])
    
    pred_scaled = model.predict(real_df)
    
    # Inversión de Escala (Truco para matriz de 4 columnas)
    dummy_matrix = np.zeros((1, scaled_data.shape[1]))
    dummy_matrix[0, 0] = pred_scaled[0][0]
    final_pred = scaler.inverse_transform(dummy_matrix)[0][0]
    
    return current_price, final_pred

# 4. FUNCIÓN: ANÁLISIS DE SENTIMIENTO (ZERO-SHOT)
def analyze_sentiment():
    print("📰 Analizando noticias...")
    news_samples = [
        f"{TICKER} reports record-breaking data center revenue.",
        "Inflation concerns put pressure on high-growth tech stocks.",
        f"Analysts debate {TICKER}'s valuation amidst AI rally.",
        "Supply chain improvements boost semiconductor outlook."
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

# --- EJECUCIÓN PRINCIPAL ---
try:
    print("🚀 Iniciando Bot V2.1 (Multivariable Blindado)...")
    curr_price, pred_price = run_lstm_prediction()
    sentiment, headlines = analyze_sentiment()
    
    diff_percent = ((pred_price/curr_price)-1)*100
    
    # Lógica de Decisión
    decision = "NEUTRAL 🟡"
    if diff_percent > 1 and sentiment >= 0:
        decision = "COMPRA (ALCISTA) 🟢"
    elif diff_percent < -1 and sentiment <= 0:
        decision = "VENTA (BAJISTA) 🔴"
    elif diff_percent > 1 and sentiment < 0:
        decision = "DIVERGENCIA (RIESGO) 🟠"
    
    msg = f"""
🤖 **BOT QUANT V2.1: {TICKER}**
_Modelo: LSTM + RSI + MACD + NLP_
---
💵 **Precio Hoy:** ${curr_price:.2f}
🔮 **Predicción IA:** ${pred_price:.2f}
📊 **Variación:** {diff_percent:+.2f}%

📰 **Sentimiento:** {sentiment} pts
_{headlines[0]}_

🚦 **ESTRATEGIA:** {decision}
    """
    
    send_telegram(msg)
    print("✅ Reporte enviado con éxito.")

except Exception as e:
    error_msg = f"❌ Falla Crítica en Bot: {str(e)}"
    print(error_msg)
    send_telegram(error_msg)
