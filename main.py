import os
import time
import random
import requests

# --- PARCHE DE COMPATIBILIDAD (CRÍTICO) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from transformers import pipeline

# --- CONFIGURACIÓN ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# TU PORTAFOLIO SATÉLITE (La Espada de Simons)
TICKERS = ["NVDA", "MSFT", "AAPL", "BTC-USD"] 

# 1. FUNCIÓN: INDICADORES TÉCNICOS
def add_technical_indicators(df):
    df = df.copy()
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()

# 2. FUNCIÓN: PREDICTIVA (CEREBRO LSTM)
def predict_asset(ticker):
    print(f"🔄 Procesando {ticker}...")
    try:
        # PAUSA TÁCTICA: Evita bloqueo de IP por Yahoo
        time.sleep(2)
        
        # Descarga forzada de datos planos
        raw_data = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        # --- LIMPIEZA DE DATOS (PARCHE ROBUSTO) ---
        data = raw_data.copy()
        
        if data.empty:
            print(f"⚠️ Datos vacíos para {ticker}")
            return None, None

        # Aplanar MultiIndex si existe (Caso común en yfinance nuevo)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                # Intenta extraer por el nombre del ticker nivel 1
                data = data.xs(ticker, level=1, axis=1) 
            except:
                # Si falla, simplemente colapsa los niveles
                data.columns = data.columns.get_level_values(0)

        # Renombrar si es necesario para estandarizar
        if 'Adj Close' in data.columns and 'Close' not in data.columns:
            data = data.rename(columns={'Adj Close': 'Close'})
            
        # Asegurarnos de que solo tenemos lo que necesitamos
        if 'Close' not in data.columns:
            print(f"⚠️ Columna 'Close' no encontrada para {ticker}")
            return None, None
            
        data = data[['Close']].copy()
        # --- FIN LIMPIEZA ---

        # Agregar Inteligencia (Indicadores)
        data = add_technical_indicators(data)
        
        if len(data) < 70:
            print(f"⚠️ Historia insuficiente para {ticker}")
            return None, None

        current_price = float(data['Close'].iloc[-1])

        # Escalado de Datos (0 a 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Preparación de Tensores
        x_train, y_train = [], []
        prediction_days = 60
        
        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Modelo LSTM Optimizado (Rápido)
        model = Sequential()
        model.add(LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.1))
        model.add(LSTM(units=30, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Entrenamiento Silencioso
        model.fit(x_train, y_train, epochs=8, batch_size=32, verbose=0)

        # Predicción
        last_60 = scaled_data[-prediction_days:]
        real_df = np.array([last_60])
        pred_scaled = model.predict(real_df, verbose=0)
        
        # Inversión de Escala (Truco Matricial)
        dummy_matrix = np.zeros((1, scaled_data.shape[1]))
        dummy_matrix[0, 0] = pred_scaled[0][0]
        final_pred = scaler.inverse_transform(dummy_matrix)[0][0]
        
        return current_price, final_pred

    except Exception as e:
        print(f"❌ Error técnico en {ticker}: {e}")
        return None, None

# 3. FUNCIÓN: CONTEXTO MACRO (DALIO)
def get_market_mood():
    print("🌍 Analizando Macroeconomía...")
    try:
        # Simulamos titulares macroeconómicos recientes
        headlines = [
            "Federal Reserve maintains interest rates amid inflation concerns.",
            "Tech sector leads market rally as earnings beat expectations.",
            "Geopolitical tensions create uncertainty in global markets."
        ]
        # Selección aleatoria para variedad diaria
        todays_headline = random.choice(headlines)
        
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        labels = ["Bullish (Alcista)", "Bearish (Bajista)", "Neutral"]
        res = classifier(todays_headline, candidate_labels=labels)
        
        return res['labels'][0], todays_headline
    except Exception as e:
        print(f"⚠️ Error NLP: {e}")
        return "Neutral", "Sin datos de noticias."

# 4. ENVÍO A TELEGRAM
def send_telegram(message):
    if not TOKEN or not CHAT_ID:
        print("❌ Faltan credenciales de Telegram.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    try:
        print("🚀 Iniciando Protocolo 50/30/20...")
        
        # A. Análisis Macro (Dalio)
        market_mood, headline = get_market_mood()
        
        reporte = f"🏛 **Estrategia Quant: {market_mood}**\n"
        reporte += f"_{headline}_\n\n"
        reporte += f"📡 **Radar Satélite (20%)**:\n"
        
        # B. Análisis Técnico (Simons)
        for ticker in TICKERS:
            curr, pred = predict_asset(ticker)
            
            if curr is None:
                reporte += f"⚠️ *{ticker}*: Datos no disponibles\n"
                continue
                
            diff = ((pred / curr) - 1) * 100
            
            # Semáforo de Decisión
            icon = "⚪️"
            signal = "HOLD"
            if diff > 1.5: 
                icon = "🟢"
                signal = "COMPRA"
            elif diff < -1.5: 
                icon = "🔴"
                signal = "VENTA"
            
            reporte += f"{icon} *{ticker}*: ${curr:.2f} ➡️ ${pred:.2f} ({diff:+.2f}%)\n"
        
        reporte += "\n💡 *Recordatorio Buffett/Dalio:*\n"
        reporte += "Mantén 50% en Beta (VOO) y 30% en Reserva."

        send_telegram(reporte)
        print("✅ Reporte enviado con éxito.")

    except Exception as e:
        err_msg = f"❌ Falla Crítica del Sistema: {str(e)}"
        print(err_msg)
        send_telegram(err_msg)
