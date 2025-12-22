import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json

# --- TUS LLAVES ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- CONFIGURACIÓN DE TU REALIDAD ACTUAL ---
shares_beta = {"VOO": 0} 
dinero_efectivo = 100  # Tu capital inicial
shares_reserva = {"GLD": 0} 
shares_satelite = {
    "NVDA": 0, "MSFT": 0, "AAPL": 0, "BTC-USD": 0
}

# --- ESTRATEGIA 50/20/30 ---
META_BETA = 0.50
META_RESERVA = 0.20
META_SATELITE = 0.30
UMBRAL_SUELDO = 1000.0  

# ------------------------------------------------------------------

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def get_crypto_sentiment():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        return value, classification
    except:
        return None, "Error API"

def analizar_riesgo_stock():
    try:
        data = yf.download("VOO", period="1y", interval="1d", auto_adjust=True, progress=False)['Close']
        retornos = data.pct_change().dropna()
        vol_rolling = retornos.rolling(window=30).std() * np.sqrt(252) * 100
        actual = vol_rolling.iloc[-1]
        media = vol_rolling.mean()
        if actual < media * 0.85: return f"🟢 Calma ({actual:.1f}%)"
        elif actual > media * 1.3: return f"🔴 Tormenta ({actual:.1f}%)"
        else: return f"🟡 Normal ({actual:.1f}%)"
    except:
        return "⚠️ Error VOO"

def calcular_patrimonio():
    print("🧮 Ejecutando Auditoría V4.0 (Rebalanceo)...")
    
    # 1. OBTENER PRECIOS
    todos_tickers = list(shares_beta.keys()) + list(shares_reserva.keys()) + list(shares_satelite.keys())
    try:
        data = yf.download(todos_tickers, period="1d", interval="1d", auto_adjust=True, progress=False)['Close']
        if not isinstance(data, dict) and len(todos_tickers) == 1:
             current_prices = {todos_tickers[0]: data.iloc[-1]}
        else:
             current_prices = data.iloc[-1]
    except Exception as e:
        send_telegram("❌ Error de conexión con Yahoo Finance.")
        return

    # 2. CALCULO DE VALOR REAL
    val_beta = sum(shares_beta[t] * current_prices[t] for t in shares_beta)
    val_reserva = dinero_efectivo + sum(shares_reserva[t] * current_prices[t] for t in shares_reserva)
    val_satelite = sum(shares_satelite[t] * current_prices[t] for t in shares_satelite)
    
    total_net_worth = val_beta + val_reserva + val_satelite
    if total_net_worth == 0: total_net_worth = 1

    # 3. GAP ANALYSIS (MÓDULO 8)
    # Cuánto DEBERÍAS tener en cada cubeta
    ideal_beta = total_net_worth * META_BETA
    ideal_reserva = total_net_worth * META_RESERVA
    ideal_satelite = total_net_worth * META_SATELITE
    
    # Cuánto te FALTA (o sobra)
    gap_beta = ideal_beta - val_beta
    gap_reserva = ideal_reserva - val_reserva
    gap_satelite = ideal_satelite - val_satelite

    # 4. INTELIGENCIA DE MERCADO
    fng_val, fng_class = get_crypto_sentiment()
    riesgo_voo = analizar_riesgo_stock()

    # --- REPORTE EJECUTIVO ---
    msg = f"💰 **CAPITAL: ${total_net_worth:,.2f}**\n"
    msg += f"_(Estrategia 50/20/30)_\n"
    msg += "----------------------------\n"
    
    msg += "🧠 **CEREBRO:**\n"
    msg += f"• Crypto: {fng_val}/100 ({fng_class})\n"
    msg += f"• S&P500: {riesgo_voo}\n"
    msg += "----------------------------\n"

    msg += "⚖️ **CALCULADORA DE REBALANCEO:**\n"
    
    # BETA
    icon_beta = "🟢 Compra" if gap_beta > 1 else "✅ Ok"
    msg += f"🏛 **Beta:** ${val_beta:.1f} (Meta ${ideal_beta:.1f})\n"
    if gap_beta > 5: msg += f"   👉 **Falta: ${gap_beta:.2f}** (Prioridad)\n"
    
    # SATÉLITE
    icon_sat = "🟢 Compra" if gap_satelite > 1 else "✅ Ok"
    if gap_satelite < -10: icon_sat = "🔴 Vende (Exceso)"
    msg += f"🚀 **Satélite:** ${val_satelite:.1f} (Meta ${ideal_satelite:.1f})\n"
    if gap_satelite > 5: msg += f"   👉 **Falta: ${gap_satelite:.2f}**\n"
    elif gap_satelite < -5: msg += f"   ⚠️ **Sobra: ${abs(gap_satelite):.2f}**\n"

    # RESERVA
    msg += f"🛡 **Reserva:** ${val_reserva:.1f} (Meta ${ideal_reserva:.1f})\n"
    if gap_reserva > 5: msg += f"   👉 **Ahorra: ${gap_reserva:.2f}**\n"
    elif gap_reserva < -5: msg += f"   💰 **Disponible: ${abs(gap_reserva):.2f}** (Úsalo para comprar)\n"
    
    msg += "----------------------------\n"
    
    # PLAN DE ACCIÓN INTELIGENTE
    msg += "💡 **ORDEN DE EJECUCIÓN:**\n"
    
    # Caso 1: Tienes exceso de Reserva (Cash para gastar)
    if gap_reserva < -1: 
        cash_disponible = abs(gap_reserva)
        msg += f"Tienes **${cash_disponible:.2f}** extra en Reserva.\n"
        
        # ¿Dónde lo ponemos? Depende del GAP y del Sentimiento
        if gap_beta > 0 and gap_satelite > 0:
            msg += "• Distribúyelo proporcionalmente en Beta y Satélite.\n"
        elif gap_satelite > 0 and fng_val < 30: # Miedo Extremo
            msg += "🔥 **Oportunidad:** Miedo en Crypto. Mete todo al Satélite.\n"
        elif gap_beta > 0:
            msg += "• Rellena tu Beta (VOO) primero.\n"
    
    # Caso 2: Falta dinero en todo (Fase de Ahorro)
    elif gap_beta > 0 and gap_satelite > 0 and gap_reserva > 0:
        msg += "🚧 **Fase de Acumulación:**\n"
        msg += "• Deposita dinero nuevo. Todas las cubetas están vacías.\n"
    
    else:
        msg += "👌 **Sistema en Equilibrio.** No hagas nada.\n"

    send_telegram(msg)
    print("✅ Reporte V4.0 enviado.")

if __name__ == "__main__":
    calcular_patrimonio()



