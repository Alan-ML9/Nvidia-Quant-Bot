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
    """Módulo 7: Feature Engineering (External API)"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        return value, classification
    except Exception as e:
        return None, "Error API"

def analizar_riesgo_stock():
    """Módulo 3: Sismógrafo de Volatilidad para VOO"""
    try:
        # Descargamos solo VOO para ser eficientes
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
    print("🧮 Ejecutando Auditoría Quant V3.0...")
    
    # 1. OBTENER PRECIOS
    todos_tickers = list(shares_beta.keys()) + list(shares_reserva.keys()) + list(shares_satelite.keys())
    try:
        data = yf.download(todos_tickers, period="1d", interval="1d", auto_adjust=True, progress=False)['Close']
        if not isinstance(data, dict) and len(todos_tickers) == 1:
             current_prices = {todos_tickers[0]: data.iloc[-1]}
        else:
             current_prices = data.iloc[-1]
    except Exception as e:
        print(f"Error Mercado: {e}")
        send_telegram("❌ Error de conexión con Yahoo Finance.")
        return

    # 2. CALCULO DE VALOR
    val_beta = sum(shares_beta[t] * current_prices[t] for t in shares_beta)
    val_reserva = dinero_efectivo + sum(shares_reserva[t] * current_prices[t] for t in shares_reserva)
    val_satelite = sum(shares_satelite[t] * current_prices[t] for t in shares_satelite)
    
    total_net_worth = val_beta + val_reserva + val_satelite
    if total_net_worth == 0: total_net_worth = 1

    # 3. INTELIGENCIA DE MERCADO (SENTIMIENTO + RIESGO)
    fng_val, fng_class = get_crypto_sentiment()
    riesgo_voo = analizar_riesgo_stock()

    # --- CONSTRUCCIÓN DEL REPORTE ---
    msg = f"💰 **CAPITAL: ${total_net_worth:,.2f}**\n"
    msg += f"_(Estrategia 50/20/30 | V3.0)_\n"
    msg += "----------------------------\n"
    
    # SECCIÓN DE INTELIGENCIA
    msg += "🧠 **CEREBRO DE MERCADO:**\n"
    msg += f"• **Sentimiento Crypto:** {fng_val}/100 ({fng_class})\n"
    msg += f"• **Riesgo S&P500:** {riesgo_voo}\n"
    
    # Interpretación Táctica
    tactica = "NEUTRO"
    if fng_val is not None:
        if fng_val < 25: 
            msg += "💎 **Oportunidad Satélite:** Miedo Extremo detected. Compra fuerte.\n"
            tactica = "AGRESIVO"
        elif fng_val > 75: 
            msg += "⚠️ **Alerta Satélite:** Avaricia Extrema. No compres, toma ganancias.\n"
            tactica = "DEFENSIVO"
    msg += "----------------------------\n"

    # LÓGICA DE MASA CRÍTICA
    if total_net_worth < UMBRAL_SUELDO:
        faltante = UMBRAL_SUELDO - total_net_worth
        msg += f"🚧 **FASE DE CONSTRUCCIÓN** (-${faltante:,.0f})\n"
    else:
        msg += "🎉 **Masa Crítica Lograda** (Retiros habilitados)\n"
    msg += "----------------------------\n"

    # PLAN DE ACCIÓN
    msg += "💡 **PLAN TÁCTICO:**\n"
    p_reserva = (val_reserva / total_net_worth) * 100
    
    if p_reserva > 90: # Caso Inicial
        msg += "🚀 **DESPLIEGUE INICIAL:**\n"
        if tactica == "DEFENSIVO":
            msg += "• Mercado caliente. Entra con el 50% hoy y 50% en 1 semana.\n"
        else:
            msg += "• Mercado favorable. Ejecuta compras hoy.\n"
            msg += f"1. VOO: ${total_net_worth * META_BETA:.0f}\n"
            msg += f"2. Satélite: ${total_net_worth * META_SATELITE:.0f}\n"
    else:
        # Rebalanceo normal
        p_satelite = (val_satelite / total_net_worth) * 100
        if tactica == "AGRESIVO" and p_satelite < 35:
            msg += "🔥 **Ataque:** Usa Reserva excedente para comprar Satélite."
        elif tactica == "DEFENSIVO" and p_satelite > 25:
            msg += "🛡️ **Defensa:** Reduce Satélite, aumenta Reserva."
        else:
            msg += "👌 Mantén el rumbo."

    send_telegram(msg)
    print("✅ Reporte V3.0 enviado.")

if __name__ == "__main__":
    calcular_patrimonio()


