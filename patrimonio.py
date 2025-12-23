
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
dinero_efectivo = 100  
shares_reserva = {"GLD": 0} 

# Satélite: Lista de activos (sin montos fijos todavía)
tickers_satelite = ["NVDA", "MSFT", "AAPL", "BTC-USD"]
# Aquí guardamos cuántas ACCIONES tienes (ahora es 0)
shares_satelite = {t: 0 for t in tickers_satelite}

# --- ESTRATEGIA ---
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
        response = requests.get(url, timeout=5)
        data = response.json()
        return int(data['data'][0]['value']), data['data'][0]['value_classification']
    except:
        return None, "Error API"

def calcular_pesos_risk_parity(capital_satelite):
    """
    MÓDULO 8: Asigna dinero inversamente proporcional a la volatilidad.
    Más riesgo = Menos dinero.
    """
    try:
        # Descargamos 3 meses de historia para calcular volatilidad reciente
        data = yf.download(tickers_satelite, period="3mo", interval="1d", auto_adjust=True, progress=False)['Close']
        
        # Calculamos volatilidad diaria (Desviación Estándar)
        retornos = data.pct_change().dropna()
        volatilidades = retornos.std()
        
        # Inversa de la volatilidad (1 / Vol)
        inv_vol = 1 / volatilidades
        sum_inv_vol = inv_vol.sum()
        
        # Pesos Normalizados (que sumen 100%)
        pesos = inv_vol / sum_inv_vol
        
        # Asignación de capital
        asignacion = {}
        reporte_pesos = ""
        
        for t in tickers_satelite:
            monto = capital_satelite * pesos[t]
            asignacion[t] = monto
            # Formateamos para el reporte: NVDA (25% -> $7.50)
            reporte_pesos += f"   • {t}: {pesos[t]*100:.1f}% (${monto:.2f})\n"
            
        return asignacion, reporte_pesos
        
    except Exception as e:
        # Fallback: Si falla Yahoo, usamos pesos iguales (1/N)
        peso_igual = capital_satelite / len(tickers_satelite)
        asignacion = {t: peso_igual for t in tickers_satelite}
        return asignacion, "⚠️ Error Data. Usando pesos iguales."

def calcular_patrimonio():
    print("🧮 Ejecutando V5.0 (Risk Parity)...")
    
    # 1. OBTENER PRECIOS ACTUALES
    todos_tickers = list(shares_beta.keys()) + list(shares_reserva.keys()) + tickers_satelite
    try:
        data = yf.download(todos_tickers, period="1d", interval="1d", auto_adjust=True, progress=False)['Close']
        if not isinstance(data, dict) and len(todos_tickers) == 1:
             current_prices = {todos_tickers[0]: data.iloc[-1]}
        else:
             current_prices = data.iloc[-1]
    except:
        send_telegram("❌ Error de conexión con Mercado.")
        return

    # 2. CALCULO VALOR ACTUAL
    val_beta = sum(shares_beta[t] * current_prices[t] for t in shares_beta)
    val_reserva = dinero_efectivo + sum(shares_reserva[t] * current_prices[t] for t in shares_reserva)
    val_satelite = sum(shares_satelite[t] * current_prices[t] for t in tickers_satelite) # Corrección aquí
    
    total_net_worth = val_beta + val_reserva + val_satelite
    if total_net_worth == 0: total_net_worth = 1

    # 3. GAP ANALYSIS
    ideal_satelite = total_net_worth * META_SATELITE
    gap_satelite = ideal_satelite - val_satelite
    
    # 4. CÁLCULO RISK PARITY (Solo si falta dinero en Satélite)
    detalle_risk_parity = ""
    compra_sugerida = {}
    if gap_satelite > 5:
        compra_sugerida, detalle_risk_parity = calcular_pesos_risk_parity(gap_satelite)

    # 5. INTELIGENCIA
    fng_val, fng_class = get_crypto_sentiment()

    # --- REPORTE ---
    msg = f"💰 **CAPITAL: ${total_net_worth:,.2f}**\n"
    msg += f"_(Modo: Risk Parity V5.0)_\n"
    msg += "----------------------------\n"
    
    msg += f"🧠 **Sentimiento:** {fng_val}/100 ({fng_class})\n"
    msg += "----------------------------\n"

    # SECCIÓN SATÉLITE INTELIGENTE
    msg += f"🚀 **Satélite:** ${val_satelite:.1f} (Meta ${ideal_satelite:.1f})\n"
    
    if gap_satelite > 5:
        msg += f"👉 **Falta: ${gap_satelite:.2f}**. Distribución Óptima:\n"
        msg += detalle_risk_parity
        msg += "_(Nota: Menos peso a lo más volátil)_"
    elif gap_satelite < -5:
        msg += f"⚠️ **Exceso:** ${abs(gap_satelite):.2f}. Rebalancear.\n"
    else:
        msg += "✅ En equilibrio.\n"

    msg += "\n----------------------------\n"
    
    # RESUMEN GENERAL
    gap_beta = (total_net_worth * META_BETA) - val_beta
    if gap_beta > 5: 
        msg += f"🏛 **Beta:** Faltan ${gap_beta:.2f} (Comprar VOO)\n"
    
    gap_reserva = (total_net_worth * META_RESERVA) - val_reserva
    if gap_reserva < -5:
         msg += f"💰 **Cash Disponible:** ${abs(gap_reserva):.2f}\n"

    send_telegram(msg)
    print("✅ Reporte V5.0 enviado.")

if __name__ == "__main__":
    calcular_patrimonio()



