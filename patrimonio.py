import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests

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

# --- REGLA DE MASA CRÍTICA ---
UMBRAL_SUELDO = 1000.0  
PCT_SUELDO = 0.15       

# ------------------------------------------------------------------

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def analizar_riesgo():
    """Módulo 3: Análisis de Volatilidad (GARCH Simplificado)"""
    try:
        tickers = ["VOO", "BTC-USD"]
        # Descargamos 1 año para tener contexto
        data = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False)['Close']
        
        retornos = data.pct_change().dropna()
        # Volatilidad Rolling de 30 días anualizada
        vol_rolling = retornos.rolling(window=30).std() * np.sqrt(252) * 100
        
        # Último dato vs Promedio
        reporte = ""
        clima_global = "NEUTRO"
        
        for t in tickers:
            actual = vol_rolling[t].iloc[-1]
            media = vol_rolling[t].mean()
            
            if actual < media * 0.85:
                estado = "🟢 Calma"
            elif actual > media * 1.3:
                estado = "🔴 Tormenta"
                if t == "VOO": clima_global = "DEFENSIVO" # Si el S&P500 tiembla, cuidado
            else:
                estado = "🟡 Normal"
            
            reporte += f"• {t}: {estado} (Vol: {actual:.1f}%)\n"
            
        return reporte, clima_global
    except Exception as e:
        return f"⚠️ Error calculando riesgo: {e}", "NEUTRO"

def calcular_patrimonio():
    print("🧮 Auditando Patrimonio + Riesgo...")
    
    todos_tickers = list(shares_beta.keys()) + list(shares_reserva.keys()) + list(shares_satelite.keys())
    
    try:
        data = yf.download(todos_tickers, period="1d", interval="1d", auto_adjust=True, progress=False)['Close']
        if not isinstance(data, dict) and len(todos_tickers) == 1:
             current_prices = {todos_tickers[0]: data.iloc[-1]}
        else:
             current_prices = data.iloc[-1]
    except Exception as e:
        print(f"Error: {e}")
        send_telegram("❌ Error de conexión con Mercado.")
        return

    # Valor Total
    val_beta = sum(shares_beta[t] * current_prices[t] for t in shares_beta)
    val_reserva = dinero_efectivo + sum(shares_reserva[t] * current_prices[t] for t in shares_reserva)
    val_satelite = sum(shares_satelite[t] * current_prices[t] for t in shares_satelite)
    
    total_net_worth = val_beta + val_reserva + val_satelite
    if total_net_worth == 0: total_net_worth = 1

    # Ejecutar Módulo de Riesgo
    info_riesgo, clima = analizar_riesgo()

    # Construcción del Reporte
    msg = f"💰 **CAPITAL: ${total_net_worth:,.2f}**\n"
    msg += f"_(Estrategia 50/20/30)_\n"
    msg += "----------------------------\n"
    
    # Sección de Clima de Mercado
    msg += "🌪️ **SISMÓGRAFO DE MERCADO:**\n"
    msg += info_riesgo
    msg += f"**Modo Sugerido:** {clima}\n"
    msg += "----------------------------\n"

    # Lógica de Masa Crítica
    if total_net_worth < UMBRAL_SUELDO:
        faltante = UMBRAL_SUELDO - total_net_worth
        msg += "🚧 **FASE DE CONSTRUCCIÓN**\n"
        msg += f"• Meta Nivel 1: ${UMBRAL_SUELDO}\n"
        msg += f"• Falta: ${faltante:,.2f}\n"
    else:
        msg += "🎉 **Masa Crítica Lograda**\n"
        msg += "• Puedes retirar 15% de ganancias.\n"

    msg += "----------------------------\n"
    msg += "💡 **PLAN DE ACCIÓN:**\n"
    
    p_reserva = (val_reserva / total_net_worth) * 100
    
    if p_reserva > 90: # Inicio ($100 Cash)
        msg += "🚀 **DESPLIEGUE INICIAL:**\n"
        if clima == "DEFENSIVO":
            msg += "⚠️ Mercado nervioso. Entra despacio.\n"
            msg += "Sugerencia: Divide tu compra en 2 semanas.\n"
        else:
            msg += "✅ Mercado en calma. Ejecuta plan estándar:\n"
            msg += f"1. VOO (Beta): ${total_net_worth * META_BETA:.0f}\n"
            msg += f"2. Satélite: ${total_net_worth * META_SATELITE:.0f}\n"
            msg += "3. Reserva: Mantén el resto.\n"
    else:
        # Rebalanceo normal
        p_satelite = (val_satelite / total_net_worth) * 100
        if clima == "DEFENSIVO" and p_satelite > 25:
             msg += "🛡️ **ALERTA:** Alta volatilidad. Considera reducir Satélite y aumentar Reserva."
        elif p_satelite > 35:
            msg += "🚨 **Toma Ganancias:** Satélite muy alto (>35%)."
        else:
            msg += "👌 Mantén el rumbo."

    send_telegram(msg)
    print("✅ Auditoría Inteligente enviada.")

if __name__ == "__main__":
    calcular_patrimonio()

