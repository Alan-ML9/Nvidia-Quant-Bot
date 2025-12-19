
import os
import yfinance as yf
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

# --- REGLA DE MASA CRÍTICA (TU "SUELDO") ---
UMBRAL_SUELDO = 1000.0  # Hasta no tener $1,000, no sacamos nada.
PCT_SUELDO = 0.15       # 15% para ti, 85% reinversión.

# ------------------------------------------------------------------

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def calcular_patrimonio():
    print("🧮 Auditando Patrimonio (Masa Crítica)...")
    
    todos_tickers = list(shares_beta.keys()) + list(shares_reserva.keys()) + list(shares_satelite.keys())
    
    try:
        data = yf.download(todos_tickers, period="1d", interval="1d", auto_adjust=True)['Close']
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

    # Reporte Base
    msg = f"💰 **CAPITAL TOTAL: ${total_net_worth:,.2f}**\n"
    msg += f"_(Estrategia 50/20/30 | Meta Sueldo: ${UMBRAL_SUELDO})_\n"
    msg += "----------------------------\n"
    
    # Lógica de Masa Crítica (¿Cobras o Reviertes?)
    ganancia_teorica = total_net_worth - 100 # Asumiendo 100 de base inicial, esto se ajustará con el tiempo
    
    if total_net_worth < UMBRAL_SUELDO:
        faltante = UMBRAL_SUELDO - total_net_worth
        msg += "🚧 **FASE DE CONSTRUCCIÓN**\n"
        msg += f"• Objetivo: Llegar a ${UMBRAL_SUELDO}\n"
        msg += f"• Faltan: ${faltante:,.2f}\n"
        msg += "• Acción: **Reinvertir el 100% de ganancias.**\n"
    else:
        # Aquí ya superaste los $1,000
        msg += "🎉 **ZONA DE COBRO ACTIVADA**\n"
        # Calculamos sobre el exceso o ganancia del periodo (simplificado)
        msg += f"• Tu capital supera la masa crítica.\n"
        msg += f"• **Regla 15/85:** Puedes retirar el 15% de tus ganancias nuevas.\n"

    msg += "----------------------------\n"
    msg += "💡 **PLAN DE DESPLIEGUE (Rebalanceo):**\n"
    
    # Semáforos y Recomendaciones (Tu Plan de Compra para los $100)
    p_reserva = (val_reserva / total_net_worth) * 100
    
    if p_reserva > 90: # Caso Inicial ($100 Cash)
        monto_beta = total_net_worth * META_BETA
        monto_sat = total_net_worth * META_SATELITE
        msg += "🚀 **INICIO DE SISTEMA:**\n"
        msg += f"1. Compra **${monto_beta:.0f}** de VOO (Beta).\n"
        msg += f"2. Compra **${monto_sat:.0f}** dividido en NVDA/BTC/Tech.\n"
        msg += f"3. Mantén el resto (${total_net_worth * META_RESERVA:.0f}) en efectivo."
    else:
        # Rebalanceo normal
        p_satelite = (val_satelite / total_net_worth) * 100
        if p_satelite > 35:
            msg += "🚨 **Toma Ganancias:** Satélite muy alto. Vende y reinvierte en Beta."
        else:
            msg += "👌 Mantén el rumbo. Sigue acumulando."

    send_telegram(msg)
    print("✅ Reporte Masa Crítica enviado.")

if __name__ == "__main__":
    calcular_patrimonio()
