
import os
import yfinance as yf
import requests

# --- TUS LLAVES ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- CONFIGURACIÓN DE ACTIVOS (CANTIDAD DE ACCIONES QUE TIENES) ---
# Pon aquí cuántas acciones tienes REALMENTE en tu broker hoy.
shares_beta = {"VOO": 0.08}  # Ejemplo basado en tu compra de hoy
dinero_efectivo = 30         # Tu reserva en cash (aprox)
shares_reserva = {"GLD": 0.0} 
shares_satelite = {
    "NVDA": 0.0279,
    "MSFT": 0.0103,
    "AAPL": 0.0184,
    "BTC-USD": 0.0001
}

# --- METAS ESTRATÉGICAS (EL PLAN 50/20/30) ---
META_BETA = 0.50      # Buffett (Cimiento)
META_RESERVA = 0.20   # Dalio (Seguridad Mínima)
META_SATELITE = 0.30  # Simons (Crecimiento Agresivo)

# ------------------------------------------------------------------

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def calcular_patrimonio():
    print("🧮 Auditando Patrimonio (Estrategia 50/20/30)...")
    
    todos_tickers = list(shares_beta.keys()) + list(shares_reserva.keys()) + list(shares_satelite.keys())
    
    try:
        data = yf.download(todos_tickers, period="1d", interval="1d", auto_adjust=True)['Close']
        if not isinstance(data, dict) and len(todos_tickers) == 1:
             current_prices = {todos_tickers[0]: data.iloc[-1]}
        else:
             current_prices = data.iloc[-1]
    except Exception as e:
        print(f"Error descargando: {e}")
        send_telegram("❌ Error de conexión con Mercado.")
        return

    # Valor Total por Cubeta
    val_beta = sum(shares_beta[t] * current_prices[t] for t in shares_beta)
    
    val_reserva_invested = sum(shares_reserva[t] * current_prices[t] for t in shares_reserva)
    val_reserva = dinero_efectivo + val_reserva_invested
    
    val_satelite = sum(shares_satelite[t] * current_prices[t] for t in shares_satelite)
    
    total_net_worth = val_beta + val_reserva + val_satelite
    
    # Porcentajes Reales
    p_beta = (val_beta / total_net_worth) * 100
    p_reserva = (val_reserva / total_net_worth) * 100
    p_satelite = (val_satelite / total_net_worth) * 100
    
    # Reporte
    msg = f"💰 **PATRIMONIO: ${total_net_worth:,.2f}**\n"
    msg += f"_(Estrategia Agresiva 50/20/30)_\n"
    msg += "----------------------------\n"
    
    # Semáforos (Umbral de tolerancia +/- 5%)
    icon_beta = "✅" if abs(p_beta - 50) < 5 else "⚠️"
    msg += f"🏛 **Beta:** {p_beta:.1f}% {icon_beta} (Meta 50%)\n"
    
    icon_res = "✅" if abs(p_reserva - 20) < 5 else "⚠️"
    msg += f"🛡 **Reserva:** {p_reserva:.1f}% {icon_res} (Meta 20%)\n"
    
    icon_sat = "✅" if abs(p_satelite - 30) < 5 else "⚠️"
    msg += f"🚀 **Satélite:** {p_satelite:.1f}% {icon_sat} (Meta 30%)\n"
    msg += "----------------------------\n"
    
    # Cerebro de Rebalanceo
    msg += "💡 **DIAGNÓSTICO:**\n"
    if p_satelite > 35:
        excess = val_satelite - (total_net_worth * 0.30)
        msg += f"🚨 **Toma Ganancias:** Vende ${excess:,.0f} de Tech y pásalo a Reserva."
    elif p_reserva > 30:
        msg += f"📉 **Exceso de Cash:** Tienes mucho dinero parado. Compra más Beta (VOO)."
    elif p_reserva < 15:
        msg += f"🆘 **Alerta de Liquidez:** Tu colchón es peligroso (<15%). Ahorra."
    else:
        msg += "👌 **Sistema en Equilibrio.**"

    send_telegram(msg)

if __name__ == "__main__":
    calcular_patrimonio()
