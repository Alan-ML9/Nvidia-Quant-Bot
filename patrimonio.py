
import os
import yfinance as yf
import requests

# --- TUS LLAVES ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- CONFIGURACIÓN DE TU REALIDAD ACTUAL ($100 CASH) ---
# Tienes 0 acciones porque aún no compras nada
shares_beta = {"VOO": 0} 

# Tienes $100 dólares listos para disparar
dinero_efectivo = 100         

shares_reserva = {"GLD": 0} 

shares_satelite = {
    "NVDA": 0,
    "MSFT": 0,
    "AAPL": 0,
    "BTC-USD": 0
}

# --- TUS NUEVAS METAS (ESTRATEGIA 50/20/30) ---
META_BETA = 0.50      # Cimiento (Buffett)
META_RESERVA = 0.20   # Seguridad (Dalio)
META_SATELITE = 0.30  # Crecimiento (Simons)

# ------------------------------------------------------------------

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def calcular_patrimonio():
    print("🧮 Auditando Patrimonio Inicial...")
    
    # Descargamos precios para tenerlos listos (aunque tengas 0, los necesitamos para calcular cuánto puedes comprar)
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

    # Valor Total por Cubeta (Será 0 en acciones, 100 en reserva)
    val_beta = sum(shares_beta[t] * current_prices[t] for t in shares_beta)
    
    val_reserva_invested = sum(shares_reserva[t] * current_prices[t] for t in shares_reserva)
    val_reserva = dinero_efectivo + val_reserva_invested
    
    val_satelite = sum(shares_satelite[t] * current_prices[t] for t in shares_satelite)
    
    total_net_worth = val_beta + val_reserva + val_satelite
    
    # Evitar división por cero si no hubiera dinero
    if total_net_worth == 0:
        total_net_worth = 1 # Parche temporal

    # Porcentajes Reales
    p_beta = (val_beta / total_net_worth) * 100
    p_reserva = (val_reserva / total_net_worth) * 100
    p_satelite = (val_satelite / total_net_worth) * 100
    
    # Reporte
    msg = f"💰 **CAPITAL INICIAL: ${total_net_worth:,.2f}**\n"
    msg += f"_(Modo Despliegue 50/20/30)_\n"
    msg += "----------------------------\n"
    
    msg += f"🏛 **Beta:** {p_beta:.1f}% (Meta {META_BETA*100:.0f}%)\n"
    msg += f"🛡 **Reserva:** {p_reserva:.1f}% (Meta {META_RESERVA*100:.0f}%)\n"
    msg += f"🚀 **Satélite:** {p_satelite:.1f}% (Meta {META_SATELITE*100:.0f}%)\n"
    msg += "----------------------------\n"
    
    # Lógica de "Primeros Pasos" para $100
    msg += "💡 **PLAN DE COMPRA SUGERIDO:**\n"
    
    if p_reserva > 90: # Si tienes todo en cash
        monto_beta = total_net_worth * META_BETA
        monto_sat = total_net_worth * META_SATELITE
        
        msg += f"1️⃣ **Compra Beta (${monto_beta:.0f}):**\n"
        msg += f"   • {monto_beta/current_prices['VOO']:.4f} acciones de VOO\n\n"
        
        msg += f"2️⃣ **Compra Satélite (${monto_sat:.0f}):**\n"
        # Dividimos el capital satélite entre los 4 activos (25% c/u del satélite)
        usd_per_asset = monto_sat / 4
        for t in shares_satelite:
            msg += f"   • {t}: {usd_per_asset/current_prices[t]:.4f} títulos (${usd_per_asset:.2f})\n"
            
        msg += f"\n3️⃣ **Reserva:** Quédate con los ${total_net_worth * META_RESERVA:.0f} restantes en cash."
    
    else:
        # Lógica estándar de rebalanceo
        if p_satelite > 35:
            msg += "🚨 Tu Satélite está muy alto. Vende y asegura ganancias."
        elif p_reserva < 15:
            msg += "🆘 Reserva baja. Detén compras."
        else:
            msg += "👌 Mantén el rumbo."

    send_telegram(msg)
    print("✅ Plan de Despliegue enviado.")

if __name__ == "__main__":
    calcular_patrimonio()
