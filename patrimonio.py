import os
import yfinance as yf
import requests

# --- TUS LLAVES ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- CONFIGURACIÓN DE TU REALIDAD ACTUAL ---
# Si vendiste todo, tus acciones son 0.
shares_beta = {"VOO": 0}
shares_reserva = {"GLD": 0} 
shares_satelite = {"NVDA": 0, "MSFT": 0, "AAPL": 0, "BTC-USD": 0}

# ¿Cuánto dinero tienes listo para invertir? (Pon aquí el total en tu moneda)
dinero_total_inicial = 100  # Ejemplo: $10,000 USD (o lo equivalente en tu moneda)

# METAS DE TU ESTRATEGIA (50/30/20)
META_BETA = 0.50
META_RESERVA = 0.30
META_SATELITE = 0.20

# ------------------------------------------------------------------

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def calcular_despliegue():
    print("🧮 Calculando Órdenes de Entrada...")
    
    # 1. Obtenemos precios actuales para saber cuánto comprar
    todos_tickers = list(shares_beta.keys()) + list(shares_reserva.keys()) + list(shares_satelite.keys())
    
    try:
        data = yf.download(todos_tickers, period="1d", interval="1d", auto_adjust=True)['Close']
        if not isinstance(data, dict) and len(todos_tickers) == 1:
             current_prices = {todos_tickers[0]: data.iloc[-1]}
        else:
             current_prices = data.iloc[-1]
    except Exception as e:
        send_telegram("❌ Error obteniendo precios del mercado.")
        return

    # 2. Calculamos cuánto dinero va a cada cubeta
    monto_beta = dinero_total_inicial * META_BETA
    monto_reserva = dinero_total_inicial * META_RESERVA
    monto_satelite = dinero_total_inicial * META_SATELITE
    
    # 3. Construimos el Plan de Compras
    msg = f"💵 **PLAN DE DESPLIEGUE DE CAPITAL**\n"
    msg += f"Capital Total: ${dinero_total_inicial:,.2f}\n"
    msg += "---------------------------------\n\n"
    
    # --- A) BETA (Buffett) ---
    # Asumimos que todo el 50% va a VOO
    precio_voo = current_prices["VOO"]
    acciones_voo = monto_beta / precio_voo
    msg += f"🏛 **BETA (50%) - ${monto_beta:,.0f}**\n"
    msg += f"• Compra **{acciones_voo:.2f} acciones** de VOO\n"
    msg += f"_(Precio aprox: ${precio_voo:.2f})_\n\n"
    
    # --- B) RESERVA (Dalio) ---
    msg += f"🛡 **RESERVA (30%) - ${monto_reserva:,.0f}**\n"
    msg += f"• Mantén esto en **Cetes Directo** o Bonos.\n"
    msg += f"• Opcional: Compra {(monto_reserva*0.2)/current_prices['GLD']:.2f} de GLD (Oro).\n\n"
    
    # --- C) SATÉLITE (Simons) ---
    # Dividimos el capital satélite entre los 4 activos (Equiponderado: 25% c/u del 20% total)
    capital_por_activo = monto_satelite / 4
    
    msg += f"🚀 **SATÉLITE (20%) - ${monto_satelite:,.0f}**\n"
    for ticker in shares_satelite.keys():
        precio = current_prices[ticker]
        cantidad = capital_por_activo / precio
        msg += f"• {ticker}: **{cantidad:.4f}** títulos (${capital_por_activo:,.0f})\n"
        
    msg += "\n⚠️ _Nota: Estos cálculos no incluyen comisiones del broker._"

    send_telegram(msg)
    print("✅ Plan enviado.")

if __name__ == "__main__":
    calcular_despliegue()
