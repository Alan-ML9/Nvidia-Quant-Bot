import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🔬 INICIANDO STRESS TEST: CRISIS COVID-19 (2020)...")

# 1. CONFIGURACIÓN DE LA MÁQUINA DEL TIEMPO
start_date = "2020-01-01"
end_date = "2020-12-31"

# Activos de tu estrategia
tickers = ["VOO", "NVDA", "MSFT", "AAPL", "BTC-USD"]
# Nota: La Reserva (20%) se asume como CASH (Variación 0%)

# 2. OBTENCIÓN DE DATOS HISTÓRICOS
try:
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    data = data.ffill().dropna()
    print("✅ Datos de 2020 descargados.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# 3. CONSTRUCCIÓN DE LOS PORTAFOLIOS

# A) PORTAFOLIO "SOLO VOO" (El Benchmark)
# Normalizamos a 1000 USD iniciales
voo_only = (data['VOO'] / data['VOO'].iloc[0]) * 1000

# B) TU PORTAFOLIO (50/20/30 Risk Parity Simplificado)
# Simularemos rebalanceo mensual para ser realistas
capital = 1000
historia_capital = []

# Pesos Objetivo
W_BETA = 0.50     # VOO
W_RESERVA = 0.20  # CASH (Sin rendimiento)
W_SAT = 0.30      # Satélite (Risk Parity entre Tech/Crypto)

# Subconjunto Satélite
sat_tickers = ["NVDA", "MSFT", "AAPL", "BTC-USD"]

# Iteramos día a día (Simplificación: Buy & Hold con pesos fijos iniciales para ver la caída bruta)
# Calculamos retornos diarios
retornos = data.pct_change().dropna()

# Pesos Risk Parity Iniciales (Basados en volatilidad de Enero 2020)
# Calculamos volatilidad de los primeros 20 días para asignar pesos iniciales
vol_inicial = retornos[sat_tickers].iloc[:20].std()
inv_vol = 1 / vol_inicial
pesos_sat_risk_parity = inv_vol / inv_vol.sum()

# Construimos el retorno compuesto del portafolio diario
# Retorno = (50% * RetornoVOO) + (20% * 0) + (30% * RetornoSatélitePonderado)

# Paso 1: Calcular retorno del bloque satélite
retorno_satelite = (retornos[sat_tickers] * pesos_sat_risk_parity).sum(axis=1)

# Paso 2: Calcular retorno total del sistema
# Nota: La reserva no genera retorno, por eso no se suma nada por el 20%
retorno_sistema = (retornos['VOO'] * W_BETA) + (retorno_satelite * W_SAT)

# Paso 3: Crear curva de capital
curva_sistema = (1 + retorno_sistema).cumprod() * 1000

# 4. CÁLCULO DE CAÍDA MÁXIMA (DRAWDOWN)
def get_max_drawdown(series):
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown.min() * 100

dd_voo = get_max_drawdown(voo_only)
dd_sistema = get_max_drawdown(curva_sistema)

# Retorno Final
ret_voo = ((voo_only.iloc[-1] / 1000) - 1) * 100
ret_sistema = ((curva_sistema.iloc[-1] / 1000) - 1) * 100

print("\n📊 --- RESULTADOS DEL DUELO ---")
print(f"💀 Caída Máxima (Max Drawdown) en Marzo 2020:")
print(f"   • S&P 500 (Solo VOO): {dd_voo:.2f}% (¡Doloroso!)")
print(f"   • TU SISTEMA (50/20/30): {dd_sistema:.2f}% (Amortiguado)")

print(f"\n💰 Retorno Final (Dic 2020):")
print(f"   • S&P 500: +{ret_voo:.2f}%")
print(f"   • TU SISTEMA: +{ret_sistema:.2f}%")

# 5. GRÁFICA VISUAL
plt.figure(figsize=(12, 6))
plt.plot(voo_only, label='Solo VOO (Benchmark)', color='grey', alpha=0.6)
plt.plot(curva_sistema, label='Tu Sistema (Risk Parity)', color='blue', linewidth=2)
plt.title('Simulacro de Crisis: COVID-19 Crash (2020)')
plt.ylabel('Valor del Portafolio ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
