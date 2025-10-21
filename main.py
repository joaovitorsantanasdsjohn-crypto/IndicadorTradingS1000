import websocket
import json
import pandas as pd
from telegram import Bot
from ml_model import SignalFilter
from flask import Flask
import threading
import time

# ========================
# CONFIGURAÇÕES
# ========================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"
DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

# 15 pares Forex
ativos = [
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD",
    "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP", "frxAUDJPY",
    "frxEURCHF", "frxUSDCHF", "frxGBPCHF", "frxAUDNZD", "frxNZDJPY"
]

bot = Bot(token=TELEGRAM_TOKEN)
ml_filter = SignalFilter()

# Candles e ticks
candles = {ativo: [] for ativo in ativos}
ticks = {ativo: [] for ativo in ativos}
candle_time = {ativo: None for ativo in ativos}

# ========================
# FUNÇÕES
# ========================
def send_telegram(msg):
    bot.send_message(chat_id=CHAT_ID, text=msg)

def calculate_indicators(df):
    import ta
    df['EMA_short'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
    df['EMA_medium'] = ta.trend.EMAIndicator(df['close'], window=13).ema_indicator()
    df['EMA_long'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df

def generate_signal(df, ativo):
    last = df.iloc[-1]
    features = [last['EMA_short'], last['EMA_medium'], last['EMA_long'], last['RSI'], last['BB_upper'], last['BB_lower']]
    prob = ml_filter.predict(features)
    # Sinal COMPRA
    if last['EMA_short'] > last['EMA_medium'] > last['EMA_long'] and last['RSI'] < 70 and last['close'] > last['BB_lower'] and prob > 0.6:
        send_telegram(f"📈 {ativo}: COMPRA! Prob {prob:.2f}")
    # Sinal VENDA
    elif last['EMA_short'] < last['EMA_medium'] < last['EMA_long'] and last['RSI'] > 30 and last['close'] < last['BB_upper'] and prob > 0.6:
        send_telegram(f"📉 {ativo}: VENDA! Prob {prob:.2f}")

# ========================
# WEBSOCKET DERIV
# ========================
def on_message(ws, msg):
    data = json.loads(msg)
    if 'tick' not in data: return
    tick = data['tick']
    ativo = tick['symbol']
    if ativo not in ativos: return

    price = tick['quote']
    ts = int(tick['epoch'])

    # Inicializa candle
    if candle_time[ativo] is None:
        candle_time[ativo] = ts - (ts % 300)  # múltiplo de 5 minutos

    ticks[ativo].append(price)

    # Fecha candle a cada 5 minutos
    if ts >= candle_time[ativo] + 300:
        tick_buffer = ticks[ativo]
        candle = {
            'time': candle_time[ativo],
            'open': tick_buffer[0],
            'high': max(tick_buffer),
            'low': min(tick_buffer),
            'close': tick_buffer[-1]
        }
        candles[ativo].append(candle)
        if len(candles[ativo]) > 100: candles[ativo].pop(0)

        ticks[ativo] = []
        candle_time[ativo] += 300

        df = pd.DataFrame(candles[ativo])
        df = calculate_indicators(df)
        generate_signal(df, ativo)

def on_error(ws, err):
    print("Erro WS:", err)

def on_close(ws, code, msg):
    print(f"WS fechado: {code}, {msg}")

def on_open(ws):
    print("WS Deriv aberto")
    for ativo in ativos:
        ws.send(json.dumps({"ticks": ativo, "subscribe": 1}))

def run_ws_forever():
    while True:
        try:
            ws = websocket.WebSocketApp(DERIV_WS_URL, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
            ws.run_forever()
        except Exception as e:
            print("WS caiu, reconectando em 5s...", e)
            time.sleep(5)

# ========================
# FLASK APENAS PARA UPTIME
# ========================
app = Flask(__name__)
@app.route('/')
def home():
    return "IndicadorTradingS1000 ativo!"

def run_flask():
    app.run(host="0.0.0.0", port=5000)

# ========================
# EXECUÇÃO
# ========================
if __name__ == "__main__":
    # Flask em thread secundária
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # WebSocket como processo principal
    run_ws_forever()
