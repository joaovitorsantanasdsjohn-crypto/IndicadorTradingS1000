import websocket
import json
import pandas as pd
import numpy as np
from telegram import Bot
from ml_model import SignalFilter
from flask import Flask
import threading
from datetime import datetime
import time

# ========================
# CONFIGURAÇÕES
# ========================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"
DERIV_WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

ativos = [
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD",
    "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP", "frxAUDJPY",
    "frxEURCHF", "frxUSDCHF", "frxGBPCHF", "frxAUDNZD", "frxNZDJPY"
]

bot = Bot(token=TELEGRAM_TOKEN)
ml_filter = SignalFilter()

# Candles por ativo
candles_por_ativo = {ativo: [] for ativo in ativos}
ticks_por_ativo = {ativo: [] for ativo in ativos}
current_candle_time = {ativo: None for ativo in ativos}

# ========================
# FUNÇÕES
# ========================
def send_telegram(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
        print(f"[Telegram] {message}")
    except Exception as e:
        print(f"[Erro Telegram] {e}")

def calculate_indicators(df):
    import ta
    df['EMA_short'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
    df['EMA_medium'] = ta.trend.EMAIndicator(df['close'], window=13).ema_indicator()
    df['EMA_long'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df

def generate_signal(df, ativo):
    last = df.iloc[-1]
    features = [last['EMA_short'], last['EMA_medium'], last['EMA_long'], last['RSI'], last['BB_upper'], last['BB_lower']]
    prob = ml_filter.predict(features)

    # COMPRA
    if last['EMA_short'] > last['EMA_medium'] > last['EMA_long'] and last['RSI'] < 70 and last['close'] > last['BB_lower'] and prob > 0.6:
        send_telegram(f"📈 {ativo}: Sinal de COMPRA! Probabilidade {prob:.2f}")

    # VENDA
    elif last['EMA_short'] < last['EMA_medium'] < last['EMA_long'] and last['RSI'] > 30 and last['close'] < last['BB_upper'] and prob > 0.6:
        send_telegram(f"📉 {ativo}: Sinal de VENDA! Probabilidade {prob:.2f}")

# ========================
# WEBSOCKET
# ========================
def on_message(ws, message):
    data = json.loads(message)
    
    if 'tick' not in data:
        return
    
    tick = data['tick']
    ativo = tick['symbol']
    if ativo not in ativos:
        return

    tick_price = tick['quote']
    tick_time = int(tick['epoch'])

    if current_candle_time[ativo] is None:
        current_candle_time[ativo] = tick_time - (tick_time % 300)

    ticks_por_ativo[ativo].append(tick_price)

    # Fecha candle de 5 minutos
    if tick_time >= current_candle_time[ativo] + 300:
        candle_ticks = ticks_por_ativo[ativo]
        candle = {
            'time': current_candle_time[ativo],
            'open': candle_ticks[0],
            'high': max(candle_ticks),
            'low': min(candle_ticks),
            'close': candle_ticks[-1]
        }
        candles_por_ativo[ativo].append(candle)
        if len(candles_por_ativo[ativo]) > 100:
            candles_por_ativo[ativo].pop(0)

        ticks_por_ativo[ativo] = []
        current_candle_time[ativo] += 300

        df = pd.DataFrame(candles_por_ativo[ativo])
        df = calculate_indicators(df)
        generate_signal(df, ativo)

def on_error(ws, error):
    print(f"[Erro WS] {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"[WS fechado] code={close_status_code} msg={close_msg}")
    time.sleep(5)
    start_ws()  # reconecta automaticamente

def on_open(ws):
    print("[WS aberto]")
    for ativo in ativos:
        subscribe_msg = {
            "ticks": ativo,
            "subscribe": 1
        }
        ws.send(json.dumps(subscribe_msg))

def start_ws():
    ws = websocket.WebSocketApp(
        DERIV_WEBSOCKET_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# ========================
# FLASK PARA UPTIME
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
    # Flask em thread separada
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # WebSocket principal
    start_ws()
