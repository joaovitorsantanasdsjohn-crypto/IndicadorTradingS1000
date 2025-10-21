import websocket
import json
import pandas as pd
import numpy as np
from telegram import Bot
from ml_model import SignalFilter
from flask import Flask
import threading

# ========================
# CONFIGURAÇÕES
# ========================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"
OLYMP_WEBSOCKET_URL = "wss://ws.olymptrade.com/otp?cid_ver=1&cid_app=web%40OlympTrade%402025.4.27904%4027904&cid_device=%40%40desktop&cid_os=windows%4010"

# Pares principais
ativos = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY",
    "EURCHF", "USDCHF", "GBPCHF", "AUDNZD", "NZDJPY"
]

# Inicializa bot Telegram
bot = Bot(token=TELEGRAM_TOKEN)
ml_filter = SignalFilter()

# Armazena candles por ativo
candles_por_ativo = {ativo: [] for ativo in ativos}

# ========================
# FUNÇÕES
# ========================
def send_telegram(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

def calculate_indicators(df):
    import ta

    # Médias móveis exponenciais
    df['EMA_short'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
    df['EMA_medium'] = ta.trend.EMAIndicator(df['close'], window=13).ema_indicator()
    df['EMA_long'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Bandas de Bollinger
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()

    return df

def generate_signal(df, ativo):
    last = df.iloc[-1]
    features = [last['EMA_short'], last['EMA_medium'], last['EMA_long'], last['RSI'], last['BB_upper'], last['BB_lower']]
    prob = ml_filter.predict(features)

    # Sinal COMPRA
    if last['EMA_short'] > last['EMA_medium'] > last['EMA_long'] and last['RSI'] < 70 and last['close'] > last['BB_lower'] and prob > 0.6:
        send_telegram(f"📈 {ativo}: Sinal de COMPRA! Probabilidade {prob:.2f}")

    # Sinal VENDA
    elif last['EMA_short'] < last['EMA_medium'] < last['EMA_long'] and last['RSI'] > 30 and last['close'] < last['BB_upper'] and prob > 0.6:
        send_telegram(f"📉 {ativo}: Sinal de VENDA! Probabilidade {prob:.2f}")

# ========================
# WEBSOCKET
# ========================
def on_message(ws, message):
    data = json.loads(message)
    ativo = data.get('symbol')
    if ativo not in ativos:
        return

    candle = {
        'time': data['time'],
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close']
    }
    candles_por_ativo[ativo].append(candle)
    if len(candles_por_ativo[ativo]) > 100:
        candles_por_ativo[ativo].pop(0)

    df = pd.DataFrame(candles_por_ativo[ativo])
    df = calculate_indicators(df)
    generate_signal(df, ativo)

def on_error(ws, error):
    print("Erro:", error)

def on_close(ws, close_status_code, close_msg):
    print("Conexão fechada")

def on_open(ws):
    print("Conexão WebSocket aberta")
    for ativo in ativos:
        subscribe_msg = {
            "type": "subscribe",
            "symbol": ativo,
            "interval": 5
        }
        ws.send(json.dumps(subscribe_msg))

def run_ws():
    ws = websocket.WebSocketApp(
        OLYMP_WEBSOCKET_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# ========================
# FLASK PARA UPTIME ROBOT
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
    # Rodar WebSocket em thread separada
    ws_thread = threading.Thread(target=run_ws)
    ws_thread.start()

    # Rodar Flask para Uptime Robot
    run_flask()
