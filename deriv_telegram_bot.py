import os
import asyncio
import json
import time
import requests
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timezone
from dotenv import load_dotenv
import websockets
from flask import Flask
import threading

load_dotenv()

# ==============================
# CONFIGURAÇÕES
# ==============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOL = "frxEURUSD"
TIMEFRAME = 300  # 5 minutos (em segundos)
CANDLES_QTD = 100

DERIV_API = "wss://ws.derivws.com/websockets/v3?app_id=1089"
URL_TELEGRAM = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# ==============================
# FUNÇÃO PARA ENVIAR MENSAGEM TELEGRAM
# ==============================
def send_telegram_message(text):
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        requests.post(URL_TELEGRAM, data=payload)
    except Exception as e:
        print("Erro ao enviar mensagem Telegram:", e)

# ==============================
# ANÁLISE DE INDICADORES
# ==============================
def analisar_indicadores(df):
    df["EMA_curta"] = EMAIndicator(df["close"], window=5).ema_indicator()
    df["EMA_media"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["EMA_longa"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    ema_c, ema_m, ema_l = df.iloc[-1][["EMA_curta", "EMA_media", "EMA_longa"]]
    rsi = df.iloc[-1]["RSI"]
    close = df.iloc[-1]["close"]
    bb_high, bb_low = df.iloc[-1][["BB_high", "BB_low"]]

    if ema_c > ema_m > ema_l and rsi < 70 and close > bb_low:
        return "🟢 <b>SINAL DE COMPRA</b> - EUR/USD"
    elif ema_c < ema_m < ema_l and rsi > 30 and close < bb_high:
        return "🔴 <b>SINAL DE VENDA</b> - EUR/USD"
    return None

# ==============================
# LOOP PRINCIPAL DO BOT
# ==============================
async def deriv_loop():
    async with websockets.connect(DERIV_API) as ws:
        await ws.send(json.dumps({
            "ticks_history": SYMBOL,
            "adjust_start_time": 1,
            "count": CANDLES_QTD,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": TIMEFRAME
        }))

        print("✅ Conectado ao WebSocket da Deriv!")
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                if "candles" in data:
                    df = pd.DataFrame(data["candles"])
                    df["close"] = df["close"].astype(float)
                    df["epoch"] = pd.to_datetime(df["epoch"], unit="s")

                    sinal = analisar_indicadores(df)
                    if sinal:
                        send_telegram_message(sinal)
                        print(datetime.now().strftime("%H:%M:%S"), "->", sinal)
                    else:
                        print(datetime.now().strftime("%H:%M:%S"), "-> Nenhum sinal.")
                    
                    await asyncio.sleep(TIMEFRAME)
            except Exception as e:
                print("Erro no loop:", e)
                await asyncio.sleep(5)

# ==============================
# INICIAR FLASK (para Render gratuito)
# ==============================
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 Bot de Sinais Deriv Online ✅"

def start_bot():
    asyncio.run(deriv_loop())

if __name__ == "__main__":
    threading.Thread(target=start_bot).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
