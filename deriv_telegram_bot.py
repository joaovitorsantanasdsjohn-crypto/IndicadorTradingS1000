import os
import asyncio
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd
import requests
import websockets
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from dotenv import load_dotenv

load_dotenv()

# Configurações principais
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")  # App ID da Deriv
SYMBOLS = os.getenv("SYMBOLS", "frxEURUSD").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Parâmetros dos indicadores
EMA_SHORT = 8
EMA_MEDIUM = 21
EMA_LONG = 50
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2

CANDLE_INTERVAL = 300  # 5 minutos
CANDLE_HISTORY = 200

print("Starting Deriv -> Telegram bot")
print(f"Symbol(s): {SYMBOLS}")
print(f"Indicators: EMA({EMA_SHORT},{EMA_MEDIUM},{EMA_LONG}), RSI({RSI_PERIOD}), BB({BB_PERIOD}@{BB_STD})")

async def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram token ou chat_id não configurados.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Erro ao enviar mensagem para o Telegram:", e)

async def get_candles(symbol):
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
    async with websockets.connect(ws_url) as ws:
        request = {
            "ticks_history": symbol,
            "end": "latest",
            "count": CANDLE_HISTORY,
            "granularity": CANDLE_INTERVAL,
            "style": "candles"
        }
        await ws.send(json.dumps(request))
        response = await ws.recv()
        data = json.loads(response)
        candles = data.get("candles", [])
        return pd.DataFrame(candles)

def analyze_indicators(df):
    df["ema_short"] = EMAIndicator(df["close"], window=EMA_SHORT).ema_indicator()
    df["ema_medium"] = EMAIndicator(df["close"], window=EMA_MEDIUM).ema_indicator()
    df["ema_long"] = EMAIndicator(df["close"], window=EMA_LONG).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
    bb = BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    latest = df.iloc[-1]
    signal = None

    if (latest["ema_short"] > latest["ema_medium"] > latest["ema_long"]) and (latest["rsi"] > 55) and (latest["close"] > latest["bb_high"]):
        signal = "📈 <b>COMPRA</b> - Todos indicadores alinhados"

    elif (latest["ema_short"] < latest["ema_medium"] < latest["ema_long"]) and (latest["rsi"] < 45) and (latest["close"] < latest["bb_low"]):
        signal = "📉 <b>VENDA</b> - Todos indicadores alinhados"

    return signal

async def main():
    while True:
        for symbol in SYMBOLS:
            try:
                df = await get_candles(symbol)
                df["close"] = df["close"].astype(float)
                signal = analyze_indicators(df)
                if signal:
                    msg = f"{signal}\n💹 Ativo: {symbol}\n🕔 Intervalo: 5m"
                    print(msg)
                    await send_telegram_message(msg)
                else:
                    print(f"Sem sinal no momento ({symbol})")
            except Exception as e:
                print(f"Erro com {symbol}: {e}")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
