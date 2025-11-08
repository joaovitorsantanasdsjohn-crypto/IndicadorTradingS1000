import os
import json
import time
import threading
import asyncio
import websockets
import numpy as np
import pandas as pd
import telebot
from flask import Flask
from datetime import datetime, timedelta

# ==========================================
# CONFIGURAÇÕES
# ==========================================
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN)

PAIRS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxAUDUSD", "frxUSDCAD",
    "frxUSDCHF", "frxNZDUSD", "frxEURAUD", "frxEURGBP", "frxEURJPY",
    "frxGBPJPY", "frxAUDJPY", "frxCADJPY", "frxCHFJPY", "frxGBPCAD",
    "frxGBPAUD", "frxAUDNZD", "frxUSDNOK", "frxUSDSEK", "frxEURCAD"
]

RECONNECT_COOLDOWN = 600  # 10 minutos
last_reconnect_notification = {}

# ==========================================
# FUNÇÕES DE ANÁLISE (RSI + BOLLINGER)
# ==========================================
def rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(prices, period=20, std_dev=2):
    series = pd.Series(prices)
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

# ==========================================
# FUNÇÃO PRINCIPAL DE CONEXÃO
# ==========================================
async def connect_to_deriv(symbol):
    url = f"wss://ws.derivws.com/websockets/v3?app_id=1089&l=EN"
    headers = [("Authorization", f"Bearer {DERIV_TOKEN}")]
    candles = []

    while True:
        try:
            async with websockets.connect(url, extra_headers=headers) as ws:
                await ws.send(json.dumps({"ticks_history": symbol, "adjust_start_time": 1, "count": 1000, "end": "latest", "granularity": 60}))
                await ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))
                await bot.send_message(CHAT_ID, f"✅ Conexão ativa com WebSocket da Deriv para {symbol}")

                async for msg in ws:
                    data = json.loads(msg)
                    if "tick" in data:
                        tick_time = datetime.utcfromtimestamp(data["tick"]["epoch"])
                        tick_price = float(data["tick"]["quote"])
                        candles.append({"time": tick_time, "price": tick_price})

                        df = pd.DataFrame(candles[-100:])
                        if len(df) > 20:
                            rsi_value = rsi(df["price"].values)[-1]
                            upper, lower = bollinger_bands(df["price"].values)
                            last_price = df["price"].iloc[-1]

                            if rsi_value < 30 and last_price < lower.iloc[-1]:
                                await bot.send_message(CHAT_ID, f"📉 VENDA detectada em {symbol} — RSI={rsi_value:.2f}")
                            elif rsi_value > 70 and last_price > upper.iloc[-1]:
                                await bot.send_message(CHAT_ID, f"📈 COMPRA detectada em {symbol} — RSI={rsi_value:.2f}")

        except Exception as e:
            now = time.time()
            last_time = last_reconnect_notification.get(symbol, 0)
            if now - last_time >= RECONNECT_COOLDOWN:
                await bot.send_message(CHAT_ID, f"⚠️ Reconectando {symbol}: {e}")
                last_reconnect_notification[symbol] = now
            await asyncio.sleep(5)

# ==========================================
# LOOP PRINCIPAL DO BOT
# ==========================================
async def main():
    await bot.send_message(CHAT_ID, "✅ Bot iniciado com sucesso no Render e pronto para análise!")
    tasks = [asyncio.create_task(connect_to_deriv(symbol)) for symbol in PAIRS]
    await asyncio.gather(*tasks)

# ==========================================
# SERVIDOR FLASK (para manter Render ativo)
# ==========================================
@app.route("/")
def home():
    return "IndicadorTradingS1000 rodando ✅"

def start_flask():
    app.run(host="0.0.0.0", port=10000)

# ==========================================
# INÍCIO
# ==========================================
if __name__ == "__main__":
    threading.Thread(target=start_flask, daemon=True).start()
    asyncio.run(main())
