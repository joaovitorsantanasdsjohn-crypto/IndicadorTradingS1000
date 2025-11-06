import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import threading
from flask import Flask
import time

load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CANDLE_INTERVAL = 5  # minutos
CANDLE_COUNT = 200   # número de velas armazenadas

SYMBOLS = os.getenv("SYMBOLS", "").split(",")
if not SYMBOLS or SYMBOLS == [""]:
    SYMBOLS = [
        "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
        "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
        "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
        "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
    ]

# ---------------- Telegram ----------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message[:4000], "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}")

# ---------------- Indicadores ----------------
def calcular_indicadores(df):
    if len(df) < 20:
        return df

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    return df

# ---------------- Sinal ----------------
def gerar_sinal(df):
    ultima = df.iloc[-1]

    sinal_bollinger_compra = ultima["close"] <= ultima["bb_lower"]
    sinal_bollinger_venda = ultima["close"] >= ultima["bb_upper"]

    sinal_rsi_compra = ultima["rsi"] < 30
    sinal_rsi_venda = ultima["rsi"] > 70

    if sinal_bollinger_compra and sinal_rsi_compra:
        return "COMPRA"
    elif sinal_bollinger_venda and sinal_rsi_venda:
        return "VENDA"
    else:
        return None

# ---------------- Histórico ----------------
def salvar_historico(symbol, df):
    os.makedirs("historico", exist_ok=True)
    path = f"historico/{symbol}.csv"
    df.tail(CANDLE_COUNT).to_csv(path, index=False)

# ---------------- WebSocket ----------------
async def monitor_symbol(symbol):
    url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

    while True:
        try:
            async with websockets.connect(url, ping_interval=30) as ws:
                send_telegram(f"✅ Conexão ativa com WebSocket da Deriv para {symbol}")
                print(f"🚀 WebSocket conectado para {symbol}")

                req = {
                    "ticks_history": symbol,
                    "count": CANDLE_COUNT,
                    "end": "latest",
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req))

                response = await ws.recv()
                data = json.loads(response)

                if "history" not in data:
                    send_telegram(f"❌ Erro (history) do WS para {symbol}: {data.get('error', {}).get('message', 'sem detalhes')}")
                    await asyncio.sleep(10)
                    continue

                candles = data["history"]["candles"]
                df = pd.DataFrame(candles)
                df["open"] = df["open"].astype(float)
                df["close"] = df["close"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["epoch"] = pd.to_datetime(df["epoch"], unit="s")

                df = calcular_indicadores(df)
                salvar_historico(symbol, df)

                ultima = df.iloc[-1]
                send_telegram(
                    f"📡 [{symbol}] Candles recebidos e armazenados.\n"
                    f"Último fechamento: {ultima['close']:.5f}\nRSI: {ultima['rsi']:.2f}"
                )

                sinal = gerar_sinal(df)
                if sinal:
                    send_telegram(f"💹 Sinal {sinal} detectado para {symbol} (vela {CANDLE_INTERVAL} min)")

                # Aguarda até o próximo candle
                await asyncio.sleep(CANDLE_INTERVAL * 60)

        except Exception as e:
            send_telegram(f"⚠️ Erro ou desconexão no WebSocket para {symbol}: {e}")
            await asyncio.sleep(10)

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- Principal ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")

    for s in SYMBOLS:
        send_telegram(f"📊 Iniciando monitoramento de {s}")
        await asyncio.sleep(3)  # atraso para evitar flood de conexões

    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
