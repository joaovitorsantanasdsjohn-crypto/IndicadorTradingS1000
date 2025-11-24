# ===============================================================
# deriv_telegram_bot.py — versão com NOVA LÓGICA DE SINAIS
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands

import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxGBPAUD", "frxGBPCAD", "frxAUDNZD",
    "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# Controle
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
sent_download_message = {s: False for s in SYMBOLS}
last_notify_time = {}

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

def log(msg: str):
    logger.info(msg)
    print(msg, flush=True)


# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()
    if symbol:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            return
        last_notify_time[symbol] = now

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)
    except:
        pass


# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame):
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    return df


# ---------------- NOVA LÓGICA DO USUÁRIO ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """ NOVO SETUP:
        COMPRA:
            - EMA20 > EMA50
            - close < banda inferior
            - RSI < 50
        VENDA:
            - EMA20 < EMA50
            - close > banda superior
            - RSI > 50
    """
    ultima = df.iloc[-1]

    epoch = int(ultima["epoch"])
    close = float(ultima["close"])
    ema20 = ultima["ema20"]
    ema50 = ultima["ema50"]
    rsi = ultima["rsi"]
    bb_lower = ultima["bb_lower"]
    bb_upper = ultima["bb_upper"]

    log(f"[{symbol}] close={close:.5f} EMA20={ema20:.5f} EMA50={ema50:.5f} RSI={rsi:.1f} BB_low={bb_lower:.5f} BB_up={bb_upper:.5f}")

    if any(pd.isna([ema20, ema50, rsi, bb_lower, bb_upper])):
        return None

    # ---------------- COMPRA ----------------
    cond_buy = (
        ema20 > ema50 and
        close < bb_lower and
        rsi < 50
    )

    # ---------------- VENDA ----------------
    cond_sell = (
        ema20 < ema50 and
        close > bb_upper and
        rsi > 50
    )

    atual = last_signal_state.get(symbol)

    if cond_buy:
        if atual == "COMPRA" and last_signal_candle[symbol] == epoch:
            return None
        last_signal_state[symbol] = "COMPRA"
        last_signal_candle[symbol] = epoch
        return "COMPRA"

    if cond_sell:
        if atual == "VENDA" and last_signal_candle[symbol] == epoch:
            return None
        last_signal_state[symbol] = "VENDA"
        last_signal_candle[symbol] = epoch
        return "VENDA"

    # limpar estado
    last_signal_state[symbol] = None
    last_signal_candle[symbol] = None

    return None


# ---------------- Monitor Símbolo ----------------
async def monitor_symbol(symbol: str):
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:

                send_telegram(f"🔌 [{symbol}] Conectado ao WebSocket.", symbol)
                log(f"[{symbol}] WebSocket conectado.")

                # Autorizar
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth = json.loads(await ws.recv())
                if "authorize" not in auth:
                    raise Exception(f"Falha na autorização: {auth}")

                # Histórico 200 velas
                req = {
                    "ticks_history": symbol,
                    "count": 200,
                    "end": "latest",
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req))
                data = json.loads(await ws.recv())

                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)

                if not sent_download_message[symbol]:
                    send_telegram(f"📥 [{symbol}] Download de velas completo ({len(df)} candles).", symbol)
                    sent_download_message[symbol] = True

                # Assinar realtime
                sub = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub))

                last_candle = time.time()

                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    data = json.loads(raw)

                    candle = data.get("candle")
                    if not candle:
                        continue

                    if df.iloc[-1]["epoch"] != candle["epoch"]:
                        df.loc[len(df)] = candle
                        df = calcular_indicadores(df)

                        sinal = gerar_sinal(df, symbol)
                        if sinal:
                            arrow = "🟢" if sinal == "COMPRA" else "🔴"
                            msg = (
                                f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                f"• Par: {symbol.replace('frx','')}\n"
                                f"• Direção: {arrow} *{sinal}*\n"
                                f"• Preço: {float(candle['close']):.5f}\n"
                                f"• Horário: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
                            )
                            send_telegram(msg, symbol)

        except Exception as e:
            log(f"[{symbol}] ERRO: {e}")
            await asyncio.sleep(random.uniform(2, 6))
            continue


# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Nova Lógica (EMA+BB+RSI) ✔"


# ---------------- Execução ----------------
async def main():
    threading.Thread(target=lambda: app.run(
        host="0.0.0.0", port=int(os.getenv("PORT", 10000))), daemon=True).start()

    send_telegram("✅ Bot reiniciado com NOVA LÓGICA (EMA20/50 + BB20 + RSI14).")

    tasks = [monitor_symbol(sym) for sym in SYMBOLS]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
