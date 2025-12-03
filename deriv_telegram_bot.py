# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (AJUSTADA) — Opção A — COMPLETO
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações principais ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
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

# ---------------- Parâmetros (mais assertivo) ----------------
BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

# ---------------- Estado ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    if level == "info": logger.info(msg)
    elif level == "warning": logger.warning(msg)
    elif level == "error": logger.error(msg)
    print(msg, flush=True)

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()

    if symbol:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("⚠️ Telegram não configurado.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
    except Exception as e:
        log(f"[TG] Erro: {e}", "error")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    try:
        macd = MACD(df["close"], 26, 12, 9)
        df["macd_diff"] = macd.macd_diff()
    except:
        df["macd_diff"] = pd.NA

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df

# ---------------- Lógica — CORRIGIDA + FORÇA DO SINAL ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < 3:
            return None

        now = df.iloc[-1]
        prev = df.iloc[-2]

        epoch = int(now["epoch"])
        close = float(now["close"])

        # Evitar duplicidade
        if last_signal_candle.get(symbol) == epoch:
            return None

        ema20_now, ema50_now = now["ema20"], now["ema50"]
        ema20_prev, ema50_prev = prev["ema20"], prev["ema50"]
        rsi_now = now["rsi"]
        bb_upper, bb_lower = now["bb_upper"], now["bb_lower"]
        macd_diff = now.get("macd_diff")

        if any(pd.isna([ema20_now, ema50_now, ema20_prev, ema50_prev, rsi_now])):
            return None

        # Tendência
        cruzou_up = ema20_prev <= ema50_prev and ema20_now > ema50_now
        cruzou_down = ema20_prev >= ema50_prev and ema20_now < ema50_now
        tendencia_up = ema20_now > ema50_now
        tendencia_down = ema20_now < ema50_now

        # Bollinger
        range_bb = bb_upper - bb_lower
        lim_inf = bb_lower + range_bb * BB_PROXIMITY_PCT
        lim_sup = bb_upper - range_bb * BB_PROXIMITY_PCT

        perto_lower = close <= lim_inf
        perto_upper = close >= lim_sup

        # RSI
        buy_rsi_ok = rsi_now <= RSI_BUY_MAX
        sell_rsi_ok = rsi_now >= RSI_SELL_MIN

        # MACD
        macd_buy_ok = macd_diff > -MACD_TOLERANCE if macd_diff is not None else True
        macd_sell_ok = macd_diff < MACD_TOLERANCE if macd_diff is not None else True

        cond_buy = (cruzou_up or tendencia_up) and perto_lower and buy_rsi_ok and macd_buy_ok
        cond_sell = (cruzou_down or tendencia_down) and perto_upper and sell_rsi_ok and macd_sell_ok

        # ----------- CÁLCULO DA FORÇA DO SINAL -----------
        def calc_forca():
            forca = 0

            # Bollinger força
            if cond_buy:
                dist = max(0, min(1, (lim_inf - close) / range_bb))
                forca += dist * 40
            elif cond_sell:
                dist = max(0, min(1, (close - lim_sup) / range_bb))
                forca += dist * 40

            # RSI força
            if cond_buy:
                rsi_strength = max(0, min(1, (52 - rsi_now) / 20))
                forca += rsi_strength * 30
            elif cond_sell:
                rsi_strength = max(0, min(1, (rsi_now - 48) / 20))
                forca += rsi_strength * 30

            # MACD força
            if macd_diff is not None and not pd.isna(macd_diff):
                macd_strength = min(1, abs(macd_diff) / MACD_TOLERANCE)
                forca += macd_strength * 30

            return int(forca)

        if cond_buy:
            last_signal_candle[symbol] = epoch
            return {"tipo": "COMPRA", "forca": calc_forca()}

        if cond_sell:
            last_signal_candle[symbol] = epoch
            return {"tipo": "VENDA", "forca": calc_forca()}

        return None

    except Exception as e:
        log(f"[{symbol}] Erro sinal: {e}", "error")
        return None

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    df.tail(200).to_csv(DATA_DIR / f"candles_{symbol}.csv", index=False)

# ---------------- Monitor WebSocket ----------------
async def monitor_symbol(symbol: str):
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await ws.recv()

                await ws.send(json.dumps({
                    "ticks_history": symbol,
                    "count": 200,
                    "end": "latest",
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }))

                data = json.loads(await ws.recv())
                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)

                await ws.send(json.dumps({
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "subscribe": 1
                }))

                while True:
                    msg = json.loads(await ws.recv())
                    candle = msg.get("candle") or msg.get("ohlc") or (msg.get("candles", [None])[-1])
                    if not candle:
                        continue

                    epoch = int(candle["epoch"])

                    if df.empty or int(df.iloc[-1]["epoch"]) != epoch:
                        df.loc[len(df)] = {
                            "epoch": epoch,
                            "open": candle["open"],
                            "high": candle["high"],
                            "low": candle["low"],
                            "close": candle["close"],
                            "volume": candle.get("volume", 0)
                        }

                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        sinal = gerar_sinal(df, symbol)

                        if sinal:
                            tipo = sinal["tipo"]
                            forca = sinal["forca"]

                            arrow = "🟢" if tipo == "COMPRA" else "🔴"
                            price = float(candle["close"])

                            entrada_br = datetime.utcfromtimestamp(epoch) - timedelta(hours=3)
                            entrada_str = entrada_br.strftime("%Y-%m-%d %H:%M:%S")

                            msg_final = (
                                f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                f"• Par: {symbol.replace('frx','')}\n"
                                f"• Direção: {arrow} *{tipo}*\n"
                                f"• Força do sinal: *{forca}%*\n"
                                f"• Preço: {price:.5f}\n"
                                f"• Horário de entrada (Brasília): {entrada_str}"
                            )

                            send_telegram(msg_final, symbol)

        except Exception as e:
            log(f"[WS {symbol}] erro: {e}")
            await asyncio.sleep(random.uniform(2, 5))

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Lógica B (com força do sinal)"

# ---------------- Execução ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run("0.0.0.0", port, debug=False, use_reloader=False)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("Bot iniciado — Lógica B + Força do Sinal")

    await asyncio.gather(*(monitor_symbol(s) for s in SYMBOLS))

if __name__ == "__main__":
    asyncio.run(main())
