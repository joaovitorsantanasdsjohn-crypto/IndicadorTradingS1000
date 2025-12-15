# Indicador Trading S1000

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
from pathlib import Path
import time
import random
import logging
import traceback
from collections import deque
import html
from flask import Flask
import threading
from typing import Optional

# ---------------- ML availability ----------------
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60
SIGNAL_ADVANCE_SECONDS = 3

# lista de símbolos (mantive a sua lista original)
SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Parâmetros (ajustáveis via env) ----------------
TRADE_MODE = os.getenv("TRADE_MODE", "FTT")
BB_PROXIMITY_PCT = float(os.getenv("BB_PROXIMITY_PCT", "0.20"))
RSI_BUY_MAX = int(os.getenv("RSI_BUY_MAX", "52"))
RSI_SELL_MIN = int(os.getenv("RSI_SELL_MIN", "48"))
MACD_TOLERANCE = float(os.getenv("MACD_TOLERANCE", "0.002"))
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_MID = int(os.getenv("EMA_MID", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
MIN_SECONDS_BETWEEN_SIGNALS = int(os.getenv("MIN_SECONDS_BETWEEN_SIGNALS", "0"))
MIN_CANDLES_BETWEEN_SIGNALS = int(os.getenv("MIN_CANDLES_BETWEEN_SIGNALS", "2"))
REL_EMA_SEP_PCT = float(os.getenv("REL_EMA_SEP_PCT", "5e-06"))
FORCE_MIN = int(os.getenv("FORCE_MIN", "35"))
MICRO_FORCE_ALLOW_THRESHOLD = int(os.getenv("MICRO_FORCE_ALLOW_THRESHOLD", "25"))
ML_ENABLED = SKLEARN_AVAILABLE and os.getenv("ENABLE_ML", "1") != "0"
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "40"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "4"))
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "200"))
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "2000"))
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_RETRAIN_INTERVAL = int(os.getenv("ML_RETRAIN_INTERVAL", "50"))
MIN_SIGNALS_PER_HOUR = int(os.getenv("MIN_SIGNALS_PER_HOUR", "5"))
FALLBACK_WINDOW_SEC = int(os.getenv("FALLBACK_WINDOW_SEC", "3600"))
FALLBACK_DURATION_SECONDS = int(os.getenv("FALLBACK_DURATION_SECONDS", str(15*60)))
INITIAL_HISTORY_COUNT = int(os.getenv("INITIAL_HISTORY_COUNT", "1200"))
MAX_CANDLES = int(os.getenv("MAX_CANDLES", "300"))
WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "60"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "30"))
RECV_TIMEOUT = int(os.getenv("RECV_TIMEOUT", "1200"))

# ---------------- Estado ----------------
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}
ml_models = {}
ml_model_ready = {}
sent_timestamps = deque()
fallback_active_until = 0.0
ml_trained_samples = {s: 0 for s in SYMBOLS}
last_epoch_seen = {s: None for s in SYMBOLS}

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    if level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)

# ---------------- telegram utils ----------------
def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False) -> bool:
    now_ts = time.time()
    if symbol and not bypass_throttle:
        last = last_notify_time.get(symbol, 0)
        if now_ts - last < 3:
            return False
        last_notify_time[symbol] = now_ts
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAMTOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def convert_utc_to_brasilia(dt_utc: datetime) -> str:
    brasilia = dt_utc - timedelta(hours=3)
    return brasilia.strftime("%H:%M:%S") + " BRT"

# ---------------- Geração de sinais (ajustada) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if df is None or len(df) < 8:
            return None

        now = df.iloc[-1]
        epoch = int(now["epoch"])
        prev_epoch = last_epoch_seen.get(symbol)
        last_epoch_seen[symbol] = epoch

        if prev_epoch is None or epoch == prev_epoch:
            return None

        candle_id = epoch - (epoch % GRANULARITY_SECONDS)
        if last_signal_candle.get(symbol) == candle_id:
            return None

        if MIN_SECONDS_BETWEEN_SIGNALS > 0 and time.time() - last_signal_time.get(symbol, 0) < MIN_SECONDS_BETWEEN_SIGNALS:
            return None

        ema_fast = df[f"ema{EMA_FAST}"].iloc[-1] if f"ema{EMA_FAST}" in df.columns else df["close"].iloc[-1]
        ema_mid = df[f"ema{EMA_MID}"].iloc[-1] if f"ema{EMA_MID}" in df.columns else df["close"].iloc[-1]
        ema_slow = df[f"ema{EMA_SLOW}"].iloc[-1] if f"ema{EMA_SLOW}" in df.columns else df["close"].iloc[-1]
        triple_up = ema_fast > ema_mid > ema_slow
        triple_down = ema_fast < ema_mid < ema_slow

        close = float(now["close"])
        rsi_now = float(now["rsi"]) if not pd.isna(now["rsi"]) else 50.0
        macd_diff = now.get("macd_diff")

        macd_buy_ok = True if macd_diff is None or pd.isna(macd_diff) else macd_diff > -MACD_TOLERANCE
        macd_sell_ok = True if macd_diff is None or pd.isna(macd_diff) else macd_diff < MACD_TOLERANCE

        buy_ok = triple_up and rsi_now <= RSI_BUY_MAX and macd_buy_ok
        sell_ok = triple_down and rsi_now >= RSI_SELL_MIN and macd_sell_ok

        if not buy_ok and not sell_ok:
            return None

        tipo = "COMPRA" if buy_ok else "VENDA"

        next_epoch = epoch + GRANULARITY_SECONDS
        send_ts = next_epoch - SIGNAL_ADVANCE_SECONDS

        def delayed_send(send_ts, symbol, tipo, next_epoch):
            nowt = time.time()
            wait = send_ts - nowt
            if wait > 0:
                time.sleep(wait)
            entry_dt_utc = datetime.fromtimestamp(next_epoch, tz=timezone.utc)
            msg = (
                f"💱 <b>{symbol.replace('frx','')}</b> ({TRADE_MODE})\n\n"
                f"📈 DIREÇÃO: <b>{tipo}</b>\n"
                f"⏱ ENTRADA: <b>{convert_utc_to_brasilia(entry_dt_utc)}</b>"
            )
            send_telegram(msg, symbol)
            last_signal_candle[symbol] = candle_id
            last_signal_time[symbol] = time.time()

        threading.Thread(target=delayed_send, args=(send_ts, symbol, tipo, next_epoch), daemon=True).start()
        return {"tipo": tipo, "candle_id": candle_id}
    except Exception:
        return None

# ---------------- Monitor WebSocket (robusto) ----------------
async def monitor_symbol(symbol: str):
    columns = ["epoch", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(columns=columns)

    csv_path = DATA_DIR / f"candles_{symbol}.csv"
    if csv_path.exists():
        try:
            tmp = pd.read_csv(csv_path)
            if not tmp.empty:
                tmp = tmp.loc[:, tmp.columns.intersection(columns)]
                df = calcular_indicadores(pd.DataFrame(tmp, columns=columns))
        except:
            pass

    connect_attempt = 0

    while True:
        try:
            connect_attempt += 1
            if connect_attempt > 1:
                await asyncio.sleep(min(120, (2**connect_attempt)) + random.random())

            async with websockets.connect(WS_URL, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT) as ws:
                if DERIV_TOKEN:
                    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=10)
                    except:
                        pass

                await ws.send(json.dumps({
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": INITIAL_HISTORY_COUNT,
                    "end": "latest",
                    "style": "candles",
                    "granularity": GRANULARITY_SECONDS
                }))

                await ws.send(json.dumps({
                    "ticks_history": symbol,
                    "subscribe": 1,
                    "style": "candles",
                    "granularity": GRANULARITY_SECONDS
                }))

                while True:
                    raw = await ws.recv()
                    try:
                        msg = json.loads(raw)
                    except:
                        continue

                    if "candles" in msg and isinstance(msg["candles"], list):
                        hist = pd.DataFrame(msg["candles"])
                        df = calcular_indicadores(hist)
                        save_last_candles(df, symbol)
                        continue

                    candle_data = None
                    if "candle" in msg:
                        candle_data = msg["candle"]
                    elif "ohlc" in msg:
                        candle_data = msg["ohlc"]
                    elif "candles" in msg and isinstance(msg["candles"], list) and msg["candles"]:
                        candle_data = msg["candles"][-1]

                    if candle_data:
                        try:
                            epoch = int(candle_data.get("epoch"))
                        except:
                            continue
                        if epoch % GRANULARITY_SECONDS != 0:
                            continue
                        row = {
                            "epoch": epoch,
                            "open": float(candle_data.get("open", 0)),
                            "high": float(candle_data.get("high", 0)),
                            "low": float(candle_data.get("low", 0)),
                            "close": float(candle_data.get("close", 0)),
                            "volume": float(candle_data.get("volume", 0) or 0)
                        }
                        df.loc[len(df)] = row
                        if len(df) > MAX_CANDLES:
                            df = df.tail(MAX_CANDLES).reset_index(drop=True)
                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)
                        gerar_sinal(df, symbol)
        except:
            await asyncio.sleep(3)

async def main():
    start_msg = "BOT INICIADO!"
    send_telegram(start_msg, bypass_throttle=True)
    await asyncio.gather(*(monitor_symbol(s) for s in SYMBOLS))

app = Flask(__name__)
@app.get("/")
def home():
    return "BOT ONLINE", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",10000)))

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    try:
        asyncio.run(main())
    except:
        pass
