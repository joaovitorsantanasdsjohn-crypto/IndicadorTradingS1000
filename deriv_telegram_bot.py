# Indicador Trading S1000

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator   # <<< ADICIONADO
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from pathlib import Path
import time
import logging
import traceback
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
APP_ID = os.getenv("DERIV_APP_ID", "111022")

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SIGNAL_ADVANCE_MINUTES = 5

WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# Watchdog WS: reconecta se parar de vir candle
WS_CANDLE_TIMEOUT_SECONDS = int(os.getenv("WS_CANDLE_TIMEOUT_SECONDS", "300"))

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

# ---------------- Parâmetros Técnicos ----------------

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 34

RSI_BUY_MAX = 45
RSI_SELL_MIN = 55

BB_PERIOD = 20
BB_STD = 2.3

MFI_PERIOD = 14

# ---------------- ML ----------------

ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 300
ML_MAX_SAMPLES = 2000
ML_CONF_THRESHOLD = 0.55
ML_N_ESTIMATORS = 60
ML_MAX_DEPTH = 5

# ---------------- Estado ----------------

candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {}
last_signal_epoch = {s: None for s in SYMBOLS}

# evita reprocessar várias vezes o mesmo candle
last_processed_epoch = {s: None for s in SYMBOLS}

# ---------------- Logging ----------------

logger = logging.getLogger("IndicadorTradingS1000")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    brt = (datetime.utcnow() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S BRT")
    utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    full = f"{utc} | {brt} | {msg}"
    if level == "info":
        logger.info(full)
    elif level == "warning":
        logger.warning(full)
    elif level == "error":
        logger.error(full)

# ---------------- Telegram ----------------

def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        log(f"Erro Telegram: {e}", "error")

# ---------------- Indicadores ----------------

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) < EMA_SLOW:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    if "volume" not in df.columns:
        df["volume"] = 1  # volume neutro forex

    df["mfi"] = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=MFI_PERIOD
    ).money_flow_index()

    return df

# ---------------- ML ----------------

def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df.drop(columns=["future","epoch"]).iloc[:-1]
    y = df["future"].iloc[:-1]
    return X.tail(ML_MAX_SAMPLES), y.tail(ML_MAX_SAMPLES)

def train_ml(symbol: str):
    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return
    X, y = build_ml_dataset(df)
    model = RandomForestClassifier(
        n_estimators=ML_N_ESTIMATORS,
        max_depth=ML_MAX_DEPTH,
        random_state=42
    )
    model.fit(X, y)
    ml_models[symbol] = (model, X.columns.tolist())
    ml_model_ready[symbol] = True

def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ml_model_ready.get(symbol):
        return None
    model, cols = ml_models[symbol]
    vals = [float(row[c]) for c in cols]
    return model.predict_proba([vals])[0][1]

# ---------------- SINAL ----------------

def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 5:
        return

    row = df.iloc[-1]

    direction = "COMPRA" if row["ema_fast"] >= row["ema_mid"] else "VENDA"

    ml_prob = ml_predict(symbol, row)
    if ml_prob is None or ml_prob < ML_CONF_THRESHOLD:
        return

    epoch = int(row["epoch"])
    if last_signal_epoch[symbol] == epoch:
        return
    last_signal_epoch[symbol] = epoch

    entry_time = datetime.utcfromtimestamp(epoch) - timedelta(hours=3)
    entry_time += timedelta(minutes=SIGNAL_ADVANCE_MINUTES)

    ativo = symbol.replace("frx", "")

    msg = (
        f"📊 <b>ATIVO:</b> {ativo}\n"
        f"📈 <b>DIREÇÃO:</b> {direction}\n"
        f"⏰ <b>ENTRADA:</b> {entry_time.strftime('%H:%M')}\n"
        f"🤖 <b>ML:</b> {ml_prob*100:.0f}%"
    )

    send_telegram(msg)
    log(f"{symbol} — sinal enviado {direction}")

# ---------------- Utils candles ----------------

def update_history(symbol: str, df_new: pd.DataFrame):
    if df_new is None or df_new.empty:
        return

    df_new = df_new.copy()
    for col in ["epoch", "open", "high", "low", "close"]:
        if col not in df_new.columns:
            return

    df_new["epoch"] = df_new["epoch"].astype(int)

    if candles[symbol].empty:
        base = df_new
    else:
        base = pd.concat([candles[symbol], df_new], ignore_index=True)
        base = base.drop_duplicates(subset=["epoch"], keep="last")
        base = base.sort_values("epoch").reset_index(drop=True)

    if len(base) > ML_MAX_SAMPLES:
        base = base.tail(ML_MAX_SAMPLES).reset_index(drop=True)

    candles[symbol] = calcular_indicadores(base)

# ---------------- WebSocket ----------------

async def ws_loop(symbol: str):
    while True:
        try:
            log(f"{symbol} WS conectando...", "info")

            async with websockets.connect(
                WS_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT
            ) as ws:

                log(f"{symbol} WS conectado ✅", "info")

                # 1) HISTÓRICO (sem subscribe)
                req_hist = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1200,
                    "end": "latest",
                    "granularity": GRANANULARITY_SECONDS if False else GRANULARITY_SECONDS,
                    "style": "candles"
                }
                await ws.send(json.dumps(req_hist))

                # aguardar histórico
                hist_ok = False
                start_wait = time.time()

                while time.time() - start_wait < 15:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if "candles" in data:
                        df = pd.DataFrame(data["candles"])
                        update_history(symbol, df)
                        hist_ok = True
                        break

                if not hist_ok:
                    log(f"{symbol} Não recebeu histórico — reconectando", "warning")
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    continue

                # treina com histórico inicial
                train_ml(symbol)
                avaliar_sinal(symbol)

                # 2) SUBSCRIBE CANDLES (stream real)
                sub_req = {
                    "candles": symbol,
                    "granularity": GRANULARITY_SECONDS,
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))
                log(f"{symbol} Subscribe candles ✅", "info")

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_CANDLE_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        log(f"{symbol} WS ficou mudo {WS_CANDLE_TIMEOUT_SECONDS}s — reconectando 🔁", "warning")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    data = json.loads(raw)

                    # stream vem como: {"candles": [...] } ou {"ohlc": {...}}
                    if "candles" in data:
                        df_new = pd.DataFrame(data["candles"])
                        update_history(symbol, df_new)

                    elif "ohlc" in data:
                        df_new = pd.DataFrame([data["ohlc"]])
                        update_history(symbol, df_new)

                    else:
                        continue

                    # processa somente quando candle muda de epoch
                    try:
                        current_epoch = int(candles[symbol].iloc[-1]["epoch"])
                    except Exception:
                        continue

                    if last_processed_epoch[symbol] != current_epoch:
                        last_processed_epoch[symbol] = current_epoch
                        train_ml(symbol)
                        avaliar_sinal(symbol)

        except Exception as e:
            log(f"{symbol} WS erro: {e}", "error")
            await asyncio.sleep(5)

# ---------------- Flask ----------------

app = Flask(__name__)

@app.route("/", methods=["GET","HEAD"])
def health():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

# ---------------- MAIN ----------------

async def main():
    send_telegram("🚀 BOT INICIADO — M5 ATIVO")
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
