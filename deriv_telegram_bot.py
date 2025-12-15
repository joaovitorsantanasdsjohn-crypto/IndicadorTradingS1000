# deriv_telegram_bot.py
# IndicadorTradingS1000 — versão corrigida e funcional
# Estrutura preservada, correções aplicadas conforme solicitado

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

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60
SIGNAL_ADVANCE_SECONDS = 3

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Parâmetros Técnicos ----------------
EMA_FAST = 9
EMA_MID = 20
EMA_SLOW = 50

RSI_BUY_MAX = 52
RSI_SELL_MIN = 48

BB_PERIOD = 20
BB_STD = 2.0

# ---------------- ML ----------------
ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 200
ML_MAX_SAMPLES = 2000
ML_CONF_THRESHOLD = 0.55
ML_N_ESTIMATORS = 40
ML_MAX_DEPTH = 4

# ---------------- Estado ----------------
candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {}
last_signal_epoch = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}

# ---------------- Logging ----------------
logger = logging.getLogger("IndicadorS1000")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

def log(msg, level="info"):
    if level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)

# ---------------- Telegram ----------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
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

    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    return df

# ---------------- ML ----------------
def build_ml_dataset(df: pd.DataFrame):
    df = df.copy().dropna().reset_index(drop=True)
    df["future_up"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df.drop(columns=["future_up", "epoch"]).iloc[:-1]
    y = df["future_up"].iloc[:-1]
    if len(X) > ML_MAX_SAMPLES:
        X = X.tail(ML_MAX_SAMPLES)
        y = y.tail(ML_MAX_SAMPLES)
    return X, y

def train_ml(symbol: str):
    if not ML_ENABLED:
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return

    try:
        X, y = build_ml_dataset(df)
        if len(X) < ML_MIN_TRAINED_SAMPLES:
            ml_model_ready[symbol] = False
            return

        model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            random_state=42
        )
        model.fit(X, y)
        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True
        log(f"[ML {symbol}] Modelo treinado ({len(X)} amostras)")
    except Exception as e:
        ml_model_ready[symbol] = False
        log(f"[ML {symbol}] Erro treino: {e}", "error")

def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ml_model_ready.get(symbol):
        return None
    try:
        model, cols = ml_models[symbol]
        X = [[float(row[c]) for c in cols]]
        return model.predict_proba(X)[0][1]
    except Exception:
        return None

# ---------------- Sinal ----------------
def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 5:
        return

    row = df.iloc[-1]

    tendencia_alta = row["ema_fast"] > row["ema_mid"]
    tendencia_baixa = row["ema_fast"] < row["ema_mid"]

    buy = (
        tendencia_alta and
        row["rsi"] < RSI_BUY_MAX and
        row["close"] <= row["bb_mid"]
    )

    sell = (
        tendencia_baixa and
        row["rsi"] > RSI_SELL_MIN and
        row["close"] >= row["bb_mid"]
    )

    if not buy and not sell:
        return

    ml_prob = ml_predict(symbol, row)
    if ml_prob is not None and ml_prob < ML_CONF_THRESHOLD:
        log(f"[{symbol}] Bloqueado ML: {ml_prob:.2f}")
        return

    epoch = int(row["epoch"])
    if last_signal_epoch[symbol] == epoch:
        return

    direction = "CALL 🟢" if buy else "PUT 🔴"
    horario = datetime.utcfromtimestamp(epoch).strftime("%H:%M:%S")

    msg = (
        f"<b>SINAL {symbol}</b>\n"
        f"{direction}\n"
        f"⏱ Entrada: {horario} UTC\n"
        f"📊 RSI: {row['rsi']:.1f}\n"
        f"🤖 ML: {ml_prob:.2f}" if ml_prob else "🤖 ML: N/A"
    )

    send_telegram(msg)
    last_signal_epoch[symbol] = epoch
    last_signal_time[symbol] = time.time()
    log(f"[SINAL {symbol}] {direction}")

# ---------------- WebSocket ----------------
async def ws_loop(symbol: str):
    try:
        async with websockets.connect(WS_URL, ping_interval=60, ping_timeout=30) as ws:
            send_telegram(f"🔌 WS conectado: {symbol}")
            await ws.send(json.dumps({
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 1200,
                "end": "latest",
                "granularity": GRANULARITY_SECONDS,
                "style": "candles"
            }))

            while True:
                msg = await ws.recv()
                data = json.loads(msg)

                if "candles" in data:
                    df = pd.DataFrame(data["candles"])
                    df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)
                    candles[symbol] = calcular_indicadores(df)
                    train_ml(symbol)
                    avaliar_sinal(symbol)

    except Exception as e:
        log(f"[WS {symbol}] Erro: {e}", "error")
        await asyncio.sleep(5)
        await ws_loop(symbol)

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def health():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# ---------------- Main ----------------
async def main():
    send_telegram("🚀 Bot iniciado com sucesso")
    tasks = [ws_loop(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
