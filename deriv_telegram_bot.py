# deriv_telegram_bot.py
# IndicadorTradingS1000 — versão com logs detalhados

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

# Ping/Pong agressivo para melhor manter conexão
WS_PING_INTERVAL = 30
WS_PING_TIMEOUT  = 10

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
EMA_MID  = 20
EMA_SLOW = 50

RSI_BUY_MAX  = 52
RSI_SELL_MIN = 48

BB_PERIOD = 20
BB_STD    = 2

# ---------------- ML ----------------
ML_ENABLED             = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 200
ML_MAX_SAMPLES         = 2000
ML_CONF_THRESHOLD      = 0.55
ML_N_ESTIMATORS        = 40
ML_MAX_DEPTH           = 4

# ---------------- Estado ----------------
candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models       = {}
ml_model_ready  = {}
last_signal_epoch = {s: None for s in SYMBOLS}
last_signal_time  = {s: 0 for s in SYMBOLS}

# ---------------- Logging ----------------
logger = logging.getLogger("IndicadorTradingS1000")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s UTC | %(levelname)s | %(message)s"
)
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    brt = (datetime.utcnow() + timedelta(hours=-3)).strftime("%Y-%m-%d %H:%M:%S BRT")
    utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    full = f"{utc} | {brt} | {msg}"
    if level == "info":
        logger.info(full)
    elif level == "warning":
        logger.warning(full)
    elif level == "error":
        logger.error(full)
    else:
        logger.debug(full)

# ---------------- Telegram ----------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("Telegram não configurado", "warning")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"Telegram envio falhou: {r.text}", "error")
        else:
            log("Telegram mensagem enviada", "info")
    except Exception as e:
        log(f"Erro ao enviar Telegram: {e}", "error")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) < EMA_SLOW:
        log("Indicadores pulados — candles insuficientes", "warning")
        return df

    try:
        df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
        df["ema_mid"]  = EMAIndicator(df["close"], EMA_MID).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
        df["rsi"]      = RSIIndicator(df["close"], 14).rsi()
        macd           = MACD(df["close"])
        df["macd_diff"] = macd.macd_diff()

        bb = BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()

        log(f"Indicadores calculados — última RSI {df['rsi'].iloc[-1]:.2f}", "info")
    except Exception as e:
        log(f"Erro ao calcular indicadores: {e}", "error")

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
    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        log(f"[ML {symbol}] aguardando candles para treinar ({len(df)})", "info")
        return

    try:
        X, y = build_ml_dataset(df)
        model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            random_state=42
        )
        model.fit(X, y)
        ml_models[symbol]      = (model, X.columns.tolist())
        ml_model_ready[symbol] = True
        log(f"[ML {symbol}] treinado com {len(X)} amostras", "info")
    except Exception as e:
        ml_model_ready[symbol] = False
        log(f"[ML {symbol}] erro no treino: {e}", "error")

def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ml_model_ready.get(symbol):
        return None
    try:
        model, cols = ml_models[symbol]
        vals = [float(row[c]) for c in cols]
        return model.predict_proba([vals])[0][1]
    except Exception as e:
        log(f"[ML {symbol}] erro na predição: {e}", "error")
        return None

# ---------------- SINAL ----------------
def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 5:
        return

    row = df.iloc[-1]

    buy = (
        row["ema_fast"] > row["ema_mid"] and
        row["rsi"] < RSI_BUY_MAX and
        row["close"] <= row["bb_mid"]
    )
    sell = (
        row["ema_fast"] < row["ema_mid"] and
        row["rsi"] > RSI_SELL_MIN and
        row["close"] >= row["bb_mid"]
    )
    if not buy and not sell:
        log(f"{symbol} — sem setup técnico", "info")
        return

    ml_prob = ml_predict(symbol, row)
    if ml_prob is not None and ml_prob < ML_CONF_THRESHOLD:
        log(f"{symbol} — sinal bloqueado pelo ML ({ml_prob:.2f})", "info")
        return

    direction = "CALL" if buy else "PUT"
    epoch    = int(row["epoch"])
    if last_signal_epoch[symbol] == epoch:
        return

    last_signal_epoch[symbol] = epoch

    dt_utc = datetime.utcfromtimestamp(epoch)
    dt_brt = dt_utc + timedelta(hours=-3)
    msg = (
        f"📊 <b>{symbol}</b>\n"
        f"🎯 Direção: {direction}\n"
        f"⏱ Entrada: {dt_brt.strftime('%H:%M:%S')} BRT\n"
        f"🤖 ML: {ml_prob if ml_prob is not None else 'N/A'}\n"
        f"📈 RSI: {row['rsi']:.2f}"
    )
    send_telegram(msg)
    log(f"{symbol} — sinal enviado {direction}", "info")

# ---------------- WebSocket ----------------
async def ws_loop(symbol: str):
    while True:
        try:
            log(f"{symbol} — conectando WS...", "info")
            async with websockets.connect(
                WS_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT
            ) as ws:
                log(f"{symbol} — WS conectado", "info")
                send_telegram(f"🔌 WS conectado: {symbol}")

                req_hist = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": INITIAL_HISTORY_COUNT,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles"
                }
                await ws.send(json.dumps(req_hist))
                log(f"{symbol} — histórico solicitado", "info")

                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)

                    if "candles" in data:
                        df = pd.DataFrame(data["candles"])
                        candles[symbol] = calcular_indicadores(df)
                        log(f"{symbol} — histórico recebido ({len(df)})", "info")
                        train_ml(symbol)
                        avaliar_sinal(symbol)
                    await asyncio.sleep(0)

        except Exception as e:
            log(f"{symbol} — WS erro: {e}", "error")
            await asyncio.sleep(5)

# ---------------- Flask Keep Alive ----------------
app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def home():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

# ---------------- MAIN ----------------
async def main():
    send_telegram("🚀 BOT INICIADO — logs detalhados ativados")
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
