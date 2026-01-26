import asyncio
import json
import os
import random
import time
import threading
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Tuple

import pandas as pd
import requests
import websockets
from flask import Flask
from dotenv import load_dotenv

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# ✅ BLOCO 1 — CONFIGURAÇÕES / PARÂMETROS (AJUSTE AQUI)
# ============================================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

DERIV_TOKEN = os.getenv("DERIV_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID", "111022")
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD","frxUSDCAD",
    "frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP","frxEURAUD","frxAUDJPY",
    "frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

CANDLE_INTERVAL_MINUTES = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL_MINUTES * 60

FINAL_ADVANCE_MINUTES = int(os.getenv("FINAL_ADVANCE_MINUTES", "5"))
PREDICT_CANDLES_AHEAD = int(os.getenv("PREDICT_CANDLES_AHEAD", "2"))

WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_OPEN_TIMEOUT = int(os.getenv("WS_OPEN_TIMEOUT", "20"))
WS_CANDLE_TIMEOUT_SECONDS = int(os.getenv("WS_CANDLE_TIMEOUT_SECONDS", "600"))

HISTORY_COUNT = int(os.getenv("HISTORY_COUNT", "1200"))
MAX_CANDLES_IN_RAM = int(os.getenv("MAX_CANDLES_IN_RAM", "1800"))

EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_MID = int(os.getenv("EMA_MID", "21"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "34"))

BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "2.3"))

MFI_PERIOD = int(os.getenv("MFI_PERIOD", "14"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_MIN = float(os.getenv("RSI_MIN", "0"))
RSI_MAX = float(os.getenv("RSI_MAX", "100"))

ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))

ML_ENABLED = bool(int(os.getenv("ML_ENABLED", "1"))) and SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "300"))
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "2000"))
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "80"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "6"))
ML_TRAIN_EVERY_N_CANDLES = int(os.getenv("ML_TRAIN_EVERY_N_CANDLES", "3"))

ML_CALIBRATION_ENABLED = bool(int(os.getenv("ML_CALIBRATION_ENABLED", "1"))) and SKLEARN_AVAILABLE
ML_CALIBRATION_METHOD = os.getenv("ML_CALIBRATION_METHOD", "sigmoid")
ML_CALIBRATION_CV = int(os.getenv("ML_CALIBRATION_CV", "3"))

MIN_SECONDS_BETWEEN_SIGNALS = int(os.getenv("MIN_SECONDS_BETWEEN_SIGNALS", "3"))
STARTUP_STAGGER_MAX_SECONDS = int(os.getenv("STARTUP_STAGGER_MAX_SECONDS", "10"))
MARKET_CLOSED_RECONNECT_WAIT_SECONDS = int(os.getenv("MARKET_CLOSED_RECONNECT_WAIT_SECONDS", "1800"))
ML_FEATURES_SEND_ON_READY = bool(int(os.getenv("ML_FEATURES_SEND_ON_READY", "0")))


# ============================================================
# ✅ BLOCO 2 — ESTADO GLOBAL
# ============================================================
candles: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in SYMBOLS}
ml_models: Dict[str, Tuple["RandomForestClassifier", list]] = {}
ml_model_ready: Dict[str, bool] = {s: False for s in SYMBOLS}

last_signal_time: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
last_signal_epoch: Dict[str, Optional[int]] = {s: None for s in SYMBOLS}
last_processed_epoch: Dict[str, Optional[int]] = {s: None for s in SYMBOLS}
candle_counter: Dict[str, int] = {s: 0 for s in SYMBOLS}
scheduled_signal_epoch: Dict[str, Optional[int]] = {s: None for s in SYMBOLS}


# ============================================================
# ✅ BLOCO 3 — LOGGING
# ============================================================
logger = logging.getLogger("IndicadorTradingS1000")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    utc_now = datetime.now(timezone.utc)
    brt_now = utc_now - timedelta(hours=3)
    utc = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")
    brt = brt_now.strftime("%Y-%m-%d %H:%M:%S BRT")
    full = f"{utc} | {brt} | {msg}"
    getattr(logger, level if level in ["info","warning","error"] else "info")(full)


# ============================================================
# ✅ BLOCO 4 — TELEGRAM
# ============================================================
def send_telegram(message: str):
    try:
        if not TELEGRAM_TOKEN or not CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        log(f"Erro Telegram: {e}", "error")


# ============================================================
# ✅ BLOCO 5 — INDICADORES + FEATURES
# ============================================================
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) < EMA_SLOW + 120:
        return df

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["ema_trend_strength"] = (df["ema_fast"] - df["ema_slow"]).abs()
    df["ema_trend_strength_pct"] = df["ema_trend_strength"] / df["close"]
    df["ema_slow_slope"] = df["ema_slow"].diff().fillna(0)
    df["ema_slope_accel"] = df["ema_slow_slope"].diff().fillna(0)

    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi().clip(RSI_MIN, RSI_MAX)

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]).abs()
    df["bb_width_pct"] = (df["bb_width"] / df["bb_mid"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    df["bb_squeeze"] = (df["bb_width_pct"] < df["bb_width_pct"].rolling(50).mean()).astype(int)

    if "volume" not in df.columns:
        df["volume"] = 1
    df["mfi"] = MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=MFI_PERIOD).money_flow_index()

    df["candle_range"] = (df["high"] - df["low"]).abs()
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = (df["high"] - df[["open","close"]].max(axis=1)).clip(lower=0)
    df["lower_wick"] = (df[["open","close"]].min(axis=1) - df["low"]).clip(lower=0)

    df["ret_1"] = df["close"].pct_change().fillna(0)
    df["close_pos"] = ((df["close"] - df["low"]) / df["candle_range"].replace(0,1)).clip(0,1)
    df["volatility_10"] = df["ret_1"].rolling(10).std().fillna(0)
    df["volatility_20"] = df["ret_1"].rolling(20).std().fillna(0)

    df["dist_close_ema_slow"] = ((df["close"] - df["ema_slow"]) / df["ema_slow"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    df["dist_extreme"] = (df["dist_close_ema_slow"].abs() > df["dist_close_ema_slow"].rolling(100).std()).astype(int)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr_14"] = tr.rolling(14).mean()
    df["atr_14_pct"] = (df["atr_14"] / df["close"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    df["body_vs_atr"] = (df["candle_body"] / df["atr_14"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    df["range_vs_atr"] = (df["candle_range"] / df["atr_14"]).replace([float("inf"), -float("inf")], 0).fillna(0)

    df["range"] = (df["high"] - df["low"]).abs()
    df["adr_5"] = df["range"].rolling(5).mean()
    df["adr_10"] = df["range"].rolling(10).mean()
    df["adr_5_pct"] = (df["adr_5"] / df["close"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    df["adr_10_pct"] = (df["adr_10"] / df["close"]).replace([float("inf"), -float("inf")], 0).fillna(0)

    try:
        adx = ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD)
        df["adx_14"] = adx.adx()
        df["di_plus_14"] = adx.adx_pos()
        df["di_minus_14"] = adx.adx_neg()
    except Exception:
        df["adx_14"] = df["di_plus_14"] = df["di_minus_14"] = 0

    return df


# ============================================================
# ✅ BLOCO 6 — MACHINE LEARNING
# ============================================================
def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    if "epoch" not in df.columns:
        return None, None

    shift_n = int(max(1, PREDICT_CANDLES_AHEAD))
    df["future"] = (df["close"].shift(-shift_n) > df["close"]).astype(int)

    X = df.drop(columns=[c for c in ["future","epoch"] if c in df.columns]).iloc[:-shift_n]
    y = df["future"].iloc[:-shift_n]

    if len(X) <= 10:
        return None, None

    return X.tail(ML_MAX_SAMPLES), y.tail(ML_MAX_SAMPLES)


async def train_ml_async(symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return

    X, y = build_ml_dataset(df)
    if X is None:
        ml_model_ready[symbol] = False
        return

    def _fit():
        base = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            random_state=42,
            n_jobs=-1
        )
        base.fit(X, y)

        if ML_CALIBRATION_ENABLED:
            model = CalibratedClassifierCV(base, method=ML_CALIBRATION_METHOD, cv=ML_CALIBRATION_CV)
            model.fit(X, y)
            return model, X.columns.tolist()

        return base, X.columns.tolist()

    model, cols = await asyncio.to_thread(_fit)
    ml_models[symbol] = (model, cols)
    ml_model_ready[symbol] = True
    log(f"{symbol} ML pronto ✅", "info")


def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not (ML_ENABLED and ml_model_ready.get(symbol) and symbol in ml_models):
        return None
    model, cols = ml_models[symbol]
    X_pred = pd.DataFrame([[float(row[c]) for c in cols]], columns=cols)
    return float(model.predict_proba(X_pred)[0][1])


# ============================================================
# ✅ BLOCO 7 — SINAIS
# ============================================================
def floor_to_granularity(ts_epoch: int, gran_seconds: int) -> int:
    return (ts_epoch // gran_seconds) * gran_seconds


async def schedule_telegram_signal(symbol: str, when_epoch_utc: int, msg: str):
    wait_s = when_epoch_utc - int(time.time())
    if wait_s > 0:
        await asyncio.sleep(wait_s)
    send_telegram(msg)
    log(f"{symbol} — sinal enviado ✅", "info")


def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 120:
        return

    row = df.iloc[-1]
    epoch = int(row["epoch"])
    candle_open_epoch = floor_to_granularity(epoch, GRANULARITY_SECONDS)
    target_candle_open = candle_open_epoch + (GRANULARITY_SECONDS * PREDICT_CANDLES_AHEAD)

    if scheduled_signal_epoch[symbol] == target_candle_open:
        return

    if (time.time() - last_signal_time[symbol]) < MIN_SECONDS_BETWEEN_SIGNALS:
        return

    prob = ml_predict(symbol, row)
    if prob is None:
        return

    direction = "COMPRA" if prob >= 0.5 else "VENDA"
    confidence = prob if direction == "COMPRA" else (1 - prob)
    if confidence < ML_CONF_THRESHOLD:
        return

    entry_time_brt = datetime.fromtimestamp(target_candle_open, tz=timezone.utc) - timedelta(hours=3)
    notify_epoch_utc = target_candle_open - (FINAL_ADVANCE_MINUTES * 60)

    ativo = symbol.replace("frx","")
    msg = f"📊 ATIVO: {ativo}\n📌 DIREÇÃO: {direction}\n⏰ ENTRADA: {entry_time_brt.strftime('%H:%M')}\n🤖 ML: {confidence*100:.0f}%"

    asyncio.create_task(schedule_telegram_signal(symbol, notify_epoch_utc, msg))
    last_signal_time[symbol] = time.time()
    scheduled_signal_epoch[symbol] = target_candle_open


# ============================================================
# ✅ BLOCO 8 — WEBSOCKET
# ============================================================
async def deriv_authorize(ws):
    if not DERIV_TOKEN:
        return
    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    await ws.recv()


async def request_history(ws, symbol: str) -> pd.DataFrame:
    await ws.send(json.dumps({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": HISTORY_COUNT,
        "end": "latest",
        "granularity": GRANULARITY_SECONDS,
        "style": "candles"
    }))
    data = json.loads(await ws.recv())
    return pd.DataFrame(data.get("candles", []))


async def subscribe_candles(ws, symbol: str):
    await ws.send(json.dumps({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 1,
        "end": "latest",
        "granularity": GRANULARITY_SECONDS,
        "style": "candles",
        "subscribe": 1
    }))


def df_trim(df: pd.DataFrame) -> pd.DataFrame:
    return df.tail(MAX_CANDLES_IN_RAM).reset_index(drop=True)


async def ws_loop(symbol: str):
    await asyncio.sleep(random.uniform(0, STARTUP_STAGGER_MAX_SECONDS))
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                await deriv_authorize(ws)
                df = calcular_indicadores(df_trim(await request_history(ws, symbol)))
                candles[symbol] = df

                if ML_ENABLED:
                    await train_ml_async(symbol)

                await subscribe_candles(ws, symbol)

                while True:
                    data = json.loads(await ws.recv())
                    if "candles" in data:
                        new_row = data["candles"][0]
                        df = pd.concat([candles[symbol], pd.DataFrame([new_row])], ignore_index=True)
                        df = calcular_indicadores(df_trim(df))
                        candles[symbol] = df
                        avaliar_sinal(symbol)

        except Exception as e:
            log(f"{symbol} WS erro: {e}", "error")
            await asyncio.sleep(5)


# ============================================================
# ✅ BLOCO 9 — FLASK HEALTHCHECK
# ============================================================
app = Flask(__name__)

@app.route("/", methods=["GET","HEAD"])
def health():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))


# ============================================================
# ✅ BLOCO 10 — MAIN
# ============================================================
async def main():
    send_telegram("✅ BOT INICIADO — M5 ATIVO")
    await asyncio.gather(*[ws_loop(s) for s in SYMBOLS])


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
