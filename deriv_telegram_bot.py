# deriv_telegram_bot.py — LÓGICA A (Opção A — Precisão Profissional para FOREX M5)
# Versão ATUALIZADA: filtros de PRICE ACTION e ATR removidos para mais sinais

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
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback
import math
from collections import deque
import html
import gc

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações principais ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxGBPAUD", "frxGBPCAD", "frxAUDNZD",
    "frxEURCAD"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Parâmetros ----------------
BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

MIN_SECONDS_BETWEEN_SIGNALS = 3
MIN_SECONDS_BETWEEN_OPPOSITE = 45
MIN_CANDLES_BETWEEN_SIGNALS = 4

REL_EMA_SEP_PCT = 5e-06
MICRO_FORCE_ALLOW_THRESHOLD = 25
FORCE_MIN = 35

ML_ENABLED = SKLEARN_AVAILABLE
ML_N_ESTIMATORS = 40
ML_MAX_DEPTH = 4
ML_MIN_TRAINED_SAMPLES = 200
ML_CONF_THRESHOLD = 0.55
ML_MAX_SAMPLES = 2000
ML_RETRAIN_INTERVAL = 50

MIN_SIGNALS_PER_HOUR = 3
FALLBACK_WINDOW_SEC = 3600
FALLBACK_FORCE_MIN = 30
FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD = 20
FALLBACK_REL_EMA_SEP_PCT = 2e-05
FALLBACK_DURATION_SECONDS = 15 * 60

DEFAULT_EMA_SEP_SCALE = 0.01

INITIAL_HISTORY_COUNT = 500
HISTORY_MAX_TRIES = 5
HISTORY_RETRY_DELAY = 10 * 60
HISTORY_LOG_THROTTLE = 60

EMA_FAST = 9
EMA_MID = 20
EMA_SLOW = 200

# ---------------- Estado ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}

ml_models = {}
ml_model_ready = {}
sent_timestamps = deque()
fallback_active_until = 0.0
pending_signals = {}
historical_loaded = {s: False for s in SYMBOLS}
live_subscribed = {s: False for s in SYMBOLS}
ml_trained_samples = {s: 0 for s in SYMBOLS}
notify_flags = {s: {"connected": False, "history": False, "ml": False, "subscribed": False} for s in SYMBOLS}
history_fail_count = {s: 0 for s in SYMBOLS}
last_history_log_time = {s: 0.0 for s in SYMBOLS}
MAX_CANDLES = 300

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
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
    print(msg, flush=True)

# ---------------- Telegram helper ----------------
def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False):
    now = time.time()
    if symbol and not bypass_throttle:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            log(f"[TG] throttle skip for {symbol}", "warning")
            return
        last_notify_time[symbol] = now
    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("⚠️ Telegram não configurado.", "warning")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
    except Exception as e:
        log(f"[TG] Erro ao enviar: {e}\n{traceback.format_exc()}", "error")

def notify_once(symbol: str, key: str, message: str, bypass=False):
    if symbol not in notify_flags:
        notify_flags[symbol] = {}
    flags = notify_flags.get(symbol, {})
    if flags.get(key):
        return
    try:
        send_telegram(html.escape(message), bypass_throttle=bypass)
    except Exception:
        log(f"[{symbol}] Falha ao notificar Telegram (notify_once).", "warning")
    flags[key] = True
    notify_flags[symbol] = flags

# ---------------- Utilitários ----------------
def human_pair(symbol: str) -> str:
    return symbol.replace("frx", "")

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("epoch").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float) if c in df.columns else 0.0
    df["volume"] = df.get("volume", 0.0).astype(float)

    df[f"ema{EMA_FAST}"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df[f"ema{EMA_MID}"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df[f"ema{EMA_SLOW}"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    try:
        macd = MACD(df["close"], 26, 12, 9)
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd_diff"] = pd.NA

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    df["rel_sep"] = (df[f"ema{EMA_MID}"] - df[f"ema{EMA_SLOW}"]).abs() / df["close"].replace(0, 1e-12)
    return df

# ---------------- ML ----------------
def _build_ml_dataset(df: pd.DataFrame):
    df2 = df.copy().reset_index(drop=True)
    features = [
        "open", "high", "low", "close", "volume",
        f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}",
        "rsi", "macd_diff", "bb_upper", "bb_lower", "bb_mavg", "bb_width", "rel_sep"
    ]
    for c in features:
        df2[c] = df2.get(c, 0.0)
    y = (df2["close"].shift(-1) > df2["close"]).astype(int)
    X = df2.iloc[:-1].copy()
    y = y.iloc[:-1].copy()
    if len(X) > ML_MAX_SAMPLES:
        X = X.tail(ML_MAX_SAMPLES).reset_index(drop=True)
        y = y.tail(ML_MAX_SAMPLES).reset_index(drop=True)
    return X, y

def train_ml_for_symbol(df: pd.DataFrame, symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return False
    try:
        X, y = _build_ml_dataset(df)
        if len(X) < ML_MIN_TRAINED_SAMPLES or len(y.unique()) < 2:
            ml_model_ready[symbol] = False
            return False
        last_trained = ml_trained_samples.get(symbol, 0)
        if last_trained > 0 and len(X) < last_trained + ML_RETRAIN_INTERVAL:
            ml_model_ready[symbol] = True
            return True
        model = RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH, random_state=42)
        model.fit(X, y)
        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True
        ml_trained_samples[symbol] = len(X)
        return True
    except Exception:
        ml_model_ready[symbol] = False
        return False

def ml_predict_prob(symbol: str, last_row: pd.Series) -> float:
    try:
        if not ml_model_ready.get(symbol):
            return None
        model, cols = ml_models.get(symbol, (None, None))
        if model is None or cols is None:
            return None
        Xrow = [float(last_row.get(c, 0.0)) for c in cols]
        prob_up = float(model.predict_proba([Xrow])[0][1])
        return prob_up
    except Exception:
        return None

# ---------------- Fallback ----------------
def prune_sent_timestamps():
    cutoff = time.time() - FALLBACK_WINDOW_SEC
    while sent_timestamps and sent_timestamps[0] < cutoff:
        sent_timestamps.popleft()

def check_and_activate_fallback():
    prune_sent_timestamps()
    global fallback_active_until
    if len(sent_timestamps) < MIN_SIGNALS_PER_HOUR:
        fallback_active_until = time.time() + FALLBACK_DURATION_SECONDS

def is_fallback_active():
    return time.time() < fallback_active_until

# ---------------- Gerar sinal (simplificado) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < EMA_SLOW + 5:
            return None
        now = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(now["close"])
        candle_id = int(now["epoch"]) - (int(now["epoch"]) % GRANULARITY_SECONDS)
        if last_signal_candle.get(symbol) == candle_id:
            return None
        last_candle = last_signal_candle.get(symbol)
        if last_candle is not None and (candle_id - last_candle) // GRANULARITY_SECONDS < MIN_CANDLES_BETWEEN_SIGNALS:
            return None

        ema_fast = float(now[f"ema{EMA_FAST}"])
        ema_mid = float(now[f"ema{EMA_MID}"])
        ema_slow = float(now[f"ema{EMA_SLOW}"])
        triple_up = (ema_fast > ema_mid) and (ema_mid > ema_slow)
        triple_down = (ema_fast < ema_mid) and (ema_mid < ema_slow)

        bb_upper = float(now["bb_upper"])
        bb_lower = float(now["bb_lower"])
        range_bb = bb_upper - bb_lower
        perto_lower = close <= bb_lower + range_bb * BB_PROXIMITY_PCT
        perto_upper = close >= bb_upper - range_bb * BB_PROXIMITY_PCT

        candle_bullish = now["close"] > now["open"]
        candle_bearish = now["close"] < now["open"]

        rsi_now = float(now["rsi"]) if not pd.isna(now["rsi"]) else 50.0
        macd_diff = now.get("macd_diff")
        macd_buy_ok = True if macd_diff is None or pd.isna(macd_diff) else (macd_diff > -MACD_TOLERANCE)
        macd_sell_ok = True if macd_diff is None or pd.isna(macd_diff) else (macd_diff < MACD_TOLERANCE)
        buy_rsi_ok = rsi_now <= RSI_BUY_MAX
        sell_rsi_ok = rsi_now >= RSI_SELL_MIN

        cond_buy = triple_up and (candle_bullish or perto_lower) and buy_rsi_ok and macd_buy_ok
        cond_sell = triple_down and (candle_bearish or perto_upper) and sell_rsi_ok and macd_sell_ok

        if is_fallback_active():
            if not cond_buy and (ema_mid > ema_slow) and candle_bullish:
                cond_buy = True
            if not cond_sell and (ema_mid < ema_slow) and candle_bearish:
                cond_sell = True

        if not (cond_buy or cond_sell):
            return None

        last_state = last_signal_state.get(symbol)
        last_time = last_signal_time.get(symbol, 0)
        now_ts = time.time()
        if last_state is not None and last_state != ("COMPRA" if cond_buy else "VENDA"):
            if now_ts - last_time < MIN_SECONDS_BETWEEN_OPPOSITE:
                return None

        def calc_forca(is_buy: bool):
            score = 0.0
            score += min(25.0, (abs(ema_mid - ema_slow) / max(1e-12, close)) / REL_EMA_SEP_PCT * 25.0)
            return int(max(0, min(100, round(score))))

        tipo = "COMPRA" if cond_buy else "VENDA"
        force = calc_forca(is_buy=(tipo=="COMPRA"))

        if force < (FALLBACK_FORCE_MIN if is_fallback_active() else FORCE_MIN):
            return None

        return {"tipo": tipo, "forca": force, "candle_id": candle_id, "rel_sep": now["rel_sep"]}

    except Exception:
        return None

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    try:
        df.tail(MAX_CANDLES).to_csv(DATA_DIR / f"candles_{symbol}.csv", index=False)
    except Exception:
        pass

# ---------------- Monitor WS ----------------
class HistoryMaxRetriesExceeded(Exception): pass

async def monitor_symbol(symbol: str):
    reconnect_attempt = 0
    df = pd.DataFrame()
    csv_path = DATA_DIR / f"candles_{symbol}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = calcular_indicadores(df)
            if len(df) > MAX_CANDLES:
                df = df.tail(MAX_CANDLES).reset_index(drop=True)
            historical_loaded[symbol] = True
        except Exception:
            pass

    while True:
        try:
            reconnect_attempt += 1
            async with websockets.connect(WS_URL, ping_interval=30, ping_timeout=10) as ws:
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth_raw = await asyncio.wait_for(ws.recv(), timeout=60)

                # histórico inicial
                if not historical_loaded.get(symbol, False):
                    for attempt in range(HISTORY_MAX_TRIES):
                        await ws.send(json.dumps({
                            "ticks_history": symbol,
                            "count": INITIAL_HISTORY_COUNT,
                            "end": "latest",
                            "granularity": GRANULARITY_SECONDS,
                            "style": "candles"
                        }))
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        data = json.loads(raw)
                        candles = []
                        if isinstance(data, dict):
                            if "history" in data and "candles" in data["history"]:
                                candles = data["history"]["candles"]
                            elif "candles" in data:
                                candles = data["candles"]
                        if candles:
                            df = pd.DataFrame(candles)
                            break
                        await asyncio.sleep(1 + random.random()*2)
                    df = calcular_indicadores(df)
                    historical_loaded[symbol] = True
                    save_last_candles(df, symbol)

                # subscribe candles
                if not live_subscribed.get(symbol, False):
                    await ws.send(json.dumps({
                        "ticks_history": symbol,
                        "style": "candles",
                        "granularity": GRANULARITY_SECONDS,
                        "end": "latest",
                        "subscribe": 1
                    }))
                    live_subscribed[symbol] = True

                ultimo_candle_time = time.time()
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=600)
                    msg = json.loads(raw)
                    candle = None
                    if isinstance(msg, dict):
                        if "candle" in msg: candle = msg["candle"]
                        elif "ohlc" in msg: candle = msg["ohlc"]
                        elif "candles" in msg and len(msg["candles"])>0: candle = msg["candles"][-1]
                    if candle is None: continue

                    epoch = int(candle["epoch"])
                    if epoch % GRANULARITY_SECONDS != 0: continue
                    open_p, high_p,
                    open_p, high_p, low_p, close_p = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
                    volume = float(candle.get("volume", 0.0))
                    df.loc[len(df)] = {
                        "epoch": epoch,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": volume
                    }
                    if len(df) > MAX_CANDLES:
                        df = df.tail(MAX_CANDLES).reset_index(drop=True)
                    df = calcular_indicadores(df)

                    sinal = gerar_sinal(df, symbol)
                    if sinal:
                        last_signal_state[symbol] = sinal["tipo"]
                        last_signal_candle[symbol] = sinal["candle_id"]
                        last_signal_time[symbol] = time.time()
                        sent_timestamps.append(time.time())
                        check_and_activate_fallback()
                        msg = f"<b>{human_pair(symbol)}</b>\nSinal: <b>{sinal['tipo']}</b>\nForça: <b>{sinal['forca']}%</b>\nHorário: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
                        send_telegram(msg, symbol=symbol)

                    save_last_candles(df, symbol)

        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
            log(f"[{symbol}] Conexão fechada ou timeout, reconectando... ({e})", "warning")
            await asyncio.sleep(min(30, 2 ** reconnect_attempt))
        except Exception as e:
            log(f"[{symbol}] Erro inesperado: {e}\n{traceback.format_exc()}", "error")
            await asyncio.sleep(5)

# ---------------- Loop principal ----------------
async def main():
    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
