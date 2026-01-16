import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask
import threading
import os
import time
import logging
from typing import Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID", "111022")

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SIGNAL_ADVANCE_MINUTES = int(os.getenv("SIGNAL_ADVANCE_MINUTES", "5"))

WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "25"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_CANDLE_TIMEOUT_SECONDS = int(os.getenv("WS_CANDLE_TIMEOUT_SECONDS", "180"))

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 34

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.3
MFI_PERIOD = 14

ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "240"))
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "1200"))
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "45"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "5"))

MAX_CANDLES_IN_RAM = int(os.getenv("MAX_CANDLES_IN_RAM", "1350"))
TRAIN_EVERY_N_CANDLES = int(os.getenv("TRAIN_EVERY_N_CANDLES", "3"))

STARTUP_NOTIFY_COOLDOWN_MIN = int(os.getenv("STARTUP_NOTIFY_COOLDOWN_MIN", "60"))
STARTUP_NOTIFY_FILE = "startup_notify.txt"

logger = logging.getLogger("IndicadorTradingS1000")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)


def utc_now():
    return datetime.now(timezone.utc)


def log(msg: str, level: str = "info"):
    utc = utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")
    brt = (utc_now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S BRT")
    full = f"{utc} | {brt} | {msg}"
    if level == "info":
        logger.info(full)
    elif level == "warning":
        logger.warning(full)
    else:
        logger.error(full)


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=8)
    except Exception as e:
        log(f"Erro Telegram: {e}", "error")


def maybe_send_startup_message():
    try:
        now = time.time()
        if os.path.exists(STARTUP_NOTIFY_FILE):
            with open(STARTUP_NOTIFY_FILE, "r", encoding="utf-8") as f:
                last = float(f.read().strip() or "0")
            if now - last < STARTUP_NOTIFY_COOLDOWN_MIN * 60:
                return
        with open(STARTUP_NOTIFY_FILE, "w", encoding="utf-8") as f:
            f.write(str(now))
        send_telegram("🚀 BOT INICIADO — M5 ATIVO")
    except Exception:
        send_telegram("🚀 BOT INICIADO — M5 ATIVO")


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    if len(arr) < period + 2:
        return np.full_like(arr, np.nan, dtype=float)
    alpha = 2.0 / (period + 1.0)
    out = np.empty(len(arr), dtype=float)
    out[:] = np.nan
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    if len(close) < period + 2:
        return np.full(len(close), np.nan, dtype=float)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.empty(len(close), dtype=float); avg_gain[:] = np.nan
    avg_loss = np.empty(len(close), dtype=float); avg_loss[:] = np.nan
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    out[:period] = np.nan
    return out


def bollinger_mid(close: np.ndarray, period: int) -> np.ndarray:
    if len(close) < period:
        return np.full(len(close), np.nan, dtype=float)
    out = np.full(len(close), np.nan, dtype=float)
    c = pd.Series(close)
    out[period - 1:] = c.rolling(period).mean().values[period - 1:]
    return out


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan, dtype=float)
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta_tp = np.diff(tp)
    pos_mf = np.where(delta_tp > 0, mf[1:], 0.0)
    neg_mf = np.where(delta_tp < 0, mf[1:], 0.0)
    pos = pd.Series(pos_mf).rolling(period).sum().values
    neg = pd.Series(neg_mf).rolling(period).sum().values
    out = np.full(n, np.nan, dtype=float)
    ratio = pos / (neg + 1e-12)
    mfi_vals = 100 - (100 / (1 + ratio))
    out[period + 1:] = mfi_vals[period:]
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    need = max(EMA_SLOW + 5, BB_PERIOD + 5, RSI_PERIOD + 5, MFI_PERIOD + 5)
    if len(df) < need:
        return df

    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    if "volume" not in df.columns:
        vol = np.ones(len(df), dtype=float)
    else:
        vol = df["volume"].fillna(1).astype(float).to_numpy()

    e_fast = ema(close, EMA_FAST)
    e_mid = ema(close, EMA_MID)
    e_slow = ema(close, EMA_SLOW)
    r = rsi(close, RSI_PERIOD)
    bbm = bollinger_mid(close, BB_PERIOD)
    mf = mfi(high, low, close, vol, MFI_PERIOD)

    df["ema_fast"] = e_fast
    df["ema_mid"] = e_mid
    df["ema_slow"] = e_slow
    df["rsi"] = r
    df["bb_mid"] = bbm
    df["mfi"] = mf
    return df


def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        return None, None

    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)

    feat_cols = ["open", "high", "low", "close", "ema_fast", "ema_mid", "ema_slow", "rsi", "bb_mid", "mfi"]
    for c in feat_cols:
        df[c] = df[c].astype(float)

    X = df[feat_cols].iloc[:-1].tail(ML_MAX_SAMPLES)
    y = df["future"].iloc[:-1].tail(ML_MAX_SAMPLES)
    return X, y


candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_ready = {s: False for s in SYMBOLS}

last_signal_epoch = {s: None for s in SYMBOLS}
last_processed_epoch = {s: None for s in SYMBOLS}
candle_counter = {s: 0 for s in SYMBOLS}


def train_ml(symbol: str):
    if not ML_ENABLED:
        return
    df = candles[symbol]
    X, y = build_ml_dataset(df)
    if X is None:
        ml_ready[symbol] = False
        return
    model = RandomForestClassifier(
        n_estimators=ML_N_ESTIMATORS,
        max_depth=ML_MAX_DEPTH,
        random_state=42,
        n_jobs=1
    )
    model.fit(X, y)
    ml_models[symbol] = model
    ml_ready[symbol] = True


def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ML_ENABLED or not ml_ready.get(symbol):
        return None
    model = ml_models.get(symbol)
    if model is None:
        return None
    try:
        feat = np.array([[
            float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
            float(row["ema_fast"]), float(row["ema_mid"]), float(row["ema_slow"]),
            float(row["rsi"]), float(row["bb_mid"]), float(row["mfi"])
        ]], dtype=float)
        return float(model.predict_proba(feat)[0][1])
    except Exception:
        return None


def align_entry_epoch(epoch: int) -> int:
    base = (epoch // GRANULARITY_SECONDS) * GRANULARITY_SECONDS
    return base + GRANULARITY_SECONDS


def format_brt_time_from_epoch(epoch: int) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc) - timedelta(hours=3)
    return dt.strftime("%H:%M")


def should_send_signal(symbol: str, row: pd.Series) -> Optional[str]:
    if pd.isna(row.get("ema_fast")) or pd.isna(row.get("ema_mid")) or pd.isna(row.get("ema_slow")):
        return None
    direction = "COMPRA" if float(row["ema_fast"]) >= float(row["ema_mid"]) else "VENDA"
    return direction


async def send_signal_at_time(symbol: str, direction: str, ml_prob: float, entry_epoch: int):
    send_epoch = entry_epoch - (SIGNAL_ADVANCE_MINUTES * 60)
    now = int(time.time())
    delay = send_epoch - now
    if delay > 0:
        await asyncio.sleep(delay)

    ativo = symbol.replace("frx", "")
    entry_time = format_brt_time_from_epoch(entry_epoch)

    msg = (
        f"📊 <b>ATIVO:</b> {ativo}\n"
        f"📈 <b>DIREÇÃO:</b> {direction}\n"
        f"⏰ <b>ENTRADA:</b> {entry_time}\n"
        f"🤖 <b>ML:</b> {ml_prob*100:.0f}%"
    )
    send_telegram(msg)
    log(f"{symbol} — sinal enviado {direction} (ML {ml_prob*100:.0f}%)", "info")


def evaluate_signal(symbol: str):
    df = candles[symbol]
    if len(df) < max(EMA_SLOW + 60, ML_MIN_TRAINED_SAMPLES):
        return

    row = df.iloc[-1]
    direction = should_send_signal(symbol, row)
    if not direction:
        return

    ml_prob = ml_predict(symbol, row)
    if ml_prob is None or ml_prob < ML_CONF_THRESHOLD:
        return

    candle_epoch = int(row["epoch"])
    entry_epoch = align_entry_epoch(candle_epoch)

    if last_signal_epoch[symbol] == entry_epoch:
        return
    last_signal_epoch[symbol] = entry_epoch

    asyncio.create_task(send_signal_at_time(symbol, direction, ml_prob, entry_epoch))


async def ws_loop(symbol: str):
    while True:
        try:
            log(f"{symbol} WS conectando...", "info")
            async with websockets.connect(
                WS_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
                close_timeout=5
            ) as ws:
                log(f"{symbol} WS conectado ✅", "info")

                req_hist = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1200,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles"
                }
                await ws.send(json.dumps(req_hist))
                log(f"{symbol} Histórico solicitado 📥", "info")

                raw = await ws.recv()
                data = json.loads(raw)
                if "error" in data:
                    log(f"{symbol} WS erro hist: {data.get('error')}", "error")
                    await ws.close()
                    continue

                df = pd.DataFrame(data.get("candles", []))
                if df.empty:
                    log(f"{symbol} Histórico vazio — reconectando", "warning")
                    await ws.close()
                    continue

                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")

                df = df.tail(MAX_CANDLES_IN_RAM).reset_index(drop=True)
                df = compute_features(df)
                candles[symbol] = df

                if ML_ENABLED:
                    train_ml(symbol)

                req_stream = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles",
                    "subscribe": 1
                }
                await ws.send(json.dumps(req_stream))
                log(f"{symbol} Stream (candles) ligado 🔴", "info")

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_CANDLE_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        log(f"{symbol} Sem candles por {WS_CANDLE_TIMEOUT_SECONDS}s — reconectando", "warning")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    msg = json.loads(raw)

                    if "error" in msg:
                        log(f"{symbol} WS retornou erro: {msg.get('error')}", "error")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    if "candles" not in msg:
                        continue

                    new_row = msg["candles"][0]
                    if not new_row:
                        continue

                    df = candles[symbol]
                    if df.empty:
                        continue

                    try:
                        new_epoch = int(new_row.get("epoch"))
                    except Exception:
                        continue

                    last_epoch = int(df.iloc[-1]["epoch"])

                    if new_epoch == last_epoch:
                        for k in ("open", "high", "low", "close"):
                            if k in new_row:
                                df.at[df.index[-1], k] = float(new_row[k])
                    else:
                        row_df = pd.DataFrame([new_row])
                        for col in ["open", "high", "low", "close"]:
                            row_df[col] = pd.to_numeric(row_df[col], errors="coerce")
                        row_df["epoch"] = pd.to_numeric(row_df["epoch"], errors="coerce").astype("Int64")
                        df = pd.concat([df, row_df], ignore_index=True)

                    df = df.tail(MAX_CANDLES_IN_RAM).reset_index(drop=True)
                    df = compute_features(df)
                    candles[symbol] = df

                    cur_epoch = int(df.iloc[-1]["epoch"])
                    if last_processed_epoch[symbol] == cur_epoch:
                        continue
                    last_processed_epoch[symbol] = cur_epoch

                    candle_counter[symbol] += 1
                    if ML_ENABLED and (candle_counter[symbol] % TRAIN_EVERY_N_CANDLES == 0):
                        train_ml(symbol)

                    evaluate_signal(symbol)

        except Exception as e:
            log(f"{symbol} WS erro: {e}", "error")
            await asyncio.sleep(3)


app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def health():
    return "OK", 200


def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


async def main():
    maybe_send_startup_message()
    tasks = [ws_loop(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
