import asyncio
import json
import os
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
import websockets
from dotenv import load_dotenv
from flask import Flask

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# 1) CONFIG / PARAMETROS
# ============================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

DERIV_TOKEN = os.getenv("DERIV_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID", "111022")

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

FINAL_ADVANCE_MINUTES = int(os.getenv("FINAL_ADVANCE_MINUTES", "5"))

WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

WS_CANDLE_TIMEOUT_SECONDS = int(os.getenv("WS_CANDLE_TIMEOUT_SECONDS", "300"))
WS_SLEEP_AFTER_MARKET_CLOSED_SECONDS = int(os.getenv("WS_SLEEP_AFTER_MARKET_CLOSED_SECONDS", str(30 * 60)))

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 34

BB_PERIOD = 20
BB_STD = 2.3

MFI_PERIOD = 14
RSI_PERIOD = 14

ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "300"))
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "2000"))
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "60"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "5"))

ML_RETRAIN_EVERY_CANDLES = int(os.getenv("ML_RETRAIN_EVERY_CANDLES", "3"))


# ============================================================
# 2) ESTADO
# ============================================================

candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {s: False for s in SYMBOLS}

last_signal_epoch = {s: None for s in SYMBOLS}
last_processed_epoch = {s: None for s in SYMBOLS}
candle_counter = {s: 0 for s in SYMBOLS}


# ============================================================
# 3) LOG
# ============================================================

logger = logging.getLogger("IndicadorTradingS1000")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    now_utc = datetime.now(timezone.utc)
    brt = (now_utc - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S BRT")
    utc = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    full = f"{utc} | {brt} | {msg}"
    if level == "info":
        logger.info(full)
    elif level == "warning":
        logger.warning(full)
    else:
        logger.error(full)


# ============================================================
# 4) TELEGRAM
# ============================================================

def send_telegram(message: str):
    try:
        if not TELEGRAM_TOKEN or not CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        log(f"Erro Telegram: {e}", "error")


# ============================================================
# 5) INDICADORES (FEATURES)
# ============================================================

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) < EMA_SLOW + 5:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    if "volume" not in df.columns:
        df["volume"] = 1

    df["mfi"] = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=MFI_PERIOD,
    ).money_flow_index()

    return df


def clamp_df(symbol: str):
    df = candles[symbol]
    if df is None or df.empty:
        return
    if len(df) > ML_MAX_SAMPLES:
        df = df.tail(ML_MAX_SAMPLES).reset_index(drop=True)
    candles[symbol] = df


def update_candles(symbol: str, new_rows: pd.DataFrame):
    if new_rows is None or new_rows.empty:
        return

    df = candles[symbol]
    if df is None or df.empty:
        df = new_rows.copy()
    else:
        df = pd.concat([df, new_rows], ignore_index=True)

    if "epoch" not in df.columns:
        return

    df["epoch"] = df["epoch"].astype(int)
    df = df.drop_duplicates(subset=["epoch"], keep="last")
    df = df.sort_values("epoch").reset_index(drop=True)

    if len(df) > ML_MAX_SAMPLES:
        df = df.tail(ML_MAX_SAMPLES).reset_index(drop=True)

    df = calcular_indicadores(df)
    candles[symbol] = df


# ============================================================
# 6) ML (CEREBRO)
# ============================================================

def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    if len(df) < 50:
        return None, None

    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna().copy()

    if "epoch" in df.columns:
        X = df.drop(columns=["future", "epoch"]).iloc[:-1]
    else:
        X = df.drop(columns=["future"]).iloc[:-1]

    y = df["future"].iloc[:-1]
    return X.tail(ML_MAX_SAMPLES), y.tail(ML_MAX_SAMPLES)


def train_ml(symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return

    df = candles[symbol]
    if df is None or df.empty or len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return

    X, y = build_ml_dataset(df)
    if X is None or y is None or len(X) < ML_MIN_TRAINED_SAMPLES:
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


def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ml_model_ready.get(symbol):
        return None
    model, cols = ml_models[symbol]
    try:
        vals = [float(row[c]) for c in cols]
        return float(model.predict_proba([vals])[0][1])
    except Exception:
        return None


# ============================================================
# 7) SINAIS (ML decide)
# ============================================================

def epoch_next_candle_open(epoch: int) -> int:
    g = GRANULARITY_SECONDS
    return ((epoch // g) + 1) * g


def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if df is None or df.empty or len(df) < EMA_SLOW + 20:
        return

    row = df.iloc[-1]
    epoch = int(row["epoch"])

    if last_signal_epoch[symbol] == epoch:
        return

    ml_prob = ml_predict(symbol, row)
    if ml_prob is None:
        return

    direction = "COMPRA" if ml_prob >= 0.5 else "VENDA"
    confidence = ml_prob if direction == "COMPRA" else (1.0 - ml_prob)

    if confidence < ML_CONF_THRESHOLD:
        return

    next_open = epoch_next_candle_open(epoch)

    entry_epoch = next_open
    msg_epoch = entry_epoch - (FINAL_ADVANCE_MINUTES * 60)

    entry_brt = datetime.fromtimestamp(entry_epoch, tz=timezone.utc) - timedelta(hours=3)
    msg_brt = datetime.fromtimestamp(msg_epoch, tz=timezone.utc) - timedelta(hours=3)

    ativo = symbol.replace("frx", "")

    msg = (
        f"📊 <b>ATIVO:</b> {ativo}\n"
        f"📈 <b>DIREÇÃO:</b> {direction}\n"
        f"🕐 <b>MSG:</b> {msg_brt.strftime('%H:%M')}\n"
        f"⏰ <b>ENTRADA:</b> {entry_brt.strftime('%H:%M')}\n"
        f"🤖 <b>ML:</b> {confidence*100:.0f}%"
    )

    send_telegram(msg)
    last_signal_epoch[symbol] = epoch
    log(f"{symbol} — sinal enviado {direction} ({confidence*100:.0f}%)")


# ============================================================
# 8) WEBSOCKET (1 conexão por símbolo + watchdog)
# ============================================================

async def ws_send(ws, payload: dict):
    await ws.send(json.dumps(payload))

def is_deriv_error_market_closed(err: dict) -> bool:
    try:
        return err.get("code") == "MarketIsClosed"
    except Exception:
        return False

def is_deriv_error_hard(err: dict) -> bool:
    try:
        return err.get("code") in ("UnrecognisedRequest", "WrongResponse")
    except Exception:
        return False


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

                req_hist = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1200,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles"
                }
                await ws_send(ws, req_hist)
                log(f"{symbol} Histórico solicitado 📥", "info")

                raw_hist = await ws.recv()
                hist_data = json.loads(raw_hist)

                if "error" in hist_data:
                    err = hist_data["error"]
                    log(f"{symbol} WS erro histórico: {err}", "error")
                    if is_deriv_error_market_closed(err) or is_deriv_error_hard(err):
                        log(f"{symbol} Mercado fechado/erro Deriv — aguardando {WS_SLEEP_AFTER_MARKET_CLOSED_SECONDS}s", "warning")
                        await asyncio.sleep(WS_SLEEP_AFTER_MARKET_CLOSED_SECONDS)
                    continue

                df_hist = pd.DataFrame(hist_data.get("candles", []))
                if df_hist.empty:
                    log(f"{symbol} Histórico vazio — reconectando 🔁", "warning")
                    continue

                update_candles(symbol, df_hist)
                train_ml(symbol)

                req_stream = {
                    "candles": symbol,
                    "granularity": GRANULARITY_SECONDS,
                    "subscribe": 1
                }
                await ws_send(ws, req_stream)
                log(f"{symbol} Stream (candles) ligado 🔴", "info")

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_CANDLE_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        log(f"{symbol} Sem candles por {WS_CANDLE_TIMEOUT_SECONDS}s — reconectando 🔁", "warning")
                        break

                    data = json.loads(raw)

                    if "error" in data:
                        err = data["error"]
                        log(f"{symbol} WS retornou erro: {err}", "error")
                        if is_deriv_error_market_closed(err) or is_deriv_error_hard(err):
                            log(f"{symbol} Mercado fechado/erro Deriv — aguardando {WS_SLEEP_AFTER_MARKET_CLOSED_SECONDS}s", "warning")
                            await asyncio.sleep(WS_SLEEP_AFTER_MARKET_CLOSED_SECONDS)
                        break

                    if "candles" in data:
                        df_new = pd.DataFrame(data["candles"])
                        if not df_new.empty:
                            update_candles(symbol, df_new)

                    df = candles[symbol]
                    if df is None or df.empty:
                        continue

                    current_epoch = int(df.iloc[-1]["epoch"])
                    if last_processed_epoch[symbol] == current_epoch:
                        continue

                    last_processed_epoch[symbol] = current_epoch
                    candle_counter[symbol] += 1

                    if candle_counter[symbol] % ML_RETRAIN_EVERY_CANDLES == 0:
                        train_ml(symbol)

                    avaliar_sinal(symbol)

        except Exception as e:
            log(f"{symbol} WS erro: {e}", "error")
            await asyncio.sleep(5)


# ============================================================
# 9) FLASK (KEEP ALIVE RENDER)
# ============================================================

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def health():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))


# ============================================================
# 10) MAIN
# ============================================================

async def main():
    send_telegram("🚀 BOT INICIADO — M5 ATIVO")
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
