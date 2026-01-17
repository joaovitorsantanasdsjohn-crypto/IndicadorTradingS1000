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
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator

try:
    from sklearn.ensemble import RandomForestClassifier
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
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxGBPAUD", "frxGBPCAD", "frxAUDNZD", "frxEURCAD"
]

CANDLE_INTERVAL_MINUTES = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL_MINUTES * 60

FINAL_ADVANCE_MINUTES = int(os.getenv("FINAL_ADVANCE_MINUTES", "5"))

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

ML_ENABLED = bool(int(os.getenv("ML_ENABLED", "1"))) and SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "300"))
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "2000"))
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))

ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "60"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "5"))
ML_TRAIN_EVERY_N_CANDLES = int(os.getenv("ML_TRAIN_EVERY_N_CANDLES", "3"))

MIN_SECONDS_BETWEEN_SIGNALS = int(os.getenv("MIN_SECONDS_BETWEEN_SIGNALS", "3"))
STARTUP_STAGGER_MAX_SECONDS = int(os.getenv("STARTUP_STAGGER_MAX_SECONDS", "10"))

# ✅ MELHORIA PEDIDA
# Se o mercado estiver fechado: espera 30 minutos antes de tentar reconectar
MARKET_CLOSED_RECONNECT_WAIT_SECONDS = int(os.getenv("MARKET_CLOSED_RECONNECT_WAIT_SECONDS", "1800"))  # 30 min


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

    if level == "info":
        logger.info(full)
    elif level == "warning":
        logger.warning(full)
    else:
        logger.error(full)


# ============================================================
# ✅ BLOCO 4 — TELEGRAM
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
# ✅ BLOCO 5 — INDICADORES (SOMENTE FEATURES PARA ML)
# ============================================================

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    if len(df) < EMA_SLOW + 10:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()
    df["rsi"] = df["rsi"].clip(lower=RSI_MIN, upper=RSI_MAX)

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
        window=MFI_PERIOD
    ).money_flow_index()

    return df


# ============================================================
# ✅ BLOCO 6 — MACHINE LEARNING (CÉREBRO DO BOT)
# ============================================================

def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    if "epoch" not in df.columns:
        return None, None

    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)

    drop_cols = {"future", "epoch"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).iloc[:-1]
    y = df["future"].iloc[:-1]

    if len(X) <= 10:
        return None, None

    X = X.tail(ML_MAX_SAMPLES)
    y = y.tail(ML_MAX_SAMPLES)

    return X, y


async def train_ml_async(symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return

    X, y = build_ml_dataset(df)
    if X is None or y is None:
        ml_model_ready[symbol] = False
        return

    def _fit():
        model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            random_state=42
        )
        model.fit(X, y)
        return model, X.columns.tolist()

    try:
        model, cols = await asyncio.to_thread(_fit)
        ml_models[symbol] = (model, cols)
        ml_model_ready[symbol] = True
    except Exception as e:
        ml_model_ready[symbol] = False
        log(f"{symbol} ML treino falhou: {e}", "warning")


def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ML_ENABLED:
        return None

    if not ml_model_ready.get(symbol):
        return None

    if symbol not in ml_models:
        return None

    try:
        model, cols = ml_models[symbol]
        vals = [float(row[c]) for c in cols]
        prob_buy = model.predict_proba([vals])[0][1]
        return float(prob_buy)
    except Exception:
        return None


# ============================================================
# ✅ BLOCO 7 — SINAIS (100% ML)
# ============================================================

def floor_to_granularity(ts_epoch: int, gran_seconds: int) -> int:
    return (ts_epoch // gran_seconds) * gran_seconds


def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 50:
        return

    row = df.iloc[-1]
    if "epoch" not in row:
        return

    epoch = int(row["epoch"])
    candle_open_epoch = floor_to_granularity(epoch, GRANULARITY_SECONDS)

    if last_signal_epoch[symbol] == candle_open_epoch:
        return

    now = time.time()
    if (now - last_signal_time[symbol]) < MIN_SECONDS_BETWEEN_SIGNALS:
        return

    ml_prob_buy = ml_predict(symbol, row)
    if ml_prob_buy is None:
        return

    direction = "COMPRA" if ml_prob_buy >= 0.5 else "VENDA"
    confidence = ml_prob_buy if direction == "COMPRA" else (1.0 - ml_prob_buy)

    if confidence < ML_CONF_THRESHOLD:
        return

    next_candle_open = candle_open_epoch + GRANULARITY_SECONDS

    entry_time_brt = datetime.fromtimestamp(next_candle_open, tz=timezone.utc) - timedelta(hours=3)
    notify_time_brt = entry_time_brt - timedelta(minutes=FINAL_ADVANCE_MINUTES)

    ativo = symbol.replace("frx", "")

    msg = (
        f"🚀 <b>ATIVO:</b> {ativo}\n"
        f"📌 <b>DIREÇÃO:</b> {direction}\n"
        f"🕒 <b>MENSAGEM:</b> {notify_time_brt.strftime('%H:%M')}\n"
        f"⏰ <b>ENTRADA:</b> {entry_time_brt.strftime('%H:%M')}\n"
        f"🤖 <b>ML:</b> {confidence*100:.0f}%"
    )

    send_telegram(msg)

    last_signal_time[symbol] = now
    last_signal_epoch[symbol] = candle_open_epoch

    log(f"{symbol} — sinal enviado {direction} (ML {confidence*100:.0f}%)")


# ============================================================
# ✅ BLOCO 8 — WEBSOCKET (HISTÓRICO + STREAM)
# ============================================================

async def deriv_authorize(ws):
    if not DERIV_TOKEN:
        return

    req = {"authorize": DERIV_TOKEN}
    await ws.send(json.dumps(req))
    raw = await ws.recv()
    data = json.loads(raw)

    if "error" in data:
        raise RuntimeError(f"Authorize error: {data['error']}")


async def request_history(ws, symbol: str) -> pd.DataFrame:
    req_hist = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": HISTORY_COUNT,
        "end": "latest",
        "granularity": GRANULARITY_SECONDS,
        "style": "candles"
    }

    await ws.send(json.dumps(req_hist))
    log(f"{symbol} Histórico solicitado 📥", "info")

    raw = await ws.recv()
    data = json.loads(raw)

    if "error" in data:
        raise RuntimeError(str(data.get("error")))

    df = pd.DataFrame(data.get("candles", []))
    if df.empty:
        raise RuntimeError("Histórico vazio")

    return df


async def subscribe_candles(ws, symbol: str):
    req_sub = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 1,
        "end": "latest",
        "granularity": GRANULARITY_SECONDS,
        "style": "candles",
        "subscribe": 1
    }

    await ws.send(json.dumps(req_sub))
    log(f"{symbol} Stream (candles) ligado 🔴", "info")


def df_trim(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_CANDLES_IN_RAM:
        return df
    return df.tail(MAX_CANDLES_IN_RAM).reset_index(drop=True)


def _is_market_closed_payload(data: dict) -> bool:
    """
    Detecta MarketIsClosed no payload JSON do WS.
    """
    try:
        if not isinstance(data, dict):
            return False
        err = data.get("error")
        if isinstance(err, dict) and err.get("code") == "MarketIsClosed":
            return True
        return False
    except Exception:
        return False


def _is_market_closed_exception(e: Exception) -> bool:
    """
    Detecta MarketIsClosed quando vem em texto/exception.
    """
    try:
        s = str(e)
        return ("MarketIsClosed" in s)
    except Exception:
        return False


async def ws_loop(symbol: str):
    backoff = 2
    max_backoff = 90

    await asyncio.sleep(random.uniform(0.0, float(STARTUP_STAGGER_MAX_SECONDS)))

    while True:
        try:
            log(f"{symbol} WS conectando...", "info")

            async with websockets.connect(
                WS_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
                open_timeout=WS_OPEN_TIMEOUT,
                close_timeout=10,
                max_queue=32
            ) as ws:
                log(f"{symbol} WS conectado ✅", "info")

                try:
                    await deriv_authorize(ws)
                except Exception as e:
                    log(f"{symbol} authorize falhou: {e}", "warning")

                df = await request_history(ws, symbol)
                df = calcular_indicadores(df)
                df = df_trim(df)
                candles[symbol] = df

                if ML_ENABLED:
                    await train_ml_async(symbol)

                await subscribe_candles(ws, symbol)

                backoff = 2

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_CANDLE_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        log(f"{symbol} Watchdog: sem candle por {WS_CANDLE_TIMEOUT_SECONDS}s — reconectando...", "warning")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    data = json.loads(raw)

                    # ✅ CASO 1: erro MarketIsClosed vindo no payload
                    if _is_market_closed_payload(data):
                        log(
                            f"{symbol} Mercado fechado (MarketIsClosed) — aguardando {MARKET_CLOSED_RECONNECT_WAIT_SECONDS}s para reconectar...",
                            "warning"
                        )
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        await asyncio.sleep(MARKET_CLOSED_RECONNECT_WAIT_SECONDS)
                        break

                    # Outros erros do WS
                    if "error" in data:
                        log(f"{symbol} WS retornou erro: {data.get('error')}", "error")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    if "candles" in data:
                        new_row = data["candles"][0]
                        df = candles[symbol]

                        if df.empty:
                            df = pd.DataFrame([new_row])
                        else:
                            try:
                                last_epoch = int(df.iloc[-1]["epoch"])
                                new_epoch = int(new_row["epoch"])
                            except Exception:
                                last_epoch = None
                                new_epoch = None

                            if last_epoch is not None and new_epoch == last_epoch:
                                for k, v in new_row.items():
                                    df.at[df.index[-1], k] = v
                            else:
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                        df = df_trim(df)
                        df = calcular_indicadores(df)
                        candles[symbol] = df

                        try:
                            current_epoch = int(df.iloc[-1]["epoch"])
                        except Exception:
                            continue

                        if last_processed_epoch[symbol] != current_epoch:
                            last_processed_epoch[symbol] = current_epoch
                            candle_counter[symbol] += 1

                            if ML_ENABLED and (candle_counter[symbol] % ML_TRAIN_EVERY_N_CANDLES == 0):
                                asyncio.create_task(train_ml_async(symbol))

                            avaliar_sinal(symbol)

        except Exception as e:
            # ✅ CASO 2: MarketIsClosed vindo como exception/texto
            if _is_market_closed_exception(e):
                log(
                    f"{symbol} Mercado fechado (MarketIsClosed exception) — aguardando {MARKET_CLOSED_RECONNECT_WAIT_SECONDS}s para reconectar...",
                    "warning"
                )
                await asyncio.sleep(MARKET_CLOSED_RECONNECT_WAIT_SECONDS)
                backoff = 2
                continue

            msg = str(e)

            if "UnrecognisedRequest" in msg or "WrongResponse" in msg:
                log(f"{symbol} WS request inválido/erro Deriv: {e}", "error")
            else:
                log(f"{symbol} WS erro: {e}", "error")

            sleep_s = backoff + random.uniform(0.0, 1.5)
            await asyncio.sleep(sleep_s)
            backoff = min(max_backoff, backoff * 2)


# ============================================================
# ✅ BLOCO 9 — FLASK HEALTHCHECK (RENDER)
# ============================================================

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def health():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))


# ============================================================
# ✅ BLOCO 10 — MAIN
# ============================================================

async def main():
    send_telegram("🚀 BOT INICIADO — M5 ATIVO")
    tasks = [ws_loop(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
