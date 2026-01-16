# Indicador Trading S1000

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time
import logging
from flask import Flask
import threading
from typing import Optional

# =========================
# 1) ML availability
# =========================

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# =========================
# 2) Inicialização
# =========================

load_dotenv()

# =========================
# 3) Configurações (ENV)
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID", "111022")

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

# envia mensagem X minutos antes da entrada
SIGNAL_ADVANCE_MINUTES = int(os.getenv("SIGNAL_ADVANCE_MINUTES", "5"))

WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

WS_CANDLE_TIMEOUT_SECONDS = int(os.getenv("WS_CANDLE_TIMEOUT_SECONDS", "300"))

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

# =========================
# 4) Parâmetros Técnicos (features)
# =========================

EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_MID  = int(os.getenv("EMA_MID", "21"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "34"))

BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "2.3"))

MFI_PERIOD = int(os.getenv("MFI_PERIOD", "14"))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

# RSI filtro opcional
USE_RSI_FILTER = os.getenv("USE_RSI_FILTER", "0") == "1"
RSI_MIN = float(os.getenv("RSI_MIN", "0"))
RSI_MAX = float(os.getenv("RSI_MAX", "100"))

# =========================
# 5) Parâmetros ML
# =========================

ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "300"))
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "2000"))
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "60"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "5"))

# =========================
# 6) Estado (memória)
# =========================

candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {}
last_signal_epoch = {s: None for s in SYMBOLS}
last_processed_epoch = {s: None for s in SYMBOLS}

# =========================
# 7) Logging
# =========================

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

# =========================
# 8) Telegram
# =========================

def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        log(f"Erro Telegram: {e}", "error")

# =========================
# 9) Indicadores (features para ML)
# =========================

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) < EMA_SLOW + 10:
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
        window=MFI_PERIOD
    ).money_flow_index()

    return df

# =========================
# 10) ML - treino e predição
# =========================

def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    if "epoch" not in df.columns:
        return None, None
    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df.drop(columns=["future","epoch"]).iloc[:-1]
    y = df["future"].iloc[:-1]
    return X.tail(ML_MAX_SAMPLES), y.tail(ML_MAX_SAMPLES)

def train_ml(symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
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

# =========================
# 11) Entrada alinhada com vela M5
# =========================

def get_next_candle_time_from_epoch(epoch: int) -> datetime:
    """
    epoch = início da vela atual.
    Retorna horário (BRT) do início da PRÓXIMA vela (alinhado em múltiplos de 5min).
    """
    candle_close_epoch = epoch + GRANULARITY_SECONDS
    dt_brt = datetime.utcfromtimestamp(candle_close_epoch) - timedelta(hours=3)
    return dt_brt

# =========================
# 12) Sinal (ML decide)
# =========================

def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 50:
        return

    row = df.iloc[-1]

    ml_prob = ml_predict(symbol, row)
    if ml_prob is None or ml_prob < ML_CONF_THRESHOLD:
        return

    if USE_RSI_FILTER:
        try:
            rsi_val = float(row["rsi"])
            if not (RSI_MIN <= rsi_val <= RSI_MAX):
                return
        except Exception:
            return

    epoch = int(row["epoch"])

    if last_signal_epoch[symbol] == epoch:
        return
    last_signal_epoch[symbol] = epoch

    direction = "COMPRA" if ml_prob >= 0.5 else "VENDA"

    candle_entry_time = get_next_candle_time_from_epoch(epoch)
    msg_time = candle_entry_time - timedelta(minutes=SIGNAL_ADVANCE_MINUTES)

    ativo = symbol.replace("frx", "")

    msg = (
        f"📊 <b>ATIVO:</b> {ativo}\n"
        f"📈 <b>DIREÇÃO:</b> {direction}\n"
        f"🕐 <b>MENSAGEM:</b> {msg_time.strftime('%H:%M')}\n"
        f"⏰ <b>ENTRADA:</b> {candle_entry_time.strftime('%H:%M')}\n"
        f"🤖 <b>ML:</b> {ml_prob*100:.0f}%"
    )

    send_telegram(msg)
    log(f"{symbol} — sinal enviado {direction} (ML {ml_prob*100:.0f}%)")

# =========================
# 13) WebSocket por ativo
# =========================

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
                await ws.send(json.dumps(req_hist))
                log(f"{symbol} Histórico solicitado 📥", "info")

                hist_raw = await ws.recv()
                hist_data = json.loads(hist_raw)

                if "error" in hist_data:
                    log(f"{symbol} WS retornou erro: {hist_data.get('error')}", "error")
                    await ws.close()
                    continue

                df = pd.DataFrame(hist_data.get("candles", []))
                if df.empty:
                    log(f"{symbol} Histórico vazio — reconectando 🔁", "warning")
                    await ws.close()
                    continue

                df = calcular_indicadores(df)
                candles[symbol] = df

                train_ml(symbol)

                req_stream = {
                    "candles_subscribe": 1,
                    "symbol": symbol,
                    "granularity": GRANULARITY_SECONDS
                }
                await ws.send(json.dumps(req_stream))
                log(f"{symbol} Stream (candles) ligado 🔴", "info")

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_CANDLE_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        log(f"{symbol} Sem candles por {WS_CANDLE_TIMEOUT_SECONDS}s — reconectando 🔁", "warning")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    data = json.loads(raw)

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

                        last_epoch = int(df.iloc[-1]["epoch"])
                        new_epoch = int(new_row["epoch"])

                        if new_epoch == last_epoch:
                            for k, v in new_row.items():
                                df.at[df.index[-1], k] = v
                        else:
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                        df = calcular_indicadores(df)
                        candles[symbol] = df

                        try:
                            current_epoch = int(df.iloc[-1]["epoch"])
                        except Exception:
                            continue

                        if last_processed_epoch[symbol] != current_epoch:
                            last_processed_epoch[symbol] = current_epoch
                            train_ml(symbol)
                            avaliar_sinal(symbol)

        except Exception as e:
            log(f"{symbol} WS erro: {e}", "error")
            await asyncio.sleep(5)

# =========================
# 14) Flask (health)
# =========================

app = Flask(__name__)

@app.route("/", methods=["GET","HEAD"])
def health():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

# =========================
# 15) Main
# =========================

async def main():
    send_telegram("🚀 BOT INICIADO — M5 ATIVO")
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
