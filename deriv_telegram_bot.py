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

# watchdog para reconectar se WS ficar "mudo"
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

BB_PERIOD = 20
BB_STD = 2.3

MFI_PERIOD = 14

# RSI
RSI_PERIOD = 14

# ---------------- ML ----------------

ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 300
ML_MAX_SAMPLES = 2000
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_N_ESTIMATORS = 60
ML_MAX_DEPTH = 5

# ---------------- Estado ----------------

candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {}
last_signal_epoch = {s: None for s in SYMBOLS}

# evita reprocessar candle repetido
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
    if len(df) < EMA_SLOW + 10:
        return df

    # garantir tipo numérico (vem como string da API às vezes)
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # MFI (volume neutro em Forex)
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

# ---------------- ML ----------------

def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # remove colunas que não são features
    X = df.drop(columns=["future", "epoch"]).iloc[:-1]
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

    # usa DataFrame para manter feature names
    X_row = pd.DataFrame([[float(row[c]) for c in cols]], columns=cols)
    return model.predict_proba(X_row)[0][1]

# ---------------- SETUP / CONTEXTO DE MERCADO ----------------

def has_market_context(row: pd.Series, direction: str) -> bool:
    """
    Só manda sinal quando existir SETUP com contexto,
    usando EMA + RSI + Bollinger + MFI.
    """

    close = float(row["close"])
    ema_fast = float(row["ema_fast"])
    ema_mid = float(row["ema_mid"])
    ema_slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    mfi = float(row["mfi"])
    bb_mid = float(row["bb_mid"])

    # COMPRA
    if direction == "COMPRA":
        if not (ema_fast > ema_mid > ema_slow):
            return False

        if not (35 <= rsi <= 55):
            return False

        # pullback abaixo/na BB mid
        if close > bb_mid:
            return False

        # não comprar saturado
        if mfi >= 75:
            return False

        return True

    # VENDA
    if direction == "VENDA":
        if not (ema_fast < ema_mid < ema_slow):
            return False

        if not (45 <= rsi <= 65):
            return False

        # pullback acima/na BB mid
        if close < bb_mid:
            return False

        # não vender saturado
        if mfi <= 25:
            return False

        return True

    return False

# ---------------- SINAL ----------------

def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 50:
        return

    row = df.iloc[-1]

    direction = "COMPRA" if row["ema_fast"] >= row["ema_mid"] else "VENDA"

    # 1) contexto
    if not has_market_context(row, direction):
        return

    # 2) ML filtra probabilidade
    ml_prob = ml_predict(symbol, row)
    if ml_prob is None or ml_prob < ML_CONF_THRESHOLD:
        return

    epoch = int(row["epoch"])

    # evita duplicar sinal na mesma vela
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
    log(f"{symbol} — sinal enviado {direction} (ML {ml_prob*100:.0f}%)")

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

                # ✅ HISTÓRICO + STREAM em 1 request (o jeito correto!)
                req = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1200,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles",
                    "subscribe": 1
                }

                await ws.send(json.dumps(req))
                log(f"{symbol} Histórico + stream solicitado 📥", "info")

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

                    if "candles" not in data:
                        continue

                    incoming = data["candles"]

                    # primeira carga: vem lista grande de candles
                    if isinstance(incoming, list) and len(incoming) > 1:
                        df = pd.DataFrame(incoming)
                        if df.empty:
                            continue
                        df = calcular_indicadores(df)
                        candles[symbol] = df

                        # processa o último candle do histórico uma vez
                        try:
                            current_epoch = int(df.iloc[-1]["epoch"])
                        except Exception:
                            continue

                        if last_processed_epoch[symbol] != current_epoch:
                            last_processed_epoch[symbol] = current_epoch
                            train_ml(symbol)
                            avaliar_sinal(symbol)

                        continue

                    # stream: geralmente vem 1 candle
                    if isinstance(incoming, list) and len(incoming) == 1:
                        new_row = incoming[0]
                    else:
                        # fallback caso a API retorne formato diferente
                        new_row = incoming

                    df = candles.get(symbol)
                    if df is None or df.empty:
                        # se por algum motivo o histórico ainda não carregou
                        continue

                    try:
                        last_epoch = int(df.iloc[-1]["epoch"])
                        new_epoch = int(new_row["epoch"])
                    except Exception:
                        continue

                    # atualiza candle atual ou adiciona novo
                    if new_epoch == last_epoch:
                        for k, v in new_row.items():
                            if k in df.columns:
                                df.at[df.index[-1], k] = v
                            else:
                                # caso apareça coluna nova
                                df[k] = None
                                df.at[df.index[-1], k] = v
                    else:
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                        # limita tamanho para não crescer infinito
                        if len(df) > 2500:
                            df = df.iloc[-2500:].reset_index(drop=True)

                    df = calcular_indicadores(df)
                    candles[symbol] = df

                    # processar só quando epoch muda
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

# ---------------- Flask ----------------

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
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
