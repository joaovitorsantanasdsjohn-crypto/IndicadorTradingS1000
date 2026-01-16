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

SIGNAL_ADVANCE_MINUTES = int(os.getenv("SIGNAL_ADVANCE_MINUTES", "5"))

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

EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_MID = int(os.getenv("EMA_MID", "21"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "34"))

BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "2.3"))

MFI_PERIOD = int(os.getenv("MFI_PERIOD", "14"))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

# ---------------- ML ----------------

ML_ENABLED = SKLEARN_AVAILABLE

# mínimo de candles pra treinar 1x (você pode mudar se quiser)
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "300"))

# ✅ REDUZIDO PRA CABER NO RENDER FREE (menos RAM)
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "1200"))

ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))

# ✅ REDUZIDO PRA CABER NO RENDER FREE (menos RAM)
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "30"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "5"))

# ✅ MUITO IMPORTANTE: evita treinar ML em TODO candle (isso explode RAM/CPU)
# default: treina a cada 6 candles (30min no M5)
ML_TRAIN_EVERY_N_CANDLES = int(os.getenv("ML_TRAIN_EVERY_N_CANDLES", "6"))

# ---------------- Estado ----------------

candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {}

# evita reprocessar candle repetido
last_processed_epoch = {s: None for s in SYMBOLS}

# ✅ controla quando foi o ultimo treino do ML (por candle index)
last_ml_train_idx = {s: None for s in SYMBOLS}

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
    X = df.drop(columns=["future", "epoch"]).iloc[:-1]
    y = df["future"].iloc[:-1]

    # ✅ limita dataset pra reduzir RAM
    X = X.tail(ML_MAX_SAMPLES)
    y = y.tail(ML_MAX_SAMPLES)

    return X, y

def should_train_ml(symbol: str, df: pd.DataFrame) -> bool:
    """
    ✅ treina o ML apenas a cada N candles
    para não estourar RAM / CPU no Render Free.
    """
    idx = len(df) - 1
    last_idx = last_ml_train_idx.get(symbol)

    if last_idx is None:
        return True

    if (idx - last_idx) >= ML_TRAIN_EVERY_N_CANDLES:
        return True

    return False

def train_ml(symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return

    # ✅ treino controlado (não treina toda hora)
    if not should_train_ml(symbol, df):
        return

    try:
        X, y = build_ml_dataset(df)

        if len(X) < 50:
            ml_model_ready[symbol] = False
            return

        model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            random_state=42,
            n_jobs=1  # ✅ evita threads demais (reduz RAM)
        )
        model.fit(X, y)

        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True
        last_ml_train_idx[symbol] = len(df) - 1

        log(f"{symbol} ML treinado ✅ (samples={len(X)})", "info")

    except Exception as e:
        ml_model_ready[symbol] = False
        log(f"{symbol} erro treino ML: {e}", "error")

def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ML_ENABLED:
        return None
    if not ml_model_ready.get(symbol):
        return None

    model, cols = ml_models[symbol]
    try:
        vals = [float(row[c]) for c in cols]
        return float(model.predict_proba([vals])[0][1])
    except Exception:
        return None

# ---------------- SETUP / CONTEXTO DE MERCADO ----------------

def has_market_context(row: pd.Series, direction: str) -> bool:
    """
    ✅ indicadores criam cenário: EMA + RSI + Bollinger + MFI
    ✅ ML é filtro final (rei do bot)
    """

    close = float(row["close"])
    ema_fast = float(row["ema_fast"])
    ema_mid = float(row["ema_mid"])
    ema_slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    mfi = float(row["mfi"])
    bb_mid = float(row["bb_mid"])

    # COMPRA (pullback em tendência)
    if direction == "COMPRA":
        if not (ema_fast > ema_mid > ema_slow):
            return False

        # RSI não pode estar super esticado nem morto
        if not (35 <= rsi <= 55):
            return False

        # preço abaixo ou perto do meio (pullback)
        if close > bb_mid:
            return False

        # MFI evita compra em saturação
        if mfi >= 75:
            return False

        return True

    # VENDA (pullback em tendência)
    if direction == "VENDA":
        if not (ema_fast < ema_mid < ema_slow):
            return False

        if not (45 <= rsi <= 65):
            return False

        if close < bb_mid:
            return False

        if mfi <= 25:
            return False

        return True

    return False

# ---------------- SINAL ----------------

def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 80:
        return

    row = df.iloc[-1]

    # direção base
    direction = "COMPRA" if row["ema_fast"] >= row["ema_mid"] else "VENDA"

    # 1) contexto por indicadores
    if not has_market_context(row, direction):
        return

    # 2) ML aprova probabilidade
    ml_prob = ml_predict(symbol, row)
    if ml_prob is None or ml_prob < ML_CONF_THRESHOLD:
        return

    epoch = int(row["epoch"])

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
    log(f"{symbol} — sinal enviado {direction} (ML {ml_prob*100:.0f}%)", "info")

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

                # ✅ Subscribe correto: Deriv exige inteiro (1), não boolean
                req_hist = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1200,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles",
                    "subscribe": 1  # ✅ CORRETO
                }

                await ws.send(json.dumps(req_hist))
                log(f"{symbol} Histórico + stream solicitado 📥", "info")

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=WS_CANDLE_TIMEOUT_SECONDS
                        )
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

                    # Deriv envia candles no primeiro pacote e depois updates
                    if "candles" in data:
                        df = pd.DataFrame(data["candles"])
                        df = calcular_indicadores(df)
                        candles[symbol] = df

                    elif "ohlc" in data:
                        # updates (stream)
                        o = data["ohlc"]
                        df = candles[symbol]

                        new_row = {
                            "epoch": int(o["open_time"]),
                            "open": float(o["open"]),
                            "high": float(o["high"]),
                            "low": float(o["low"]),
                            "close": float(o["close"]),
                        }

                        # se vier volume (normalmente não vem no Forex)
                        if "volume" in o:
                            new_row["volume"] = float(o["volume"])

                        if len(df) == 0:
                            df = pd.DataFrame([new_row])
                        else:
                            last_epoch = int(df.iloc[-1]["epoch"])
                            if new_row["epoch"] == last_epoch:
                                df.iloc[-1] = pd.Series(new_row)
                            else:
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                        # mantém tamanho controlado
                        if len(df) > 1500:
                            df = df.tail(1500).reset_index(drop=True)

                        df = calcular_indicadores(df)
                        candles[symbol] = df

                    # processar só quando epoch muda
                    try:
                        current_epoch = int(candles[symbol].iloc[-1]["epoch"])
                    except Exception:
                        continue

                    if last_processed_epoch[symbol] != current_epoch:
                        last_processed_epoch[symbol] = current_epoch

                        # ✅ treina ML de forma controlada pra não estourar RAM
                        train_ml(symbol)

                        # avalia sinal
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
