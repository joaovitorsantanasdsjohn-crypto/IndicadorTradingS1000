# Indicador Trading S1000

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator   # <<< ADICIONADO
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

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SIGNAL_ADVANCE_MINUTES = 5

WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# <<< watchdog para reconectar se WS ficar "mudo"
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

RSI_BUY_MAX = 45
RSI_SELL_MIN = 55

BB_PERIOD = 20
BB_STD = 2.3

MFI_PERIOD = 14   # <<< ADICIONADO (configurável)

# ---------------- ML ----------------

ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 300
ML_MAX_SAMPLES = 2000
ML_CONF_THRESHOLD = 0.55
ML_N_ESTIMATORS = 60
ML_MAX_DEPTH = 5

# ---------------- CONTROLE DE SINAIS (ANTI-SPAM) ----------------
# <<< ADICIONADO: controla frequência e evita reversão rápida

SIGNAL_COOLDOWN_SECONDS = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "900"))
# 900s = 15 minutos (por ativo)

SIGNAL_MIN_CANDLES_BETWEEN = int(os.getenv("SIGNAL_MIN_CANDLES_BETWEEN", "2"))
# precisa pelo menos 2 candles fechados antes de liberar novo sinal no mesmo ativo

SIGNAL_ANTI_REVERSAL_CANDLES = int(os.getenv("SIGNAL_ANTI_REVERSAL_CANDLES", "3"))
# se mandou COMPRA, não pode mandar VENDA logo depois (e vice-versa) por X candles

WARMUP_CANDLES_AFTER_BOOT = int(os.getenv("WARMUP_CANDLES_AFTER_BOOT", "30"))
# evita sinal "cru" logo após restart (30 candles = 150 min em M5)

# ---------------- Estado ----------------

candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {}
last_signal_epoch = {s: None for s in SYMBOLS}

# controla treino/avaliação só por candle novo
last_trained_epoch = {s: None for s in SYMBOLS}

# <<< ADICIONADO: anti-spam por ativo
last_signal_time = {s: 0.0 for s in SYMBOLS}
last_signal_direction = {s: None for s in SYMBOLS}
last_signal_index = {s: None for s in SYMBOLS}

# <<< ADICIONADO: warmup global após boot (não bloqueia ML, só evita sinais ruins no boot)
boot_time = time.time()

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
    if len(df) < EMA_SLOW:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # ---------------- MFI (somente CONTEXTO) ----------------
    if "volume" not in df.columns:
        df["volume"] = 1  # <<< Volume neutro para Forex (não bloqueia nada)

    df["mfi"] = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=MFI_PERIOD
    ).money_flow_index()
    # --------------------------------------------------------

    return df

# ---------------- ML ----------------

def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()
    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df.drop(columns=["future","epoch"]).iloc[:-1]
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
    vals = [float(row[c]) for c in cols]
    return model.predict_proba([vals])[0][1]

# ---------------- FILTROS ANTI-SPAM ----------------

def can_send_signal(symbol: str, direction: str, epoch: int) -> bool:
    """
    <<< ADICIONADO: impede spam, sinal duplicado, e reversão rápida.
    NÃO altera o ML (ML continua sendo o único filtro da assertividade).
    """
    now = time.time()

    # 1) cooldown por tempo
    if (now - last_signal_time[symbol]) < SIGNAL_COOLDOWN_SECONDS:
        return False

    # 2) 1 sinal por candle (por ativo)
    if last_signal_epoch[symbol] == epoch:
        return False

    # 3) precisa passar X candles antes de mandar outro sinal no mesmo ativo
    df = candles[symbol]
    if len(df) > 5:
        idx = len(df) - 1
        last_idx = last_signal_index[symbol]
        if last_idx is not None and (idx - last_idx) < SIGNAL_MIN_CANDLES_BETWEEN:
            return False

    # 4) anti reversão rápida
    prev_dir = last_signal_direction[symbol]
    if prev_dir is not None and prev_dir != direction:
        df = candles[symbol]
        if len(df) > 5:
            idx = len(df) - 1
            last_idx = last_signal_index[symbol]
            if last_idx is not None and (idx - last_idx) < SIGNAL_ANTI_REVERSAL_CANDLES:
                return False

    # 5) warmup após boot (evita sinais "crus" do restart)
    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES + WARMUP_CANDLES_AFTER_BOOT:
        # aqui é pra impedir aquele spam inicial e evitar que você opere sinais ruins do boot
        return False

    return True

# ---------------- SINAL ----------------

def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 5:
        return

    row = df.iloc[-1]

    # direção é apenas contexto
    direction = "COMPRA" if row["ema_fast"] >= row["ema_mid"] else "VENDA"

    ml_prob = ml_predict(symbol, row)
    if ml_prob is None or ml_prob < ML_CONF_THRESHOLD:
        return

    epoch = int(row["epoch"])

    # <<< ADICIONADO: anti-spam inteligente (sem mexer na lógica do ML)
    if not can_send_signal(symbol, direction, epoch):
        return

    # marcações de envio
    last_signal_epoch[symbol] = epoch
    last_signal_time[symbol] = time.time()
    last_signal_direction[symbol] = direction
    last_signal_index[symbol] = len(df) - 1

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
    log(f"{symbol} — sinal enviado {direction}")

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

                req_hist = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1200,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles",
                    "subscribe": 1
                }
                await ws.send(json.dumps(req_hist))
                log(f"{symbol} Histórico solicitado 📥", "info")

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=WS_CANDLE_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        log(
                            f"{symbol} Sem candles por {WS_CANDLE_TIMEOUT_SECONDS}s — reconectando 🔁",
                            "warning"
                        )
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    data = json.loads(raw)

                    # resposta de erro do WS
                    if "error" in data:
                        log(f"{symbol} WS erro retorno: {data.get('error')}", "error")
                        break

                    if "candles" in data:
                        df = pd.DataFrame(data["candles"])
                        candles[symbol] = calcular_indicadores(df)

                        try:
                            current_epoch = int(candles[symbol].iloc[-1]["epoch"])
                        except Exception:
                            continue

                        # treinar e avaliar SOMENTE quando mudar candle (epoch novo)
                        if last_trained_epoch[symbol] != current_epoch:
                            last_trained_epoch[symbol] = current_epoch
                            train_ml(symbol)
                            avaliar_sinal(symbol)

        except Exception as e:
            log(f"{symbol} WS erro: {e}", "error")
            await asyncio.sleep(5)

# ---------------- Flask ----------------

app = Flask(__name__)

@app.route("/", methods=["GET","HEAD"])
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
