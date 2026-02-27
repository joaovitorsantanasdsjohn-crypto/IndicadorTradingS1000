import asyncio
import json
import os
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple

import pandas as pd
import websockets
from flask import Flask
from dotenv import load_dotenv

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# 🔧 CONFIGURAÇÕES
# ============================================================
load_dotenv()

DERIV_TOKEN = os.getenv("DERIV_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID", "111022")
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = ["frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxAUDUSD"]
GRANULARITY_SECONDS = 900
HISTORY_COUNT = 2000

EMA_FAST, EMA_MID, EMA_SLOW = 9, 21, 55
RSI_PERIOD = 14
MFI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.3

ML_ENABLED = True and SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 500
ML_CONF_THRESHOLD = 0.70

TRADE_ENABLED = True
STAKE_AMOUNT = 1.0
MULTIPLIER = 100
TAKE_PROFIT = 0.4
STOP_LOSS = 0.2
TRADE_COOLDOWN_SECONDS = 25
DAILY_MAX_LOSS = 2.0

WATCHDOG_TIMEOUT = 1200  # 20 minutos


# ============================================================
# 📊 ESTADO GLOBAL
# ============================================================
candles: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in SYMBOLS}
ml_models: Dict[str, Tuple["RandomForestClassifier", list]] = {}
ml_model_ready: Dict[str, bool] = {s: False for s in SYMBOLS}

open_trades: Dict[str, Dict] = {s: {} for s in SYMBOLS}
last_trade_time: Dict[str, float] = {s: 0 for s in SYMBOLS}
last_activity_time: Dict[str, float] = {s: time.time() for s in SYMBOLS}

proposal_lock: Dict[str, bool] = {s: False for s in SYMBOLS}
proposal_lock_time: Dict[str, float] = {s: 0 for s in SYMBOLS}
symbol_locks: Dict[str, asyncio.Lock] = {s: asyncio.Lock() for s in SYMBOLS}

last_ml_train: Dict[str, float] = {s: 0 for s in SYMBOLS}

daily_pnl = 0.0
current_day = datetime.now(timezone.utc).date()
trading_paused = False
current_balance = 0.0

pending_proposals: Dict[int, dict] = {}
REQ_ID_SEQ = 1


# ============================================================
# 📝 LOG
# ============================================================
logger = logging.getLogger("DerivBot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.handlers.clear()
logger.addHandler(handler)

def log(msg, level="info"):
    getattr(logger, level)(msg)


# ============================================================
# 🔄 RESET DIÁRIO
# ============================================================
def check_daily_reset():
    global current_day, daily_pnl, trading_paused
    today = datetime.now(timezone.utc).date()
    if today != current_day:
        log("🔄 Reset diário executado")
        current_day = today
        daily_pnl = 0.0
        trading_paused = False


# ============================================================
# 🤖 ML (CORREÇÃO ESTRUTURAL — NÃO BLOQUEIA WS)
# ============================================================
def train_ml(symbol):

    if not ML_ENABLED:
        return

    if time.time() - last_ml_train[symbol] < 300:
        return

    df = candles[symbol]

    if len(df) < ML_MIN_TRAINED_SAMPLES:
        return

    X = df.select_dtypes(include=["number"]).dropna()

    if len(X) < 100:
        return

    y = (df["close"].shift(-1) > df["close"]).astype(int).dropna()
    X = X.iloc[:len(y)]

    model = RandomForestClassifier(n_estimators=120, max_depth=8)
    model.fit(X, y)

    ml_models[symbol] = (model, X.columns.tolist())
    ml_model_ready[symbol] = True
    last_ml_train[symbol] = time.time()


async def train_ml_background(symbol):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, train_ml, symbol)


def ml_predict(symbol, row):

    if not ml_model_ready[symbol]:
        return None

    model, cols = ml_models[symbol]

    try:
        values = [row[c] for c in cols]
    except:
        return None

    if any(pd.isna(v) for v in values):
        return None

    X = pd.DataFrame([values], columns=cols)
    return model.predict_proba(X)[0][1]


# ============================================================
# 🌐 LOOP WS
# ============================================================
async def ws_loop(symbol):

    while True:
        try:

            async with websockets.connect(
                WS_URL,
                ping_interval=60,
                ping_timeout=60,
                close_timeout=10,
                max_queue=None
            ) as ws:

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await ws.recv()

                async for raw in ws:

                    if time.time() - last_activity_time[symbol] > WATCHDOG_TIMEOUT:
                        log(f"{symbol} watchdog reiniciando conexão")
                        raise Exception("Watchdog timeout")

                    last_activity_time[symbol] = time.time()

                    data = json.loads(raw)

                    if "ohlc" in data:

                        df = candles[symbol]
                        df = pd.concat([df, pd.DataFrame([data["ohlc"]])]).tail(HISTORY_COUNT)
                        candles[symbol] = df

                        asyncio.create_task(train_ml_background(symbol))

                        if not ml_model_ready[symbol]:
                            continue

                        row = candles[symbol].iloc[-1]

                        prob = ml_predict(symbol, row)

                        if prob is None:
                            continue

        except Exception as e:
            log(f"{symbol} WS erro {e}", "error")
            await asyncio.sleep(3)


# ============================================================
# 🌍 FLASK
# ============================================================
app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200


def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))


# ============================================================
# 🚀 MAIN
# ============================================================
async def main():
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
