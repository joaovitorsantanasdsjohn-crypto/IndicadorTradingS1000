import asyncio
import json
import os
import time
import threading
import logging
from datetime import datetime, timedelta, timezone
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

SYMBOLS = ["frxEURUSD", "frxUSDJPY", "frxGBPUSD"]
GRANULARITY_SECONDS = 900
HISTORY_COUNT = 1200

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
MULTIPLIER = 50
TAKE_PROFIT = 0.4
STOP_LOSS = 0.2
TRADE_COOLDOWN_SECONDS = 15
DAILY_MAX_LOSS = 3.0

MARKET_CLOSED_WAIT = 60 * 10
LAST_TICK_TIME: Dict[str, float] = {s: time.time() for s in SYMBOLS}


# ============================================================
# 📊 ESTADO GLOBAL
# ============================================================
candles: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in SYMBOLS}
ml_models: Dict[str, Tuple["RandomForestClassifier", list]] = {}
ml_model_ready: Dict[str, bool] = {s: False for s in SYMBOLS}

open_trades: Dict[str, list] = {s: [] for s in SYMBOLS}
last_trade_time: Dict[str, float] = {s: 0 for s in SYMBOLS}

daily_pnl = 0.0
current_day = datetime.now(timezone.utc).date()
trading_paused = False
current_balance = 0.0


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
# 📈 INDICADORES
# ============================================================
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) < EMA_SLOW + 50:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()
    df["mfi"] = MFIIndicator(df["high"], df["low"], df["close"], df["volume"], MFI_PERIOD).money_flow_index()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    adx = ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD)
    df["adx"] = adx.adx()
    df["ret"] = df["close"].pct_change().fillna(0)
    return df


# ============================================================
# 🎯 NOVO ALVO ML — TP antes de SL
# ============================================================
def create_tp_sl_target(df: pd.DataFrame, horizon=12):
    df = df.copy()

    price_move_tp = TAKE_PROFIT / (MULTIPLIER * STAKE_AMOUNT)
    price_move_sl = STOP_LOSS / (MULTIPLIER * STAKE_AMOUNT)

    results = []

    for i in range(len(df) - horizon):
        entry = df.iloc[i]["close"]
        future = df.iloc[i+1:i+1+horizon]

        tp_up = entry + price_move_tp
        sl_down = entry - price_move_sl
        tp_down = entry - price_move_tp
        sl_up = entry + price_move_sl

        hit = None
        for _, row in future.iterrows():
            if row["high"] >= tp_up:
                hit = 1
                break
            if row["low"] <= sl_down:
                hit = 0
                break
            if row["low"] <= tp_down:
                hit = 1
                break
            if row["high"] >= sl_up:
                hit = 0
                break

        results.append(hit)

    df = df.iloc[:len(results)]
    df["target"] = results
    df = df.dropna()
    return df


def build_ml_dataset(df):
    df = create_tp_sl_target(df)
    if len(df) < 50:
        return None, None

    X = df.select_dtypes(include=["number"]).drop(columns=["target"], errors="ignore")
    y = df["target"]
    return X.tail(1000), y.tail(1000)


async def train_ml(symbol):
    if not ML_ENABLED:
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        return

    X, y = build_ml_dataset(df)
    if X is None:
        return

    model = RandomForestClassifier(n_estimators=120, max_depth=7)
    model.fit(X, y)

    ml_models[symbol] = (model, X.columns.tolist())
    ml_model_ready[symbol] = True
    log(f"{symbol} ML treinado (TP vs SL)")


def ml_predict(symbol, row):
    if not ml_model_ready[symbol]:
        return None
    model, cols = ml_models[symbol]
    X = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    return model.predict_proba(X)[0][1]


# ============================================================
# 🚦 FILTRO DE MERCADO
# ============================================================
def market_is_good(symbol):
    df = candles[symbol]
    if len(df) < 60:
        return False
    row = df.iloc[-1]
    if row.get("adx", 0) < 25:
        return False
    if row.get("bb_width", 0) < 0.0020:
        return False
    return True


# ============================================================
# 💰 TRADING
# ============================================================
def trade_already_open(symbol):
    return len(open_trades[symbol]) > 0


async def open_trade(ws, symbol, direction):
    global current_balance

    if trading_paused or current_balance < STAKE_AMOUNT:
        return
    if trade_already_open(symbol):
        return
    if time.time() - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
        return

    contract_type = "MULTUP" if direction == "UP" else "MULTDOWN"

    req = {
        "buy": 1,
        "price": STAKE_AMOUNT,
        "parameters": {
            "amount": STAKE_AMOUNT,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "symbol": symbol,
            "multiplier": MULTIPLIER,
            "limit_order": {"take_profit": TAKE_PROFIT, "stop_loss": STOP_LOSS}
        }
    }

    await ws.send(json.dumps(req))
    data = json.loads(await ws.recv())

    if "error" in data:
        return

    cid = data["buy"]["contract_id"]
    open_trades[symbol].append(cid)
    last_trade_time[symbol] = time.time()


# ============================================================
# 🌐 WEBSOCKET LOOP
# ============================================================
async def ws_loop(symbol):
    global daily_pnl, trading_paused, current_balance, current_day

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await ws.recv()

                await ws.send(json.dumps({"balance": 1, "subscribe": 1}))
                await ws.send(json.dumps({
                    "ticks_history": symbol,
                    "granularity": GRANULARITY_SECONDS,
                    "count": HISTORY_COUNT,
                    "end": "latest",
                    "style": "candles",
                    "subscribe": 1
                }))

                async for raw in ws:
                    data = json.loads(raw)

                    if "balance" in data:
                        current_balance = float(data["balance"]["balance"])

                    if "candles" in data:
                        df = pd.DataFrame(data["candles"])
                        df["date"] = pd.to_datetime(df["epoch"], unit="s")
                        df["volume"] = 1.0
                        candles[symbol] = calcular_indicadores(df)
                        await train_ml(symbol)

                    if "ohlc" in data:
                        new_row = pd.DataFrame([data["ohlc"]])
                        new_row["date"] = pd.to_datetime(new_row["epoch"], unit="s")
                        new_row["volume"] = 1.0
                        candles[symbol] = pd.concat([candles[symbol], new_row]).tail(HISTORY_COUNT)
                        candles[symbol] = calcular_indicadores(candles[symbol])
                        await train_ml(symbol)

                        if not market_is_good(symbol):
                            continue

                        row = candles[symbol].iloc[-1]
                        prob = ml_predict(symbol, row)
                        if prob is None or prob < ML_CONF_THRESHOLD:
                            continue

                        direction = "UP" if prob > 0.5 else "DOWN"
                        await open_trade(ws, symbol, direction)

        except Exception as e:
            log(f"{symbol} WS erro {e}", "error")
            await asyncio.sleep(10)


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
