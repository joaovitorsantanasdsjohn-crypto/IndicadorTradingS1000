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
    from sklearn.utils import resample
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
ML_MIN_TRAINED_SAMPLES = 600
ML_CONF_THRESHOLD = 0.70

TRADE_ENABLED = True
STAKE_AMOUNT = 1.0
MULTIPLIER = 100
TAKE_PROFIT = 0.4
STOP_LOSS = 0.2
TRADE_COOLDOWN_SECONDS = 25
DAILY_MAX_LOSS = 2.0

WATCHDOG_TIMEOUT = 1200


# ============================================================
# 📊 ESTADO GLOBAL
# ============================================================

candles: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in SYMBOLS}

ml_models: Dict[str, Tuple["RandomForestClassifier", list]] = {}
ml_model_ready: Dict[str, bool] = {s: False for s in SYMBOLS}
last_ml_train: Dict[str, float] = {s: 0 for s in SYMBOLS}

open_trades: Dict[str, Dict] = {s: {} for s in SYMBOLS}

pending_buy_symbol: Dict[str, bool] = {s: False for s in SYMBOLS}

last_trade_time: Dict[str, float] = {s: 0 for s in SYMBOLS}
last_activity_time: Dict[str, float] = {s: time.time() for s in SYMBOLS}

proposal_lock: Dict[str, bool] = {s: False for s in SYMBOLS}
symbol_locks: Dict[str, asyncio.Lock] = {s: asyncio.Lock() for s in SYMBOLS}

last_candle_epoch: Dict[str, int] = {s: 0 for s in SYMBOLS}
loss_streak: Dict[str, int] = {s: 0 for s in SYMBOLS}

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
# 📈 INDICADORES
# ============================================================

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) < EMA_SLOW + 100:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["ema_200"] = EMAIndicator(df["close"], 200).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()

    adx = ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD)
    df["adx"] = adx.adx()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_width"] = (
        bb.bollinger_hband() - bb.bollinger_lband()
    ) / df["close"]

    df["ret"] = df["close"].pct_change().fillna(0)
    df["volatility"] = df["close"].rolling(20).std()

    df["trend_strength"] = abs(df["ema_fast"] - df["ema_slow"])

    df["ema_200_slope"] = df["ema_200"].diff()

    return df


# ============================================================
# 🤖 ML
# ============================================================

def build_ml_dataset(df):
    df = df.dropna().copy()
    if len(df) < 100:
        return None, None

    tp_pct = TAKE_PROFIT / (STAKE_AMOUNT * MULTIPLIER)
    sl_pct = STOP_LOSS / (STAKE_AMOUNT * MULTIPLIER)

    targets = []
    for i in range(len(df) - 5):
        entry = df.iloc[i]["close"]
        future = df.iloc[i+1:i+5]

        hit_tp = (future["high"] >= entry * (1 + tp_pct)).any()
        hit_sl = (future["low"] <= entry * (1 - sl_pct)).any()

        if hit_tp and not hit_sl:
            targets.append(1)
        elif hit_sl and not hit_tp:
            targets.append(0)
        else:
            targets.append(None)

    df = df.iloc[:len(targets)]
    df["future"] = targets
    df = df.dropna()

    X = df.select_dtypes(include=["number"]).drop(columns=["future"], errors="ignore")
    y = df["future"]

    # balanceamento
    df_combined = pd.concat([X, y], axis=1)
    majority = df_combined[df_combined.future == df_combined.future.mode()[0]]
    minority = df_combined[df_combined.future != df_combined.future.mode()[0]]

    if len(minority) > 10:
        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )
        df_balanced = pd.concat([majority, minority_upsampled])
        X = df_balanced.drop("future", axis=1)
        y = df_balanced["future"]

    return X.tail(1500), y.tail(1500)


def train_ml(symbol):

    if not ML_ENABLED:
        return

    if time.time() - last_ml_train[symbol] < 300:
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        return

    X, y = build_ml_dataset(df)
    if X is None or len(set(y)) < 2:
        return

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

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
# 💰 PROPOSAL → BUY
# ============================================================

async def send_proposal(ws, symbol, direction):
    global REQ_ID_SEQ

    async with symbol_locks[symbol]:

        if not TRADE_ENABLED:
            return

        if len(open_trades[symbol]) > 0:
            return

        if pending_buy_symbol[symbol]:
            return

        if proposal_lock[symbol]:
            return

        if time.time() - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
            return

        proposal_lock[symbol] = True
        pending_buy_symbol[symbol] = True

        contract_type = "MULTUP" if direction == "UP" else "MULTDOWN"

        req_id = REQ_ID_SEQ
        REQ_ID_SEQ += 1

        pending_proposals[req_id] = {"symbol": symbol}

        await ws.send(json.dumps({
            "proposal": 1,
            "amount": STAKE_AMOUNT,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "symbol": symbol,
            "multiplier": MULTIPLIER,
            "limit_order": {
                "take_profit": TAKE_PROFIT,
                "stop_loss": STOP_LOSS
            },
            "req_id": req_id
        }))


async def handle_proposal(ws, data):
    req_id = data.get("req_id")
    if req_id not in pending_proposals:
        return

    pending_proposals.pop(req_id, None)

    await ws.send(json.dumps({
        "buy": data["proposal"]["id"],
        "price": STAKE_AMOUNT
    }))


# ============================================================
# 🌐 LOOP WS
# ============================================================

async def ws_loop(symbol):

    global daily_pnl, trading_paused, current_balance

    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=30,
                ping_timeout=30,
                max_queue=None
            ) as ws:

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await ws.recv()

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

                    if "candles" in data:
                        df = pd.DataFrame(data["candles"])
                        df["volume"] = 1.0
                        candles[symbol] = calcular_indicadores(df)
                        asyncio.create_task(train_ml_background(symbol))
                        continue

                    if "ohlc" in data:

                        epoch = data["ohlc"]["epoch"]

                        if epoch == last_candle_epoch[symbol]:
                            continue

                        last_candle_epoch[symbol] = epoch

                        df = candles[symbol]
                        df = pd.concat(
                            [df, pd.DataFrame([data["ohlc"]])]
                        ).tail(HISTORY_COUNT)

                        df["volume"] = 1.0
                        candles[symbol] = calcular_indicadores(df)

                        row = candles[symbol].iloc[-1]

                        if row["adx"] < 18:
                            continue

                        if row["bb_width"] < 0.002:
                            continue

                        prob = ml_predict(symbol, row)
                        if prob is None:
                            continue

                        conf = max(prob, 1 - prob)
                        dynamic_threshold = 0.62 if row["adx"] > 25 else 0.70

                        if conf < dynamic_threshold:
                            continue

                        trend_up = row["ema_fast"] > row["ema_slow"]
                        trend_down = row["ema_fast"] < row["ema_slow"]

                        if prob > 0.5 and trend_up:
                            direction = "UP"
                        elif prob <= 0.5 and trend_down:
                            direction = "DOWN"
                        else:
                            continue

                        if loss_streak[symbol] >= 2:
                            continue

                        await send_proposal(ws, symbol, direction)
                        continue

                    if "buy" in data:
                        cid = data["buy"]["contract_id"]
                        open_trades[symbol][cid] = True
                        last_trade_time[symbol] = time.time()

                        await ws.send(json.dumps({
                            "proposal_open_contract": 1,
                            "contract_id": cid,
                            "subscribe": 1
                        }))
                        continue

                    if "proposal_open_contract" in data:
                        poc = data["proposal_open_contract"]

                        if poc.get("is_sold"):

                            cid = poc["contract_id"]
                            profit = float(poc.get("profit", 0))
                            daily_pnl += profit

                            open_trades[symbol].pop(cid, None)
                            pending_buy_symbol[symbol] = False
                            proposal_lock[symbol] = False

                            if profit < 0:
                                loss_streak[symbol] += 1
                            else:
                                loss_streak[symbol] = 0

                            if daily_pnl <= -DAILY_MAX_LOSS:
                                trading_paused = True

                        continue

        except Exception as e:
            log(f"{symbol} WS erro {e}", "error")
            await asyncio.sleep(3)


# ============================================================
# 🚀 MAIN
# ============================================================

async def main():
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))


if __name__ == "__main__":
    asyncio.run(main())
