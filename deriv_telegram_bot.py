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
# 🆕 SINCRONIZAÇÃO PORTFÓLIO
# ============================================================
async def sync_open_contracts(ws):
    await ws.send(json.dumps({"portfolio": 1}))
    response = json.loads(await ws.recv())

    for s in SYMBOLS:
        open_trades[s].clear()

    if "portfolio" not in response:
        return

    for contract in response["portfolio"].get("contracts", []):
        if contract.get("is_sold"):
            continue
        symbol = contract.get("symbol")
        cid = contract.get("contract_id")
        if symbol in open_trades:
            open_trades[symbol][cid] = True


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
    df["ema_200"] = EMAIndicator(df["close"], 200).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()
    df["mfi"] = MFIIndicator(
        df["high"], df["low"], df["close"], df["volume"], MFI_PERIOD
    ).money_flow_index()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_width"] = (
        bb.bollinger_hband() - bb.bollinger_lband()
    ) / df["close"]

    adx = ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD)
    df["adx"] = adx.adx()

    df["ret"] = df["close"].pct_change().fillna(0)

    df["range"] = df["high"] - df["low"]
    df["body"] = abs(df["close"] - df["open"])

    df["upper_wick"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_wick"] = df[["close","open"]].min(axis=1) - df["low"]

    df["wick_ratio"] = (
        df["upper_wick"] + df["lower_wick"]
    ) / df["range"].replace(0, 1e-9)

    df["range_expansion"] = (
        df["range"] / df["range"].rolling(20).mean()
    )

    df["volatility_squeeze"] = (
        df["bb_width"] / df["bb_width"].rolling(20).mean()
    )

    # =====================================
    # ✅ FEATURES DE ESTRUTURA (CORRETO)
    # =====================================

    df["dist_ema_slow"] = (
        df["close"] - df["ema_slow"]
    ) / df["ema_slow"]
    df["dist_ema_200"] = (
    
    df["close"] - df["ema_200"]
    ) / df["ema_200"]
    
    df["ema_200_slope"] = df["ema_200"].diff()

    df["ema_slow_slope"] = df["ema_slow"].diff()

    df["rolling_high_20"] = df["high"].rolling(20).max()
    df["rolling_low_20"] = df["low"].rolling(20).min()

    df["dist_top"] = (
        df["rolling_high_20"] - df["close"]
    ) / df["close"]

    df["dist_bottom"] = (
        df["close"] - df["rolling_low_20"]
    ) / df["close"]

    return df

# ============================================================
# 🚦 FILTRO DE MERCADO
# ============================================================
def market_is_good(symbol, direction):
    df = candles[symbol]
    if len(df) < 60:
        return False

    row = df.iloc[-1]

    if row.get("adx", 0) < 18:
        return False

    if row.get("bb_width", 0) < 0.0016:
        return False

    if direction == "UP" and row.get("rsi", 50) > 70:
        return False
    if direction == "DOWN" and row.get("rsi", 50) < 30:
        return False

    return True
    
def market_is_exploding(symbol):
    
    df = candles[symbol]

    if len(df) < 30:
        return False

    row = df.iloc[-1]

    # candle muito maior que média recente
    if row.get("range_expansion", 1) > 2.2:
        return True

    # banda abrindo rápido demais
    if row.get("volatility_squeeze", 1) > 1.8:
        return True

    return False


# ============================================================
# 🤖 ML
# ============================================================
def build_ml_dataset(df):
    df = df.dropna().copy()
    if len(df) < 50:
        return None, None

    tp_pct = TAKE_PROFIT / (STAKE_AMOUNT * MULTIPLIER)
    sl_pct = STOP_LOSS / (STAKE_AMOUNT * MULTIPLIER)

    targets = []
    for i in range(len(df) - 8):
        entry = df.iloc[i]["close"]
        future = df.iloc[i+1:i+8]

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

    return X.tail(1000), y.tail(1000)


async def train_ml(symbol):
    if not ML_ENABLED:
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        return

    X, y = build_ml_dataset(df)
    if X is None or len(X) < 50:
        return

    if len(set(y)) < 2:
        return

    model = RandomForestClassifier(n_estimators=120, max_depth=8)
    model.fit(X, y)

    ml_models[symbol] = (model, X.columns.tolist())
    ml_model_ready[symbol] = True


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

        if open_trades[symbol] or proposal_lock[symbol]:
            return

        if time.time() - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
            return
            
        if market_is_exploding(symbol): 
            return

        proposal_lock[symbol] = True
        proposal_lock_time[symbol] = time.time()

        contract_type = "MULTUP" if direction == "UP" else "MULTDOWN"

        req_id = REQ_ID_SEQ
        REQ_ID_SEQ += 1

        pending_proposals[req_id] = {
            "symbol": symbol,
            "direction": direction
        }

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

    info = pending_proposals.pop(req_id)
    symbol = info["symbol"]

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
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=None
            ) as ws:

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await ws.recv()

                await sync_open_contracts(ws)

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
                    
                    last_activity_time[symbol] = time.time()
                    check_daily_reset()

                    # 🔓 auto reset proposal lock travado
                    if proposal_lock[symbol]:
                        if time.time() - proposal_lock_time.get(symbol, 0) > 60:
                              proposal_lock[symbol] = False
                              log(f"{symbol} proposal lock reset automático")

                    data = json.loads(raw)

                    if "candles" in data:
                        df = pd.DataFrame(data["candles"])
                        df["date"] = pd.to_datetime(df["epoch"], unit="s")
                        df["volume"] = 1.0
                        candles[symbol] = calcular_indicadores(df)
                        await train_ml(symbol)
                        continue

                    if "balance" in data:
                        current_balance = float(data["balance"]["balance"])
                        continue

                    if "proposal" in data:
                        await handle_proposal(ws, data)
                        continue

                    if "buy" in data:
                        cid = data["buy"]["contract_id"]
                        open_trades[symbol][cid] = True
                        proposal_lock[symbol] = False
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
                            proposal_lock[symbol] = False

                            if daily_pnl <= -DAILY_MAX_LOSS:
                                trading_paused = True
                        continue

                    if "ohlc" in data:
                        c = data["ohlc"]
                        df = candles[symbol]
                        df = pd.concat([df, pd.DataFrame([c])]).tail(HISTORY_COUNT)
                        df["volume"] = 1.0
                        candles[symbol] = calcular_indicadores(df)
                        await train_ml(symbol)

                        row = candles[symbol].iloc[-1]
                        prob = ml_predict(symbol, row)
                        if prob is None:
                            continue

                        conf = max(prob, 1 - prob)
                        if conf < ML_CONF_THRESHOLD:
                            continue

                        direction = "UP" if prob > 0.5 else "DOWN"

                        if not market_is_good(symbol, direction):
                            continue

                        if trading_paused:
                            continue

                        await send_proposal(ws, symbol, direction)

        except Exception as e:
            log(f"{symbol} WS erro {e}", "error")
            proposal_lock[symbol] = False
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
