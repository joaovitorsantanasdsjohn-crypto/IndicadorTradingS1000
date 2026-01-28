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

SYMBOLS = ["frxEURUSD", "frxUSDJPY", "frxGBPUSD"]
GRANULARITY_SECONDS = 300
HISTORY_COUNT = 1200

EMA_FAST, EMA_MID, EMA_SLOW = 9, 21, 55
RSI_PERIOD = 14
MFI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.3

ML_ENABLED = True and SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 300
ML_CONF_THRESHOLD = 0.55

TRADE_ENABLED = True
STAKE_AMOUNT = 1.0
MULTIPLIER = 20
TAKE_PROFIT = 0.5
STOP_LOSS = 0.3
MAX_OPEN_TRADES_PER_SYMBOL = 3
TRADE_COOLDOWN_SECONDS = 15
DAILY_MAX_LOSS = 3.0


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
# 📈 INDICADORES (CORRIGIDO)
# ============================================================
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Garante colunas essenciais
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return df

    # FOREX não tem volume real
    if "volume" not in df.columns:
        df["volume"] = 0.0

    if len(df) < EMA_SLOW + 50:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()

    df["mfi"] = MFIIndicator(
        df["high"], df["low"], df["close"], df["volume"], MFI_PERIOD
    ).money_flow_index()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    adx = ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD)
    df["adx"] = adx.adx()
    df["ret"] = df["close"].pct_change().fillna(0)

    return df


# ============================================================
# 🤖 MACHINE LEARNING
# ============================================================
def build_ml_dataset(df):
    df = df.dropna().copy()
    df["future"] = (df["close"].shift(-2) > df["close"]).astype(int)
    X = df.drop(columns=["future", "epoch"], errors="ignore").iloc[:-2]
    y = df["future"].iloc[:-2]
    return X.tail(1000), y.tail(1000)

async def train_ml(symbol):
    if not ML_ENABLED:
        return
    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        return
    X, y = build_ml_dataset(df)
    model = RandomForestClassifier(n_estimators=80, max_depth=6)
    model.fit(X, y)
    ml_models[symbol] = (model, X.columns.tolist())
    ml_model_ready[symbol] = True
    log(f"{symbol} ML treinado")

def ml_predict(symbol, row):
    if not ml_model_ready[symbol]:
        return None
    model, cols = ml_models[symbol]
    X = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    return model.predict_proba(X)[0][1]


# ============================================================
# 💰 TRADING
# ============================================================
async def open_trade(ws, symbol, direction):
    if trading_paused:
        return

    now = time.time()
    if now - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
        return
    if len(open_trades[symbol]) >= MAX_OPEN_TRADES_PER_SYMBOL:
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
        log(f"{symbol} Erro trade: {data['error']}", "error")
        return

    cid = data["buy"]["contract_id"]
    open_trades[symbol].append(cid)
    last_trade_time[symbol] = now

    await ws.send(json.dumps({
        "proposal_open_contract": 1,
        "contract_id": cid,
        "subscribe": 1
    }))

    log(f"{symbol} TRADE {direction} aberto ID {cid}")


# ============================================================
# 🌐 WEBSOCKET (CORRIGIDO)
# ============================================================
async def ws_loop(symbol):
    global daily_pnl, trading_paused

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
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

                    # Histórico
                    if "candles" in data and isinstance(data["candles"], list):
                        df = pd.DataFrame(data["candles"])
                        if "epoch" not in df.columns:
                            continue
                        df["date"] = pd.to_datetime(df["epoch"], unit="s")
                        if "volume" not in df.columns:
                            df["volume"] = 0.0
                        candles[symbol] = calcular_indicadores(df)
                        await train_ml(symbol)
                        continue

                    # Nova vela
                    if "ohlc" in data:
                        c = data["ohlc"]
                        if "epoch" not in c or "close" not in c:
                            continue
                        if "volume" not in c:
                            c["volume"] = 0.0

                        new_row = pd.DataFrame([c])
                        new_row["date"] = pd.to_datetime(new_row["epoch"], unit="s")

                        candles[symbol] = pd.concat([candles[symbol], new_row]).tail(HISTORY_COUNT)
                        candles[symbol] = calcular_indicadores(candles[symbol])
                        await train_ml(symbol)

                        row = candles[symbol].iloc[-1]
                        prob = ml_predict(symbol, row)
                        if prob is None:
                            continue

                        conf = max(prob, 1 - prob)
                        if conf < ML_CONF_THRESHOLD:
                            continue

                        direction = "UP" if prob > 0.5 else "DOWN"
                        await open_trade(ws, symbol, direction)
                        continue

                    # Fechamento
                    if "proposal_open_contract" in data:
                        poc = data["proposal_open_contract"]
                        if poc.get("is_sold"):
                            cid = poc["contract_id"]
                            profit = float(poc.get("profit", 0))
                            daily_pnl += profit
                            for s in SYMBOLS:
                                if cid in open_trades[s]:
                                    open_trades[s].remove(cid)
                            log(f"Trade fechado {cid} | Resultado {profit} | PnL Diário {daily_pnl}")
                            if daily_pnl <= -DAILY_MAX_LOSS:
                                trading_paused = True
                                log("🚨 LIMITE DE PERDA DIÁRIA ATINGIDO — BOT PAUSADO")

        except Exception as e:
            log(f"{symbol} WS erro {e}", "error")
            await asyncio.sleep(5)


# ============================================================
# 🌍 FLASK (KEEP ALIVE)
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
