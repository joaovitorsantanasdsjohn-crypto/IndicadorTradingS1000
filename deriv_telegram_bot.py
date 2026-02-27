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

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils import resample
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False


# ============================================================
# 🔧 CONFIG
# ============================================================

load_dotenv()

DERIV_TOKEN = os.getenv("DERIV_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = ["frxEURUSD","frxUSDJPY","frxGBPUSD","frxAUDUSD"]

GRANULARITY_SECONDS = 900
HISTORY_COUNT = 2000

EMA_FAST = 9
EMA_SLOW = 55

ML_ENABLED = True and SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 700

STAKE_AMOUNT = 1.0
MULTIPLIER = 100
TAKE_PROFIT = 0.4
STOP_LOSS = 0.2

TRADE_COOLDOWN_SECONDS = 30
DAILY_MAX_LOSS = 2.0

WATCHDOG_TIMEOUT = 900


# ============================================================
# 📊 ESTADO GLOBAL
# ============================================================

candles = {s: pd.DataFrame() for s in SYMBOLS}

ml_models = {}
ml_model_ready = {s: False for s in SYMBOLS}
last_ml_train = {s: 0 for s in SYMBOLS}

open_trades = {s:{} for s in SYMBOLS}
pending_buy_symbol = {s:False for s in SYMBOLS}

last_trade_time = {s:0 for s in SYMBOLS}
last_activity_time = {s:time.time() for s in SYMBOLS}

last_candle_epoch = {s:0 for s in SYMBOLS}

loss_streak = {s:0 for s in SYMBOLS}
loss_pause_until = {s:0 for s in SYMBOLS}

proposal_lock = {s:False for s in SYMBOLS}
symbol_locks = {s:asyncio.Lock() for s in SYMBOLS}

pending_proposals={}
REQ_ID_SEQ=1

daily_pnl=0
trading_paused=False


# ============================================================
# 📝 LOG
# ============================================================

logger=logging.getLogger("BOT")
logger.setLevel(logging.INFO)

handler=logging.StreamHandler()
handler.setFormatter(
logging.Formatter("%(asctime)s | %(message)s")
)

logger.handlers.clear()
logger.addHandler(handler)

def log(msg):
    logger.info(msg)


# ============================================================
# 📈 INDICADORES FOREX COMPLETO
# ============================================================

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) < 250:
        return df

    # =========================
    # 📊 EMAs
    # =========================
    df["ema_fast"] = EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_mid"]  = EMAIndicator(df["close"], 21).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], 55).ema_indicator()
    df["ema200"]   = EMAIndicator(df["close"], 200).ema_indicator()

    df["ema200_slope"] = df["ema200"].diff()
    df["ema_slow_slope"] = df["ema_slow"].diff()

    # =========================
    # 📊 RSI + MFI
    # =========================
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    df["mfi"] = MFIIndicator(
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        14
    ).money_flow_index()

    # =========================
    # 📊 ADX
    # =========================
    adx = ADXIndicator(df["high"], df["low"], df["close"], 14)
    df["adx"] = adx.adx()

    # =========================
    # 📊 Bollinger
    # =========================
    bb = BollingerBands(df["close"], 20, 2.2)

    df["bb_width"] = (
        bb.bollinger_hband() -
        bb.bollinger_lband()
    ) / df["close"]

    # =========================
    # 🕯 Estrutura de Vela
    # =========================
    df["range"] = df["high"] - df["low"]
    df["body"]  = abs(df["close"] - df["open"])

    df["upper_wick"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_wick"] = df[["close","open"]].min(axis=1) - df["low"]

    df["wick_ratio"] = (
        df["upper_wick"] + df["lower_wick"]
    ) / df["range"].replace(0,1e-9)

    df["body_ratio"] = df["body"] / df["range"].replace(0,1e-9)

    # =========================
    # 🏦 Manipulação Institucional
    # (Stop Hunt / Liquidez)
    # =========================

    df["rolling_high_20"] = df["high"].rolling(20).max()
    df["rolling_low_20"]  = df["low"].rolling(20).min()

    df["liquidity_sweep_high"] = (
        (df["high"] > df["rolling_high_20"].shift(1)) &
        (df["close"] < df["rolling_high_20"].shift(1))
    ).astype(int)

    df["liquidity_sweep_low"] = (
        (df["low"] < df["rolling_low_20"].shift(1)) &
        (df["close"] > df["rolling_low_20"].shift(1))
    ).astype(int)

    # =========================
    # 📈 Estrutura de Mercado
    # =========================

    df["higher_high"] = (
        df["high"] > df["high"].shift(1)
    ).astype(int)

    df["lower_low"] = (
        df["low"] < df["low"].shift(1)
    ).astype(int)

    df["market_structure_bias"] = (
        df["higher_high"] - df["lower_low"]
    )

    # =========================
    # 📊 Volatilidade
    # =========================

    df["volatility"] = df["close"].rolling(20).std()

    df["range_expansion"] = (
        df["range"] /
        df["range"].rolling(20).mean()
    )

    df["volatility_squeeze"] = (
        df["bb_width"] /
        df["bb_width"].rolling(20).mean()
    )

    # =========================
    # 📍 Distância de Estrutura
    # =========================

    df["dist_ema_slow"] = (
        df["close"] - df["ema_slow"]
    ) / df["ema_slow"]

    df["dist_ema200"] = (
        df["close"] - df["ema200"]
    ) / df["ema200"]

    return df
    
# ============================================================
# 🤖 ML DATASET FOREX
# ============================================================

def build_ml_dataset(df):

    df=df.dropna().copy()

    if len(df)<150:
        return None,None

    tp=TAKE_PROFIT/(STAKE_AMOUNT*MULTIPLIER)
    sl=STOP_LOSS/(STAKE_AMOUNT*MULTIPLIER)

    y=[]

    for i in range(len(df)-6):

        entry=df.iloc[i]["close"]
        future=df.iloc[i+1:i+6]

        hit_tp=(future["high"]>=entry*(1+tp)).any()
        hit_sl=(future["low"]<=entry*(1-sl)).any()

        if hit_tp and not hit_sl:
            y.append(1)
        elif hit_sl and not hit_tp:
            y.append(0)
        else:
            y.append(None)

    df=df.iloc[:len(y)]
    df["target"]=y
    df=df.dropna()

    X=df.select_dtypes("number").drop(columns=["target"])
    y=df["target"]

    comb=pd.concat([X,y],axis=1)

    maj=comb[comb.target==comb.target.mode()[0]]
    mino=comb[comb.target!=comb.target.mode()[0]]

    if len(mino)>20:
        mino=resample(mino,
                      replace=True,
                      n_samples=len(maj),
                      random_state=42)

    comb=pd.concat([maj,mino])

    return comb.drop("target",axis=1),comb["target"]


def train_ml(symbol):

    if time.time()-last_ml_train[symbol]<300:
        return

    df=candles[symbol]

    if len(df)<ML_MIN_TRAINED_SAMPLES:
        return

    X,y=build_ml_dataset(df)

    if X is None or len(set(y))<2:
        return

    model=RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=6,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X,y)

    ml_models[symbol]=(model,X.columns.tolist())
    ml_model_ready[symbol]=True
    last_ml_train[symbol]=time.time()


async def train_ml_background(symbol):
    loop=asyncio.get_running_loop()
    await loop.run_in_executor(None,train_ml,symbol)


def ml_predict(symbol,row):

    if not ml_model_ready[symbol]:
        return None

    model,cols=ml_models[symbol]

    try:
        vals=[row[c] for c in cols]
    except:
        return None

    if any(pd.isna(v) for v in vals):
        return None

    X=pd.DataFrame([vals],columns=cols)

    return model.predict_proba(X)[0][1]


# ============================================================
# 💰 PROPOSAL
# ============================================================

async def send_proposal(ws,symbol,direction):

    global REQ_ID_SEQ

    async with symbol_locks[symbol]:

        if pending_buy_symbol[symbol]:
            return

        if len(open_trades[symbol])>0:
            return

        if time.time()<loss_pause_until[symbol]:
            return

        if time.time()-last_trade_time[symbol]<TRADE_COOLDOWN_SECONDS:
            return

        proposal_lock[symbol]=True
        pending_buy_symbol[symbol]=True

        req=REQ_ID_SEQ
        REQ_ID_SEQ+=1

        pending_proposals[req]=symbol

        contract="MULTUP" if direction=="UP" else "MULTDOWN"

        await ws.send(json.dumps({
            "proposal":1,
            "amount":STAKE_AMOUNT,
            "basis":"stake",
            "contract_type":contract,
            "symbol":symbol,
            "currency":"USD",
            "multiplier":MULTIPLIER,
            "limit_order":{
                "take_profit":TAKE_PROFIT,
                "stop_loss":STOP_LOSS},
            "req_id":req
        }))


async def handle_proposal(ws,data):

    rid=data.get("req_id")

    if rid not in pending_proposals:
        return

    pending_proposals.pop(rid,None)

    await ws.send(json.dumps({
        "buy":data["proposal"]["id"],
        "price":STAKE_AMOUNT
    }))

# ============================================================
# 🌐 LOOP WS
# ============================================================

async def ws_loop(symbol):

    global daily_pnl,trading_paused

    while True:

        try:

            async with websockets.connect(
                WS_URL,
                ping_interval=25,
                ping_timeout=25,
                max_queue=None
            ) as ws:

                await ws.send(json.dumps({"authorize":DERIV_TOKEN}))
                await ws.recv()

                await ws.send(json.dumps({
                    "ticks_history":symbol,
                    "granularity":GRANULARITY_SECONDS,
                    "count":HISTORY_COUNT,
                    "end":"latest",
                    "style":"candles",
                    "subscribe":1
                }))

                async for raw in ws:

                    last_activity_time[symbol]=time.time()

                    data=json.loads(raw)

                    # ================= HIST =================
                    if "candles" in data:

                        df=pd.DataFrame(data["candles"])
                        df["volume"]=1

                        candles[symbol]=calcular_indicadores(df)

                        asyncio.create_task(
                            train_ml_background(symbol)
                        )
                        continue


                    # ================= NOVA VELA =================
                    if "ohlc" in data:

                        epoch=data["ohlc"]["epoch"]

                        if epoch==last_candle_epoch[symbol]:
                            continue

                        last_candle_epoch[symbol]=epoch

                        df=candles[symbol]
                        df=pd.concat(
                            [df,pd.DataFrame([data["ohlc"]])]
                        ).tail(HISTORY_COUNT)

                        df["volume"]=1
                        candles[symbol]=calcular_indicadores(df)

                        row=df.iloc[-1]

                        # ===== REGIME FOREX =====
                        if row["adx"]<16:
                            continue

                        if row["bb_width"]<0.001:
                            continue

                        if abs(row["ema200_slope"])<1e-6:
                            continue

                        prob=ml_predict(symbol,row)

                        if prob is None:
                            continue

                        conf=max(prob,1-prob)
                        threshold=0.60 if row["adx"]>25 else 0.68

                        if conf<threshold:
                            continue

                        trend_up=row["ema_fast"]>row["ema_slow"]>row["ema200"]
                        trend_down=row["ema_fast"]<row["ema_slow"]<row["ema200"]

                        if prob>0.5 and trend_up:
                            direction="UP"
                        elif prob<=0.5 and trend_down:
                            direction="DOWN"
                        else:
                            continue

                        await send_proposal(ws,symbol,direction)
                        continue


                    # ================= PROPOSAL =================
                    if "proposal" in data:
                        await handle_proposal(ws,data)
                        continue


                    # ================= BUY =================
                    if "buy" in data:

                        cid=data["buy"]["contract_id"]

                        open_trades[symbol][cid]=True
                        last_trade_time[symbol]=time.time()

                        await ws.send(json.dumps({
                            "proposal_open_contract":1,
                            "contract_id":cid,
                            "subscribe":1
                        }))
                        continue


                    # ================= CLOSE =================
                    if "proposal_open_contract" in data:

                        poc=data["proposal_open_contract"]

                        if poc.get("is_sold"):

                            cid=poc["contract_id"]
                            profit=float(poc.get("profit",0))

                            open_trades[symbol].pop(cid,None)

                            pending_buy_symbol[symbol]=False
                            proposal_lock[symbol]=False

                            daily_pnl+=profit

                            if profit<0:
                                loss_streak[symbol]+=1
                                loss_pause_until[symbol]=time.time()+900
                            else:
                                loss_streak[symbol]=0

                            if daily_pnl<=-DAILY_MAX_LOSS:
                                trading_paused=True

                        continue

        except Exception as e:

            log(f"{symbol} WS reconnect {e}")

            pending_buy_symbol[symbol]=False
            proposal_lock[symbol]=False

            await asyncio.sleep(3)


# ============================================================
# 🌍 FLASK (Health Check Render)
# ============================================================

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200


def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000"))
    )


# ============================================================
# 🚀 MAIN
# ============================================================

async def main():
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))


if __name__=="__main__":

    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()

    asyncio.run(main())
