import asyncio
import json
import os
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple
import joblib

import pandas as pd
import websockets
from flask import Flask
from dotenv import load_dotenv

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator

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

SYMBOLS = [ 
    "frxUSDJPY",
    "frxGBPJPY",
    "frxAUDUSD"
]

GRANULARITY_SECONDS = 900
HISTORY_COUNT = 1200

ML_ENABLED = True and SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 120
ML_CALIBRATION_FACTOR = 0.90

STAKE_AMOUNT = 1.0
MULTIPLIER = 100
TAKE_PROFIT = 0.4
STOP_LOSS = 0.2

TRADE_COOLDOWN_SECONDS = 60
DAILY_MAX_LOSS = 1.0

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
last_save_time = {s:0 for s in SYMBOLS}

last_candle_epoch = {s:0 for s in SYMBOLS}

loss_streak = {s:0 for s in SYMBOLS}
loss_pause_until = {s:0 for s in SYMBOLS}

proposal_lock = {s:False for s in SYMBOLS}
symbol_locks = {s:asyncio.Lock() for s in SYMBOLS}

pending_proposals={}
REQ_ID_SEQ=1

daily_pnl=0
trading_paused=False
STATE_FILE = "daily_state.json"
ML_DATA_FILE = "ml_data.json"
current_day = datetime.now(timezone.utc).date()

#=============================================================
# 📌FUNÇÕES DE ESTADO
#=============================================================

def save_daily_state():

    data = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "daily_pnl": daily_pnl,
        "trading_paused": trading_paused
    }

    with open(STATE_FILE, "w") as f:
        json.dump(data, f)

def load_daily_state():

    global daily_pnl, trading_paused

    if not os.path.exists(STATE_FILE):
        return

    with open(STATE_FILE, "r") as f:
        data = json.load(f)

    today = datetime.utcnow().strftime("%Y-%m-%d")

    if data["date"] == today:
        daily_pnl = data["daily_pnl"]
        trading_paused = data["trading_paused"]

load_daily_state()
save_daily_state()

def save_ml_data():

    data = {}

    for s in SYMBOLS:

        df = candles[s]

        if len(df) > 800:
            df = df.tail(800)

        data[s] = df.to_dict()

    with open(ML_DATA_FILE,"w") as f:
        json.dump(data,f)


def load_ml_data():

    if not os.path.exists(ML_DATA_FILE):
        return

    with open(ML_DATA_FILE,"r") as f:
        data=json.load(f)

    for s in data:

        candles[s] = pd.DataFrame(data[s])

load_ml_data()

for s in SYMBOLS:
    if os.path.exists(f"model_{s}.pkl"):
        try:
            ml_models[s] = joblib.load(f"model_{s}.pkl")
            ml_model_ready[s] = True
            print(f"{s} ML LOADED FROM FILE")
        except:
            pass

# ============================================================
# 📝 LOG
# ============================================================

logger=logging.getLogger("BOT")
logger.setLevel(logging.INFO)

handler=logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))

logger.handlers.clear()
logger.addHandler(handler)

def log(msg):
    logger.info(msg)


# ============================================================
# 🔄 RESET DIÁRIO
# ============================================================

def check_daily_reset():

    global current_day, daily_pnl, trading_paused

    today = datetime.now(timezone.utc).date()

    if today != current_day:

        log("RESET DIARIO")

        current_day = today
        daily_pnl = 0
        trading_paused = False
        save_daily_state()

        
        for s in SYMBOLS:
            loss_streak[s] = 0
            loss_pause_until[s] = 0


# ============================================================
# 🔁 SINCRONIZAÇÃO PORTFOLIO
# ============================================================

async def sync_open_contracts(ws):

    await ws.send(json.dumps({"portfolio":1}))
    response=json.loads(await ws.recv())

    for s in SYMBOLS:
        open_trades[s].clear()

    if "portfolio" not in response:

        for s in SYMBOLS:
            pending_buy_symbol[s]=False
            proposal_lock[s]=False
        return

    active=set()

    for c in response["portfolio"].get("contracts",[]):

        if c.get("is_sold"):
            continue

        symbol=c.get("symbol")
        cid=c.get("contract_id")

        if symbol in open_trades:
            open_trades[symbol][cid]=True
            active.add(symbol)

    for s in SYMBOLS:
        if s not in active:
            pending_buy_symbol[s]=False
            proposal_lock[s]=False


# ============================================================
# 📈 INDICADORES FOREX COMPLETO
# ============================================================

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) < 250:
        return df

    df["ema_fast"] = EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_mid"]  = EMAIndicator(df["close"], 21).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], 55).ema_indicator()
    df["ema200"]   = EMAIndicator(df["close"], 200).ema_indicator()
    
    df["dist_ema200"] = (df["close"] - df["ema200"]) / df["ema200"]
    df["ema200_slope"] = df["ema200"].diff()

    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    df["mfi"] = MFIIndicator(
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        14
    ).money_flow_index()

    adx = ADXIndicator(df["high"], df["low"], df["close"], 14)
    df["adx"] = adx.adx().clip(0,60)

    bb = BollingerBands(df["close"], 20, 2.2)
    df["bb_width"] = (
        bb.bollinger_hband() -
        bb.bollinger_lband()
    ) / df["close"]
    
    atr = AverageTrueRange(
        df["high"],
        df["low"],
        df["close"],
        14
    )

    df["atr"] = atr.average_true_range()

    df["range"] = df["high"] - df["low"]
    df["body"]  = abs(df["close"] - df["open"])

    df["range_expansion"] = (
        df["range"] /
        df["range"].rolling(20).mean()
    )

    df["volatility_squeeze"] = (
        df["bb_width"] /
        df["bb_width"].rolling(20).mean()
    )

    # ============================================================
    # 🕯 ESTRUTURA DE VELA (PRICE ACTION)
    # ============================================================
        
    df["upper_wick"] = (
        df["high"] - df[["open","close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open","close"]].min(axis=1) - df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0,1e-9)
    )

    df["wick_ratio"] = (
        df["upper_wick"] + df["lower_wick"]
    ) / df["range"].replace(0,1e-9)


    # ============================================================
    # 🏦 MANIPULAÇÃO INSTITUCIONAL (LIQUIDITY SWEEP)
    # ============================================================

    df["rolling_high_20"] = df["high"].rolling(20).max()
    df["rolling_low_20"]  = df["low"].rolling(20).min()
    
    df["range_position"] = (
        (df["close"] - df["rolling_low_20"]) /
        (df["rolling_high_20"] - df["rolling_low_20"])
    ).clip(0,1)

    # Stop Hunt acima
    df["liquidity_sweep_high"] = (
        (df["high"] > df["rolling_high_20"].shift(1)) &
        (df["close"] < df["rolling_high_20"].shift(1))
    ).astype(int)

    # Stop Hunt abaixo
    df["liquidity_sweep_low"] = (
        (df["low"] < df["rolling_low_20"].shift(1)) &
        (df["close"] > df["rolling_low_20"].shift(1))
    ).astype(int)


    # ============================================================
    # 📈 MICRO ESTRUTURA DE MERCADO
    # ============================================================

    df["higher_high"] = (
        df["high"] > df["high"].shift(1)
    ).astype(int)

    df["lower_low"] = (
        df["low"] < df["low"].shift(1)
    ).astype(int)

    df["market_structure_bias"] = (
        df["higher_high"] - df["lower_low"]
    )
    # ============================================================
    # 🧠 FEATURES DE CONTEXTO (PULLBACK / EXAUSTÃO)
    # ============================================================

    # 📉 Distância das EMAs (detecta preço esticado)
    df["dist_ema_fast"] = (df["close"] - df["ema_fast"]) / df["ema_fast"]
    df["dist_ema_slow"] = (df["close"] - df["ema_slow"]) / df["ema_slow"]

    # 🔥 Z-score (sobrecompra / sobrevenda real)
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    df["zscore_price"] = (df["close"] - rolling_mean) / rolling_std

    # 📊 Range position suavizado (menos ruído)
    df["range_position_smooth"] = df["range_position"].rolling(3).mean()

    # 🧠 Pullback real baseado em ATR
    df["pullback_strength"] = (
    (df["close"] - df["close"].rolling(5).max()) /
    df["atr"].replace(0,1e-9)
        
    )

    # ⚡ Velocidade e aceleração do preço
    df["price_velocity"] = df["close"].pct_change(3)
    df["price_acceleration"] = df["price_velocity"].diff()
    # 🔄 Sequência de candles (micro tendência)
    df["bullish_seq"] = (
        (df["close"] > df["open"]).astype(int)
    ).rolling(5).sum()

    df["bearish_seq"] = (
        (df["close"] < df["open"]).astype(int)
    ).rolling(5).sum()
    
    # ============================================================
    # 🔄 FEATURES DE REVERSÃO (NÍVEL AVANÇADO)
    # ============================================================

    # 📉 Perda de força da tendência
    df["trend_strength"] = abs(df["ema_fast"] - df["ema_slow"])
    df["trend_strength_change"] = df["trend_strength"].diff()

    # 🔥 Exaustão do movimento
    df["overextended_up"] = (
        (df["bullish_seq"] >= 4) &
        (df["zscore_price"] > 1.5)
    ).astype(int)

    df["overextended_down"] = (
        (df["bearish_seq"] >= 4) &
        (df["zscore_price"] < -1.5)
    ).astype(int)

    # ⚡ Falha de continuação (muito importante)
    df["failed_break_up"] = (
        (df["high"] > df["high"].shift(1)) &
        (df["close"] < df["close"].shift(1))
    ).astype(int)

    df["failed_break_down"] = (
        (df["low"] < df["low"].shift(1)) &
        (df["close"] > df["close"].shift(1))
    ).astype(int)

    # 🧠 Divergência simples (preço vs RSI)
    df["rsi_divergence_up"] = (
        (df["close"] < df["close"].shift(1)) &
        (df["rsi"] > df["rsi"].shift(1))
    ).astype(int)

    df["rsi_divergence_down"] = (
        (df["close"] > df["close"].shift(1)) &
        (df["rsi"] < df["rsi"].shift(1))
    ).astype(int)
    
    # ============================================================
    # 🏦 LIQUIDEZ INSTITUCIONAL (SMART MONEY)
    # ============================================================

    # 📊 Distância do topo/fundo recente (zona de liquidez)
    df["dist_to_high"] = (
        df["rolling_high_20"] - df["close"]
    ) / df["close"]

    df["dist_to_low"] = (
        df["close"] - df["rolling_low_20"]
    ) / df["close"]

    # 🔥 Proximidade de liquidez
    df["near_liquidity_high"] = (df["dist_to_high"] < 0.0015).astype(int)
    df["near_liquidity_low"]  = (df["dist_to_low"] < 0.0015).astype(int)

    # ⚡ Sweep com rejeição forte (chave)
    df["liquidity_rejection_up"] = (
        (df["liquidity_sweep_high"] == 1) &
        (df["close"] < df["open"])
    ).astype(int)

    df["liquidity_rejection_down"] = (
        (df["liquidity_sweep_low"] == 1) &
        (df["close"] > df["open"])
    ).astype(int)

    # 🧠 Pressão institucional (wick dominante)
    df["upper_wick_dominance"] = (
        df["upper_wick"] > df["body"] * 1.5
    ).astype(int)

    df["lower_wick_dominance"] = (
        df["lower_wick"] > df["body"] * 1.5
    ).astype(int)

    # 📉 Absorção de movimento (falsa continuação)
    df["absorption_up"] = (
        (df["close"] > df["open"]) &
        (df["upper_wick"] > df["body"] * 1.5)
    ).astype(int)

    df["absorption_down"] = (
        (df["close"] < df["open"]) &
        (df["lower_wick"] > df["body"] * 1.5)
    ).astype(int)

    return df


# ============================================================
# 🧠 MARKET FILTERS
# ============================================================

def forex_session_ok():
    utc_hour=datetime.utcnow().hour
    return 6 <= utc_hour <= 19
    
def market_is_open():
    now = datetime.utcnow()
    weekday = now.weekday()  # 0=segunda, 6=domingo

    # domingo = fechado
    if weekday == 6:
        return False

    # sábado = fechado
    if weekday == 5:
        return False

    return True

def market_is_good(row):

    if row["adx"] < 14:
        return False

    if row["bb_width"] < 0.0009:
        return False

    if abs(row["ema200_slope"]) < 5e-7:
        return False

    return True


def market_exploding(row):

    if row["range_expansion"] > 1.4:
        return True

    if row["volatility_squeeze"] > 1.3:
        return True

    return False

# ============================================================
# 🚫 FILTRO ANTI MANIPULAÇÃO INSTITUCIONAL
# ============================================================

def institutional_trap_filter(row, direction):

    # sweep acima → perigo de queda
    if row["liquidity_sweep_high"] == 1 and direction == "UP":
        return False

    # sweep abaixo → perigo de alta falsa
    if row["liquidity_sweep_low"] == 1 and direction == "DOWN":
        return False

    # Se mercado está explodindo, relaxa filtro
    if not market_exploding(row):

        if row["wick_ratio"] > 0.70:
            return False

        if row["body_ratio"] < 0.15:
            return False
   
    # relaxa distância da EMA em mercado explosivo
    if not market_exploding(row):

        if abs(row["dist_ema200"]) > 0.009:
            return False

    return True

# ============================================================
# 🤖 ML
# ============================================================

def build_ml_dataset(df):

    df = df.dropna().copy()

    if len(df) < 200:
        return None, None, None, None

    tp = TAKE_PROFIT / (STAKE_AMOUNT * MULTIPLIER)
    sl = STOP_LOSS / (STAKE_AMOUNT * MULTIPLIER)

    y_up = []
    y_down = []
    valid_idx = []

    for i in range(len(df) - 8):

        entry = df.iloc[i]["close"]
        future = df.iloc[i+1:i+8]

        # =========================
        # 🔵 UP
        # =========================
        hit_tp_up = (future["high"] >= entry * (1 + tp)).any()
        hit_sl_up = (future["low"] <= entry * (1 - sl)).any()

        # =========================
        # 🔴 DOWN
        # =========================
        hit_tp_down = (future["low"] <= entry * (1 - tp)).any()
        hit_sl_down = (future["high"] >= entry * (1 + sl)).any()

        # mantém só casos válidos (igual sua lógica)
        if (hit_tp_up != hit_sl_up) and (hit_tp_down != hit_sl_down):
            valid_idx.append(i)

            # UP target
            if hit_tp_up and not hit_sl_up:
                y_up.append(1)
            elif hit_sl_up and not hit_tp_up:
                y_up.append(0)
            else:
                y_up.append(0)

            # DOWN target
            if hit_tp_down and not hit_sl_down:
                y_down.append(1)
            elif hit_sl_down and not hit_tp_down:
                y_down.append(0)
            else:
                y_down.append(0)

    df = df.iloc[valid_idx].copy()

    df = df.dropna()

    if len(df) < 30:
        return None, None, None, None

    X = df.select_dtypes("number").copy()
    
    y_up = pd.Series(y_up).reset_index(drop=True)
    y_down = pd.Series(y_down).reset_index(drop=True)
    X = X.reset_index(drop=True)
   
    mask_up = ~y_up.isna()
    X_up = X[mask_up]
    y_up = y_up[mask_up]

    mask_down = ~y_down.isna()
    X_down = X[mask_down]
    y_down = y_down[mask_down]

    # balanceamento (igual seu código)
    def balance(X, y):
        comb = pd.concat([X, pd.Series(y, name="target")], axis=1)

        maj = comb[comb.target == comb.target.mode()[0]]
        mino = comb[comb.target != comb.target.mode()[0]]

        if len(mino) > 10:
            mino = resample(
                mino,
                replace=True,
                n_samples=len(maj),
                random_state=42
            )

        comb = pd.concat([maj, mino])

        return comb.drop("target", axis=1), comb["target"]

    X_up, y_up = balance(X_up, y_up)
    X_down, y_down = balance(X_down, y_down)

    return X_up, y_up, X_down, y_down


def train_ml(symbol):

    if time.time() - last_ml_train[symbol] < 300:
        return

    df = candles[symbol]

    if len(df) < ML_MIN_TRAINED_SAMPLES:
        return

    result = build_ml_dataset(df)

    X_up, y_up, X_down, y_down = result

    if X_up is None or y_up is None:
        return

    if X_down is None or y_down is None:
        return

    if len(set(y_up)) < 2:
        return

    if len(set(y_down)) < 2:
        return

    model_up = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=12,
        n_jobs=-1,
        random_state=42
    )

    model_down = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=12,
        n_jobs=-1,
        random_state=42
    )

    model_up.fit(X_up, y_up)
    model_down.fit(X_down, y_down)

    ml_models[symbol] = (model_up, model_down, X_up.columns.tolist())
    joblib.dump(ml_models[symbol], f"model_{symbol}.pkl")
    ml_model_ready[symbol] = True
    last_ml_train[symbol] = time.time()


async def train_ml_background(symbol):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, train_ml, symbol)


def ml_predict(symbol, row):

    if not ml_model_ready[symbol]:
        return None, None

    model_up, model_down, cols = ml_models[symbol]

    try:
        vals = [row[c] for c in cols]
    except:
        return None, None

    if any(pd.isna(v) for v in vals):
        return None, None

    X = pd.DataFrame([vals], columns=cols)

    prob_up = model_up.predict_proba(X)[0][1] * ML_CALIBRATION_FACTOR
    prob_down = model_down.predict_proba(X)[0][1] * ML_CALIBRATION_FACTOR

    return prob_up, prob_down


# ============================================================
# 💰 PROPOSAL
# ============================================================

async def send_proposal(ws,symbol,direction):

    global REQ_ID_SEQ

    async with symbol_locks[symbol]:

        if trading_paused:
            return

        if pending_buy_symbol[symbol]:
            return

        if len(open_trades[symbol])>0:
            return

        if time.time()<loss_pause_until[symbol]:
            return

        if time.time()-last_trade_time[symbol]<TRADE_COOLDOWN_SECONDS:
            return

        pending_buy_symbol[symbol]=True
        proposal_lock[symbol]=True

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
    if trading_paused:
        return

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

async def keep_alive(ws):
    while True:
        try:
            await ws.send(json.dumps({"ping": 1}))
        except:
            break
        await asyncio.sleep(30)

async def ws_loop(symbol):

    global daily_pnl, trading_paused
    
    reconnect_attempts = 0

    while True:

        if not market_is_open():
            log(f"{symbol} 💤 MARKET CLOSED")
            await asyncio.sleep(300)
            continue

        try:

            async with websockets.connect(
                WS_URL,
                ping_interval=60,
                ping_timeout=60,
                max_queue=None
            ) as ws:
                
                async def watchdog():
                    while True:
                        await asyncio.sleep(5)

                        if time.time() - last_activity_time[symbol] > WATCHDOG_TIMEOUT:
                            log(f"{symbol} ⚠️ WS STALL DETECTED")
                            await ws.close()
                            break

                asyncio.create_task(watchdog())
                asyncio.create_task(keep_alive(ws))

                await ws.send(json.dumps({"authorize":DERIV_TOKEN}))
                await ws.recv()
                reconnect_attempts = 0

                await sync_open_contracts(ws)

                await ws.send(json.dumps({
                    "ticks_history":symbol,
                    "granularity":GRANULARITY_SECONDS,
                    "count":HISTORY_COUNT,
                    "end":"latest",
                    "style":"candles",
                    "subscribe":1
                }))

                async for raw in ws:
                    
                    last_activity_time[symbol] = time.time()

                    check_daily_reset()

                    data=json.loads(raw)

                    if "candles" in data:

                        df=pd.DataFrame(data["candles"])
                        df["volume"]=1

                        candles[symbol]=calcular_indicadores(df)
                        if time.time() - last_save_time[symbol] > 30:
                            save_ml_data()
                            last_save_time[symbol] = time.time()

                        if time.time() - last_ml_train[symbol] > 600:
                            asyncio.create_task(train_ml_background(symbol))

                        continue


                    if "ohlc" in data:

                        epoch = data["ohlc"]["epoch"]

                        # ignora duplicado
                        if epoch <= last_candle_epoch[symbol]:
                            continue

                        # só valida gap se já existe histórico
                        if last_candle_epoch[symbol] != 0:

                            gap = epoch - last_candle_epoch[symbol]

                            # gap grande (normal)
                            if gap > (GRANULARITY_SECONDS * 3):
                                log(f"{symbol} ⚠️ BIG GAP DETECTED (OK - MARKET REOPEN)")
                                last_candle_epoch[symbol] = epoch
                                continue

                            # gap pequeno (erro)
                            elif gap > GRANULARITY_SECONDS * 2:
                                log(f"{symbol} ⚠️ SMALL GAP ERROR — RESET")
                                last_candle_epoch[symbol] = epoch
                                continue

                        # fluxo normal
                        last_candle_epoch[symbol] = epoch
                        
                                
                        df=candles[symbol]
                        df=pd.concat(
                            [df,pd.DataFrame([data["ohlc"]])]
                        ).tail(HISTORY_COUNT)

                        df["volume"]=1
                        candles[symbol]=calcular_indicadores(df)
                        if time.time() - last_save_time[symbol] > 30:
                            save_ml_data()
                            last_save_time[symbol] = time.time()

                        if len(df) < 300:
                            continue
                        
                        row=df.iloc[-2]

                        if not forex_session_ok():
                            continue

                        if not market_is_good(row):
                            continue

                        # ajuste fino de lateralidade
                        if row["adx"] < 20 and not market_exploding(row):
                            continue
                            

                        # ==========================
                        # 📊 ML PREDICT (NOVO)
                        # ==========================
                        prob_up, prob_down = ml_predict(symbol, row)

                        log(
                            f"ML | {symbol} | "
                            f"READY={ml_model_ready[symbol]} | "
                            f"ADX={row['adx']:.2f}"
                        )

                        if prob_up is None:
                            log(f"ML-HOLD | {symbol} | MODEL NOT READY")
                            continue

                        log(f"ML-UP   | {symbol} | {prob_up:.3f}")
                        log(f"ML-DOWN | {symbol} | {prob_down:.3f}")
                        
                        confidence_gap = abs(prob_up - prob_down)

                        if confidence_gap < 0.04:
                            continue

                        # ===================================================
                        # ✅ CONFIANÇA DINÂMICA BASEADA EM REGIME
                        # ===================================================

                        if row["adx"] >= 30:
                            dynamic_threshold = 0.58
                        elif row["adx"] >= 24:
                            dynamic_threshold = 0.62
                        elif row["adx"] >= 18:
                            dynamic_threshold = 0.65
                        else:
                            dynamic_threshold = 0.68

                        # evita entrar em vela já esticada demais
                        if row["body_ratio"] > 0.70 and row["range_expansion"] > 1.5:
                            continue

                        ema_diff = abs(row["ema_fast"] - row["ema_slow"])

                        if ema_diff < 0.00003 and row["adx"] < 22:
                            continue
                        

                        direction = None

                        if prob_up > dynamic_threshold and prob_up > prob_down:
                            direction = "UP"

                        elif prob_down > dynamic_threshold and prob_down > prob_up:
                            direction = "DOWN"

                        else:
                            continue

                        # 🚫 filtro institucional
                        if not institutional_trap_filter(row, direction):
                            if not market_exploding(row):
                                continue

                        # 🚫 BLOQUEIO DAILY LOSS (FIX)
                        if trading_paused or daily_pnl <= -DAILY_MAX_LOSS:
                            trading_paused = True
                            continue

                        await send_proposal(ws, symbol, direction)
                        await asyncio.sleep(0.01)
                        continue


                    if "proposal" in data:
                        await handle_proposal(ws,data)
                        continue


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


                    if "proposal_open_contract" in data:

                        poc=data["proposal_open_contract"]

                        if poc.get("is_sold"):

                            cid=poc["contract_id"]
                            profit=float(poc.get("profit",0))

                            open_trades[symbol].pop(cid,None)

                            pending_buy_symbol[symbol]=False
                            proposal_lock[symbol]=False

                            daily_pnl+=profit
                            save_daily_state()

                            if profit<0:
                                loss_streak[symbol]+=1
                                loss_pause_until[symbol]=time.time()+900
                            else:
                                loss_streak[symbol]=0

                            if daily_pnl <= -DAILY_MAX_LOSS:

                                trading_paused = True
                                save_daily_state()

                                logging.warning(
                                    f"DAILY MAX LOSS HIT | PNL={daily_pnl}"
                                )

                        continue

        except Exception as e:
            log(f"{symbol} WS reconnect {e}")
            reconnect_attempts += 1
        
            if reconnect_attempts >= 5:
                log(f"{symbol} 🔄 RESET FORÇADO")

                candles[symbol] = pd.DataFrame()
                ml_model_ready[symbol] = False
                last_ml_train[symbol] = 0

                reconnect_attempts = 0
            

                pending_buy_symbol[symbol] = False
                proposal_lock[symbol] = False
                pending_proposals.clear()

            # 🔥 Não tem reset mais
            log(f"{symbol} reconnect - mantendo dados")
            
            if "ws" in locals():
                try:
                    await ws.close()
                except:
                    pass

        delay = min(60, 5 * (reconnect_attempts + 1))
        await asyncio.sleep(delay)
                            
# ============================================================
# 🌍 FLASK
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
