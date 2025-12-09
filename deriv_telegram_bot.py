# deriv_telegram_bot.py — LÓGICA A (Opção A — Precisão Profissional para FOREX M5)
# Versão FINAL: filtros de PRICE ACTION e ATR removidos para mais sinais
# Corrigido erro 'float' object has no attribute 'astype'; adicionado mensagem de START e novo formato de sinal (ML em % inteiro)
# *** ATUALIZAÇÃO PEDIDA PELO USUÁRIO ***
# Horário enviado no Telegram agora é automaticamente convertido de UTC -> Horário de Brasília (BRT/BRST)

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
from pathlib import Path
import time
import random
import logging
import traceback
from collections import deque
import html

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações principais ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxGBPAUD", "frxGBPCAD", "frxAUDNZD",
    "frxEURCAD"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Parâmetros ----------------
BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

MIN_SECONDS_BETWEEN_SIGNALS = 3
MIN_SECONDS_BETWEEN_OPPOSITE = 45
MIN_CANDLES_BETWEEN_SIGNALS = 4

REL_EMA_SEP_PCT = 5e-06
MICRO_FORCE_ALLOW_THRESHOLD = 25
FORCE_MIN = 35

ML_ENABLED = SKLEARN_AVAILABLE
ML_N_ESTIMATORS = 40
ML_MAX_DEPTH = 4
ML_MIN_TRAINED_SAMPLES = 200
ML_CONF_THRESHOLD = 0.55
ML_MAX_SAMPLES = 2000
ML_RETRAIN_INTERVAL = 50

MIN_SIGNALS_PER_HOUR = 3
FALLBACK_WINDOW_SEC = 3600
FALLBACK_FORCE_MIN = 30
FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD = 20
FALLBACK_REL_EMA_SEP_PCT = 2e-05
FALLBACK_DURATION_SECONDS = 15 * 60

EMA_FAST = 9
EMA_MID = 20
EMA_SLOW = 200

INITIAL_HISTORY_COUNT = 500
HISTORY_MAX_TRIES = 5
MAX_CANDLES = 300

# ---------------- Estado ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}
ml_models = {}
ml_model_ready = {}
sent_timestamps = deque()
fallback_active_until = 0.0
historical_loaded = {s: False for s in SYMBOLS}
live_subscribed = {s: False for s in SYMBOLS}
ml_trained_samples = {s: 0 for s in SYMBOLS}
notify_flags = {s: {"connected": False, "history": False, "ml": False, "subscribed": False} for s in SYMBOLS}

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    if level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)
    print(msg, flush=True)

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False):
    now_ts = time.time()
    if symbol and not bypass_throttle:
        last = last_notify_time.get(symbol, 0)
        if now_ts - last < 3:
            log(f"[TG] throttle skip for {symbol}", "warning")
            return
        last_notify_time[symbol] = now_ts

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("⚠️ Telegram não configurado.", "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
    except Exception as e:
        log(f"[TG] Erro ao enviar: {e}\n{traceback.format_exc()}", "error")

# ---------------- Utilitários ----------------
def human_pair(symbol: str) -> str:
    return symbol.replace("frx", "")

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("epoch").reset_index(drop=True)

    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    df[f"ema{EMA_FAST}"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df[f"ema{EMA_MID}"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df[f"ema{EMA_SLOW}"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], 14).rsi().fillna(50.0)

    try:
        macd = MACD(df["close"], 26, 12, 9)
        df["macd_diff"] = macd.macd_diff().fillna(0.0)
    except:
        df["macd_diff"] = 0.0

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband().fillna(df["close"])
    df["bb_lower"] = bb.bollinger_lband().fillna(df["close"])
    df["bb_mavg"] = bb.bollinger_mavg().fillna(df["close"])
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    df["rel_sep"] = (df[f"ema{EMA_MID}"] - df[f"ema{EMA_SLOW}"]).abs() / df["close"].replace(0, 1e-12)

    ml_cols = ["open","high","low","close","volume",
               f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}",
               "rsi","macd_diff","bb_upper","bb_lower","bb_mavg","bb_width","rel_sep"]

    for c in ml_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

# ---------------- ML ----------------
def _build_ml_dataset(df: pd.DataFrame):
    df2 = df.copy().reset_index(drop=True)
    features = ["open","high","low","close","volume",
                f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}",
                "rsi","macd_diff","bb_upper","bb_lower","bb_mavg","bb_width","rel_sep"]

    for c in features:
        df2[c] = df2.get(c, 0.0)

    y = (df2["close"].shift(-1) > df2["close"]).astype(int)
    X = df2.iloc[:-1].copy()
    y = y.iloc[:-1].copy()

    if len(X) > ML_MAX_SAMPLES:
        X = X.tail(ML_MAX_SAMPLES).reset_index(drop=True)
        y = y.tail(ML_MAX_SAMPLES).reset_index(drop=True)

    return X, y

def train_ml_for_symbol(df: pd.DataFrame, symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return False
    try:
        X, y = _build_ml_dataset(df)
        if len(X) < ML_MIN_TRAINED_SAMPLES or len(y.unique()) < 2:
            ml_model_ready[symbol] = False
            return False
        model = RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH, random_state=42)
        model.fit(X, y)
        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True
        return True
    except:
        ml_model_ready[symbol] = False
        return False

def ml_predict_prob(symbol: str, last_row: pd.Series) -> float:
    try:
        if not ml_model_ready.get(symbol):
            return None
        model, cols = ml_models.get(symbol, (None, None))
        if model is None:
            return None

        Xrow = [float(last_row.get(c, 0.0)) for c in cols]
        prob_up = float(model.predict_proba([Xrow])[0][1])
        return prob_up
    except:
        return None

# ---------------- Fallback ----------------
def prune_sent_timestamps():
    cutoff = time.time() - FALLBACK_WINDOW_SEC
    while sent_timestamps and sent_timestamps[0] < cutoff:
        sent_timestamps.popleft()

def check_and_activate_fallback():
    prune_sent_timestamps()
    global fallback_active_until
    if len(sent_timestamps) < MIN_SIGNALS_PER_HOUR:
        fallback_active_until = time.time() + FALLBACK_DURATION_SECONDS

def is_fallback_active():
    return time.time() < fallback_active_until

# ---------------- Gerar sinal ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < EMA_SLOW + 5:
            return None
        now = df.iloc[-1]
        candle_id = int(now["epoch"]) - (int(now["epoch"]) % GRANULARITY_SECONDS)
        if last_signal_candle.get(symbol) == candle_id:
            return None

        ema_fast = float(now[f"ema{EMA_FAST}"])
        ema_mid = float(now[f"ema{EMA_MID}"])
        ema_slow = float(now[f"ema{EMA_SLOW}"])

        triple_up = (ema_fast > ema_mid > ema_slow)
        triple_down = (ema_fast < ema_mid < ema_slow)

        close = float(now["close"])
        bb_upper = float(now["bb_upper"])
        bb_lower = float(now["bb_lower"])
        width = bb_upper - bb_lower if (bb_upper - bb_lower) != 0 else 1.0
        perto_lower = close <= bb_lower + width * BB_PROXIMITY_PCT
        perto_upper = close >= bb_upper - width * BB_PROXIMITY_PCT

        bullish = now["close"] > now["open"]
        bearish = now["close"] < now["open"]

        rsi_now = float(now["rsi"]) if not pd.isna(now["rsi"]) else 50.0
        macd_diff = now.get("macd_diff")

        macd_buy_ok = True if macd_diff is None or pd.isna(macd_diff) else (macd_diff > -MACD_TOLERANCE)
        macd_sell_ok = True if macd_diff is None or pd.isna(macd_diff) else (macd_diff < MACD_TOLERANCE)

        buy_ok = triple_up and (bullish or perto_lower) and rsi_now <= RSI_BUY_MAX and macd_buy_ok
        sell_ok = triple_down and (bearish or perto_upper) and rsi_now >= RSI_SELL_MIN and macd_sell_ok

        if is_fallback_active():
            if not buy_ok and ema_mid > ema_slow and bullish:
                buy_ok = True
            if not sell_ok and ema_mid < ema_slow and bearish:
                sell_ok = True

        if not (buy_ok or sell_ok):
            return None

        tipo = "COMPRA" if buy_ok else "VENDA"
        last_signal_state[symbol] = tipo
        last_signal_candle[symbol] = candle_id
        return {"tipo": tipo, "candle_id": candle_id}
    except:
        return None

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    try:
        df.tail(MAX_CANDLES).to_csv(DATA_DIR / f"candles_{symbol}.csv", index=False)
    except:
        pass

# ---------------- MENSAGENS ----------------

def convert_utc_to_brasilia(dt_utc: datetime) -> str:
    """
    Converte datetime UTC → Horário de Brasília automaticamente.
    Brasília pode estar em UTC-3 ou UTC-2 (horário de verão, se existir).
    Para 2025: UTC-3 fixo.
    """
    # UTC-3
    brasilia = dt_utc - timedelta(hours=3)
    return brasilia.strftime("%H:%M:%S") + " BRT"

def format_start_message() -> str:
    return (
        "🟢 <b>BOT INICIADO!</b>\n\n"
        "O sistema está ativo e monitorando os pares configurados.\n"
        "Os horários enviados serão ajustados automaticamente para <b>Horário de Brasília</b>."
    )

def format_signal_message(symbol: str, tipo: str, entry_dt_utc: datetime, ml_prob: float | None) -> str:
    """
    Agora converte o horário UTC para o horário de Brasília.
    """
    pair = html.escape(human_pair(symbol))
    direction_emoji = "🟢" if tipo == "COMPRA" else "🔴"
    direction_label = "COMPRA" if tipo == "COMPRA" else "VENDA"

    # *** ALTERAÇÃO PEDIDA ***
    entry_brasilia = convert_utc_to_brasilia(entry_dt_utc)

    if ml_prob is None:
        ml_text = "N/A"
    else:
        ml_text = f"{int(round(ml_prob * 100))}%"

    text = (
        f"💱 <b>{pair}</b>\n\n"
        f"📈 DIREÇÃO: <b>{direction_emoji} {direction_label}</b>\n"
        f"⏱ ENTRADA: <b>{entry_brasilia}</b>\n\n"
        f"🤖 ML: <b>{ml_text}</b>"
    )
    return text

# ---------------- MONITOR WEBSOCKET ----------------
async def monitor_symbol(symbol: str):
    df = pd.DataFrame()
    csv_path = DATA_DIR / f"candles_{symbol}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = calcular_indicadores(df)
            if len(df) > MAX_CANDLES:
                df = df.tail(MAX_CANDLES).reset_index(drop=True)
            historical_loaded[symbol] = True
        except:
            pass

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=30, ping_timeout=10) as ws:
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await asyncio.wait_for(ws.recv(), timeout=60)

                if not historical_loaded.get(symbol, False):
                    for attempt in range(HISTORY_MAX_TRIES):
                        await ws.send(json.dumps({
                            "ticks_history": symbol,
                            "count": INITIAL_HISTORY_COUNT,
                            "end": "latest",
                            "granularity": GRANULARITY_SECONDS,
                            "style": "candles"
                        }))
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        except asyncio.TimeoutError:
                            await asyncio.sleep(1 + random.random()*2)
                            continue
                        try:
                            data = json.loads(raw)
                        except:
                            await asyncio.sleep(1 + random.random()*2)
                            continue

                        candles = []
                        if isinstance(data, dict):
                            if "history" in data and isinstance(data["history"], dict) and "candles" in data["history"]:
                                candles = data["history"]["candles"]
                            elif "candles" in data and isinstance(data["candles"], list):
                                candles = data["candles"]
                        if candles:
                            df = pd.DataFrame(candles)
                            break
                        await asyncio.sleep(1 + random.random()*2)

                    df = calcular_indicadores(df)
                    historical_loaded[symbol] = True
                    save_last_candles(df, symbol)

                if not live_subscribed.get(symbol, False):
                    await ws.send(json.dumps({
                        "ticks_history": symbol,
                        "style": "candles",
                        "granularity": GRANULARITY_SECONDS,
                        "end": "latest",
                        "subscribe": 1
                    }))
                    live_subscribed[symbol] = True

                # Recepção contínua
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=600)
                    try:
                        msg = json.loads(raw)
                    except:
                        continue

                    candle = None
                    if "candle" in msg and isinstance(msg["candle"], dict):
                        candle = msg["candle"]
                    elif "ohlc" in msg and isinstance(msg["ohlc"], dict):
                        candle = msg["ohlc"]
                    elif "history" in msg and isinstance(msg["history"], dict) and "candles" in msg["history"]:
                        candle = msg["history"]["candles"][-1]
                    elif "candles" in msg and isinstance(msg["candles"], list) and len(msg["candles"]) > 0:
                        candle = msg["candles"][-1]

                    if not candle:
                        continue

                    try:
                        epoch = int(candle.get("epoch"))
                        if epoch % GRANULARITY_SECONDS != 0:
                            continue
                        open_p = float(candle.get("open", 0.0))
                        high_p = float(candle.get("high", 0.0))
                        low_p = float(candle.get("low", 0.0))
                        close_p = float(candle.get("close", 0.0))
                        volume_p = float(candle.get("volume", 0.0)) if candle.get("volume") else 0.0
                    except:
                        continue

                    df.loc[len(df)] = {
                        "epoch": epoch,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": volume_p
                    }

                    if len(df) > MAX_CANDLES:
                        df = df.tail(MAX_CANDLES).reset_index(drop=True)

                    df = calcular_indicadores(df)
                    save_last_candles(df, symbol)

                    # ML incremental
                    try:
                        samples = len(df)
                        last_trained = ml_trained_samples.get(symbol, 0)
                        if ML_ENABLED and samples >= ML_MIN_TRAINED_SAMPLES and samples >= last_trained + ML_RETRAIN_INTERVAL:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, train_ml_for_symbol, df.copy(), symbol)
                            ml_trained_samples[symbol] = samples
                    except:
                        pass

                    sinal = gerar_sinal(df, symbol)
                    if sinal:
                        ml_prob = None
                        if ML_ENABLED and ml_model_ready.get(symbol):
                            try:
                                ml_prob = ml_predict_prob(symbol, df.iloc[-1])
                            except:
                                ml_prob = None

                        entry_dt_utc = datetime.utcfromtimestamp(epoch).replace(tzinfo=timezone.utc)

                        msg_text = format_signal_message(symbol, sinal["tipo"], entry_dt_utc, ml_prob)

                        send_telegram(msg_text, symbol=symbol)

                        last_signal_time[symbol] = time.time()
                        last_signal_candle[symbol] = sinal["candle_id"]
                        sent_timestamps.append(time.time())
                        prune_sent_timestamps()
                        check_and_activate_fallback()

        except Exception as e:
            log(f"[{symbol}] Erro: {e}\n{traceback.format_exc()}", "error")
            await asyncio.sleep(5)

# ---------------- LOOP PRINCIPAL ----------------
async def main():
    start_msg = format_start_message()
    send_telegram(start_msg, bypass_throttle=True)

    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
