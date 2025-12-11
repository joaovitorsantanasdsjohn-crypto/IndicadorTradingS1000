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
from flask import Flask
import threading

# ---------------- ML ----------------
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
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Parâmetros ----------------
BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

MIN_CANDLES_BETWEEN_SIGNALS = int(os.getenv("MIN_CANDLES_BETWEEN_SIGNALS", "4"))

EMA_FAST = 9
EMA_MID = 20
EMA_SLOW = 200

ML_ENABLED = SKLEARN_AVAILABLE
ML_N_ESTIMATORS = 40
ML_MAX_DEPTH = 4
ML_MIN_TRAINED_SAMPLES = 200
ML_MAX_SAMPLES = 2000
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_RETRAIN_INTERVAL = 50

MIN_SIGNALS_PER_HOUR = 5
FALLBACK_WINDOW_SEC = 3600
FALLBACK_DURATION_SECONDS = 15 * 60
INITIAL_HISTORY_COUNT = int(os.getenv("INITIAL_HISTORY_COUNT", "500"))
MAX_CANDLES = 300

# ---------------- Estado ----------------
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}
ml_models = {}
ml_model_ready = {}

sent_timestamps = deque()
fallback_active_until = 0.0
ml_trained_samples = {s: 0 for s in SYMBOLS}

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
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
        log("⚠️ Telegram não configurado", "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
    except Exception as e:
        log(f"[TG] Erro ao enviar: {e}", "error")

# ---------------- Utils ----------------
def human_pair(symbol: str) -> str:
    return symbol.replace("frx", "")

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values("epoch").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)

    df[f"ema{EMA_FAST}"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df[f"ema{EMA_MID}"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df[f"ema{EMA_SLOW}"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], 14).rsi().fillna(50.0)

    try:
        macd = MACD(df["close"], 26, 12, 9)
        df["macd_diff"] = macd.macd_diff().fillna(0.0)
    except Exception:
        df["macd_diff"] = 0.0

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband().fillna(df["close"])
    df["bb_lower"] = bb.bollinger_lband().fillna(df["close"])
    df["bb_mavg"] = bb.bollinger_mavg().fillna(df["close"])
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    df["rel_sep"] = (df[f"ema{EMA_MID}"] - df[f"ema{EMA_SLOW}"]).abs() / df["close"].replace(0, 1e-12)

    return df

# ---------------- ML ----------------
def _build_ml_dataset(df: pd.DataFrame):
    df2 = df.copy().reset_index(drop=True)
    features = [
        "open","high","low","close","volume",
        f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}",
        "rsi", "macd_diff", "bb_upper", "bb_lower",
        "bb_mavg", "bb_width", "rel_sep"
    ]

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
            log(f"[ML {symbol}] Dados insuficientes ({len(X)})", "info")
            return False

        model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            random_state=42
        )

        model.fit(X, y)
        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True

        log(f"[ML {symbol}] Modelo treinado ({len(X)} samples).", "info")
        return True

    except Exception as e:
        ml_model_ready[symbol] = False
        log(f"[ML {symbol}] Erro ao treinar: {e}", "error")
        return False

def ml_predict_prob(symbol: str, last_row: pd.Series):
    try:
        if not ml_model_ready.get(symbol):
            return None

        model, cols = ml_models.get(symbol, (None, None))
        if model is None:
            return None

        Xrow = [float(last_row.get(c, 0.0)) for c in cols]
        return float(model.predict_proba([Xrow])[0][1])

    except Exception as e:
        log(f"[ML {symbol}] Erro na previsão: {e}", "warning")
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
        log("⚠️ Fallback ativado: poucos sinais.", "warning")

def is_fallback_active():
    return time.time() < fallback_active_until

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    try:
        df.tail(MAX_CANDLES).to_csv(DATA_DIR / f"candles_{symbol}.csv", index=False)
    except Exception as e:
        log(f"[{symbol}] Erro ao salvar: {e}", "error")

# ---------------- Mensagens ----------------
def convert_utc_to_brasilia(dt_utc: datetime) -> str:
    brasilia = dt_utc - timedelta(hours=3)
    return brasilia.strftime("%H:%M:%S") + " BRT"

def format_signal_message(symbol: str, tipo: str, entry_dt_utc: datetime, ml_prob):
    pair = html.escape(human_pair(symbol))
    emoji = "🟢" if tipo == "COMPRA" else "🔴"
    entry_brasilia = convert_utc_to_brasilia(entry_dt_utc)
    ml_txt = "N/A" if ml_prob is None else f"{int(round(ml_prob * 100))}%"

    return (
        f"💱 <b>{pair}</b>\n\n"
        f"📈 DIREÇÃO: <b>{emoji} {tipo}</b>\n"
        f"⏱ ENTRADA: <b>{entry_brasilia}</b>\n\n"
        f"🤖 ML: <b>{ml_txt}</b>"
    )

def format_start_message():
    return (
        "🟢 <b>BOT INICIADO!</b>\n\n"
        "O sistema está ativo e monitorando os pares.\n"
        "Horários ajustados para <b>Horário de Brasília</b>.\n"
        "Entradas na <b>abertura da próxima vela</b> (M5)."
    )

# ---------------- GERAÇÃO DE SINAL ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < EMA_SLOW + 5:
            log(f"[{symbol}] Poucos candles ({len(df)})", "info")
            return None

        now = df.iloc[-1]
        candle_id = int(now["epoch"]) - (int(now["epoch"]) % GRANULARITY_SECONDS)

        if last_signal_candle.get(symbol) == candle_id:
            return None

        ema_fast = float(now[f"ema{EMA_FAST}"])
        ema_mid = float(now[f"ema{EMA_MID}"])
        ema_slow = float(now[f"ema{EMA_SLOW}"])

        triple_up = ema_fast > ema_mid > ema_slow
        triple_down = ema_fast < ema_mid < ema_slow

        close = float(now["close"])
        bb_upper = float(now["bb_upper"])
        bb_lower = float(now["bb_lower"])
        width = bb_upper - bb_lower if bb_upper - bb_lower != 0 else 1.0

        perto_lower = close <= bb_lower + width * BB_PROXIMITY_PCT
        perto_upper = close >= bb_upper - width * BB_PROXIMITY_PCT

        bullish = now["close"] > now["open"]
        bearish = now["close"] < now["open"]

        rsi_now = float(now["rsi"]) if not pd.isna(now["rsi"]) else 50.0
        macd_diff = now.get("macd_diff")

        macd_buy_ok = True if macd_diff is None or pd.isna(macd_diff) else macd_diff > -MACD_TOLERANCE
        macd_sell_ok = True if macd_diff is None or pd.isna(macd_diff) else macd_diff < MACD_TOLERANCE

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
        return {"tipo": tipo, "candle_id": candle_id}

    except Exception as e:
        log(f"[{symbol}] Erro gerar_sinal: {e}", "error")
        return None

# ---------------- MONITOR WEBSOCKET ----------------
async def monitor_symbol(symbol: str):
    columns = ["epoch","open","high","low","close","volume"]
    df = pd.DataFrame(columns=columns)

    csv_path = DATA_DIR / f"candles_{symbol}.csv"
    if csv_path.exists():
        try:
            tmp = pd.read_csv(csv_path)
            if not tmp.empty:
                tmp = tmp.loc[:, tmp.columns.intersection(columns)]
                df = pd.DataFrame(tmp, columns=columns)
                df = calcular_indicadores(df)
        except:
            pass

    connect_attempt = 0
    backoff_base = 2.0

    while True:
        try:
            connect_attempt += 1

            if connect_attempt > 1:
                delay = min(120, backoff_base ** min(connect_attempt, 6)) + random.random()
                log(f"{symbol} | Backoff {delay:.1f}s", "info")
                await asyncio.sleep(delay)

            log(f"{symbol} | Conectando WS...", "info")

            async with websockets.connect(
                WS_URL,
                ping_interval=40,
                ping_timeout=20,
                max_size=None
            ) as ws:

                connect_attempt = 0
                log(f"{symbol} | WS Conectado!", "info")

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await ws.recv()

                sub = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": INITIAL_HISTORY_COUNT,
                    "end": "latest",
                    "style": "candles",
                    "granularity": GRANULARITY_SECONDS,
                    "subscribe": 1
                }

                await ws.send(json.dumps(sub))
                log(f"{symbol} | Histórico solicitado + subscribe", "info")

                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=900)

                    try:
                        msg = json.loads(raw)
                    except:
                        continue

                    candle = None

                    if "history" in msg and "candles" in msg["history"]:
                        try:
                            candles = msg["history"]["candles"]
                            hist = pd.DataFrame(candles)
                            hist = hist.iloc[:, :6]
                            hist.columns = columns
                            df = calcular_indicadores(hist)
                            save_last_candles(df, symbol)

                            if ML_ENABLED:
                                train_ml_for_symbol(df, symbol)
                                ml_trained_samples[symbol] = len(df)

                        except Exception as e:
                            log(f"{symbol} | Erro history: {e}", "warning")

                        continue

                    if "candle" in msg:
                        candle = msg["candle"]
                    elif "ohlc" in msg:
                        candle = msg["ohlc"]
                    elif "candles" in msg and msg["candles"]:
                        candle = msg["candles"][-1]

                    if not candle:
                        continue

                    try:
                        epoch = int(candle["epoch"])
                        if epoch % GRANULARITY_SECONDS != 0:
                            continue

                        new_row = {
                            "epoch": epoch,
                            "open": float(candle.get("open", 0)),
                            "high": float(candle.get("high", 0)),
                            "low": float(candle.get("low", 0)),
                            "close": float(candle.get("close", 0)),
                            "volume": float(candle.get("volume", 0))
                        }

                    except:
                        continue

                    df.loc[len(df)] = new_row
                    if len(df) > MAX_CANDLES:
                        df = df.tail(MAX_CANDLES).reset_index(drop=True)

                    df = calcular_indicadores(df)
                    save_last_candles(df, symbol)

                    sinal = gerar_sinal(df, symbol)
                    if sinal:
                        ml_prob = None
                        if ML_ENABLED and ml_model_ready.get(symbol):
                            ml_prob = ml_predict_prob(symbol, df.iloc[-1])

                        if ml_prob is not None:
                            if sinal["tipo"] == "COMPRA" and ml_prob < ML_CONF_THRESHOLD:
                                continue
                            if sinal["tipo"] == "VENDA" and (1 - ml_prob) < ML_CONF_THRESHOLD:
                                continue

                        next_epoch = epoch + GRANULARITY_SECONDS
                        entry_utc = datetime.fromtimestamp(next_epoch, tz=timezone.utc)

                        msg = format_signal_message(symbol, sinal["tipo"], entry_utc, ml_prob)
                        send_telegram(msg, symbol)

                        last_signal_candle[symbol] = sinal["candle_id"]
                        last_signal_time[symbol] = time.time()

                        sent_timestamps.append(time.time())
                        prune_sent_timestamps()
                        check_and_activate_fallback()

        except Exception as e:
            log(f"{symbol} | ERRO WS: {e}", "error")
            await asyncio.sleep(3 + random.random())

# ---------------- MAIN LOOP ----------------
async def main():
    msg = format_start_message()
    send_telegram(msg, bypass_throttle=True)

    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

# ---------------- Flask ----------------
app = Flask(__name__)

@app.get("/")
def home():
    return "BOT ONLINE", 200

def run_flask():
    port = int(os.getenv("PORT", 10000))
    log(f"Flask iniciado porta {port}", "info")
    app.run(host="0.0.0.0", port=port)

# ---------------- STARTUP ----------------
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    try:
        asyncio.run(main())
    except Exception as e:
        log(f"Erro fatal main: {e}", "error")
