# indicador_s1000_fixed_v2.py
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

# ---------------- Parâmetros (ajustáveis via env) ----------------
BB_PROXIMITY_PCT = float(os.getenv("BB_PROXIMITY_PCT", "0.20"))
RSI_BUY_MAX = int(os.getenv("RSI_BUY_MAX", "52"))
RSI_SELL_MIN = int(os.getenv("RSI_SELL_MIN", "48"))
MACD_TOLERANCE = float(os.getenv("MACD_TOLERANCE", "0.002"))

# EMA params (keep EMA_SLOW reasonable for availability)
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_MID = int(os.getenv("EMA_MID", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))

# ---- Ajustes para aumentar volume de sinais (aplicados) ----
MIN_SECONDS_BETWEEN_SIGNALS = int(os.getenv("MIN_SECONDS_BETWEEN_SIGNALS", "3"))
MIN_CANDLES_BETWEEN_SIGNALS = int(os.getenv("MIN_CANDLES_BETWEEN_SIGNALS", "2"))
REL_EMA_SEP_PCT = float(os.getenv("REL_EMA_SEP_PCT", "5e-06"))
FORCE_MIN = int(os.getenv("FORCE_MIN", "35"))
MICRO_FORCE_ALLOW_THRESHOLD = int(os.getenv("MICRO_FORCE_ALLOW_THRESHOLD", "25"))

ML_ENABLED = SKLEARN_AVAILABLE and os.getenv("ENABLE_ML", "1") != "0"
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "40"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "4"))
ML_MIN_TRAINED_SAMPLES = int(os.getenv("ML_MIN_TRAINED_SAMPLES", "200"))  # conforme ajuste
ML_MAX_SAMPLES = int(os.getenv("ML_MAX_SAMPLES", "2000"))
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_RETRAIN_INTERVAL = int(os.getenv("ML_RETRAIN_INTERVAL", "50"))

MIN_SIGNALS_PER_HOUR = int(os.getenv("MIN_SIGNALS_PER_HOUR", "5"))
FALLBACK_WINDOW_SEC = int(os.getenv("FALLBACK_WINDOW_SEC", "3600"))
FALLBACK_DURATION_SECONDS = int(os.getenv("FALLBACK_DURATION_SECONDS", str(15*60)))
INITIAL_HISTORY_COUNT = int(os.getenv("INITIAL_HISTORY_COUNT", "1200"))  # aumentado para coletar mais histórico
MAX_CANDLES = int(os.getenv("MAX_CANDLES", "300"))

# WebSocket timeouts (maiores para tolerar lentidão)
WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "60"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "30"))
RECV_TIMEOUT = int(os.getenv("RECV_TIMEOUT", "1200"))

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
        log("⚠️ Telegram não configurado (TELEGRAM_TOKEN/CHAT_ID faltando).", "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
    except Exception as e:
        log(f"[TG] Erro ao enviar: {e}", "error")

# ---------------- Utilitários ----------------
def human_pair(symbol: str) -> str:
    return symbol.replace("frx", "")

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values("epoch").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)

    try:
        df[f"ema{EMA_FAST}"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    except Exception:
        df[f"ema{EMA_FAST}"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()

    try:
        df[f"ema{EMA_MID}"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    except Exception:
        df[f"ema{EMA_MID}"] = df["close"].ewm(span=EMA_MID, adjust=False).mean()

    try:
        df[f"ema{EMA_SLOW}"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    except Exception:
        df[f"ema{EMA_SLOW}"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    try:
        df["rsi"] = RSIIndicator(df["close"], 14).rsi().fillna(50.0)
    except Exception:
        df["rsi"] = 50.0

    try:
        macd = MACD(df["close"], 26, 12, 9)
        df["macd_diff"] = macd.macd_diff().fillna(0.0)
    except Exception:
        df["macd_diff"] = 0.0

    try:
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband().fillna(df["close"])
        df["bb_lower"] = bb.bollinger_lband().fillna(df["close"])
        df["bb_mavg"] = bb.bollinger_mavg().fillna(df["close"])
    except Exception:
        df["bb_upper"] = df["close"]
        df["bb_lower"] = df["close"]
        df["bb_mavg"] = df["close"]

    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["rel_sep"] = (df[f"ema{EMA_MID}"] - df[f"ema{EMA_SLOW}"]).abs() / df["close"].replace(0, 1e-12)

    return df

# ---------------- ML ----------------
def _build_ml_dataset(df: pd.DataFrame):
    df2 = df.copy().reset_index(drop=True)
    features = [
        "open", "high", "low", "close", "volume",
        f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}",
        "rsi", "macd_diff", "bb_upper", "bb_lower", "bb_mavg", "bb_width", "rel_sep"
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
            log(f"[ML {symbol}] Dados insuficientes para treinar ({len(X)} samples).", "info")
            return False
        model = RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH, random_state=42)
        model.fit(X, y)
        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True
        log(f"[ML {symbol}] Modelo treinado ({len(X)} samples).", "info")
        return True
    except Exception as e:
        ml_model_ready[symbol] = False
        log(f"[ML {symbol}] Erro ao treinar ML: {e}", "error")
        return False

def ml_predict_prob(symbol: str, last_row: pd.Series) -> float:
    try:
        if not ml_model_ready.get(symbol):
            return None
        model, cols = ml_models.get(symbol, (None, None))
        if model is None:
            return None
        Xrow = [float(last_row.get(c, 0.0)) for c in cols]
        return float(model.predict_proba([Xrow])[0][1])
    except Exception as e:
        log(f"[ML {symbol}] Erro ao prever prob: {e}", "warning")
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
        log("⚠️ Fallback ativado (poucos sinais recentes).", "warning")

def is_fallback_active():
    return time.time() < fallback_active_until

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    try:
        df.tail(MAX_CANDLES).to_csv(DATA_DIR / f"candles_{symbol}.csv", index=False)
    except Exception as e:
        log(f"[{symbol}] Erro ao salvar candles: {e}", "error")

# ---------------- Mensagens ----------------
def convert_utc_to_brasilia(dt_utc: datetime) -> str:
    brasilia = dt_utc - timedelta(hours=3)
    return brasilia.strftime("%H:%M:%S") + " BRT"

def format_signal_message(symbol: str, tipo: str, entry_dt_utc: datetime, ml_prob: float | None) -> str:
    pair = html.escape(human_pair(symbol))
    direction_emoji = "🟢" if tipo == "COMPRA" else "🔴"
    entry_brasilia = convert_utc_to_brasilia(entry_dt_utc)
    ml_text = "N/A" if ml_prob is None else f"{int(round(ml_prob * 100))}%"
    text = (
        f"💱 <b>{pair}</b>\n\n"
        f"📈 DIREÇÃO: <b>{direction_emoji} {tipo}</b>\n"
        f"⏱ ENTRADA: <b>{entry_brasilia}</b>\n\n"
        f"🤖 ML: <b>{ml_text}</b>"
    )
    return text

def format_start_message() -> str:
    return (
        "🟢 <b>BOT INICIADO!</b>\n\n"
        "O sistema está ativo e monitorando os pares configurados.\n"
        "Os horários enviados serão ajustados automaticamente para <b>Horário de Brasília</b>.\n"
        "Entradas serão disparadas para a <b>abertura da próxima vela</b> (M5)."
    )

# ---------------- Geração de sinais (robusta + menos restritiva) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if df is None or len(df) < 8:
            log(f"[{symbol}] Histórico muito curto ({0 if df is None else len(df)} candles).", "info")
            return None

        now = df.iloc[-1]
        candle_id = int(now["epoch"]) - (int(now["epoch"]) % GRANULARITY_SECONDS)

        # evita sinal duplicado na mesma vela
        if last_signal_candle.get(symbol) == candle_id:
            return None

        # verifica tempo mínimo entre sinais
        if time.time() - last_signal_time.get(symbol, 0) < MIN_SECONDS_BETWEEN_SIGNALS:
            log(f"[{symbol}] Ignorando sinal: ainda dentro do MIN_SECONDS_BETWEEN_SIGNALS.", "info")
            return None

        # verifica candles mínimos desde último sinal (por epoch)
        last_candle_epoch = last_signal_candle.get(symbol)
        if last_candle_epoch is not None:
            if (candle_id - last_candle_epoch) < (MIN_CANDLES_BETWEEN_SIGNALS * GRANULARITY_SECONDS):
                log(f"[{symbol}] Ignorando sinal: MIN_CANDLES_BETWEEN_SIGNALS não atingido.", "info")
                return None

        # get EMAs (fallback to ewm if NaN)
        def get_ema_col(df_local, span):
            col = f"ema{span}"
            if col in df_local.columns and not pd.isna(df_local[col].iloc[-1]):
                return float(df_local[col].iloc[-1])
            try:
                return float(df_local["close"].ewm(span=span, adjust=False).mean().iloc[-1])
            except Exception:
                return float(df_local["close"].iloc[-1])

        ema_fast = get_ema_col(df, EMA_FAST)
        ema_mid = get_ema_col(df, EMA_MID)
        ema_slow = get_ema_col(df, EMA_SLOW)

        triple_up = ema_fast > ema_mid > ema_slow
        triple_down = ema_fast < ema_mid < ema_slow

        close = float(now["close"])
        bb_upper = float(now.get("bb_upper", close))
        bb_lower = float(now.get("bb_lower", close))
        width = bb_upper - bb_lower if bb_upper - bb_lower != 0 else 1.0

        perto_lower = close <= bb_lower + width * BB_PROXIMITY_PCT
        perto_upper = close >= bb_upper - width * BB_PROXIMITY_PCT

        bullish = now["close"] > now["open"]
        bearish = now["close"] < now["open"]

        rsi_now = float(now["rsi"]) if not pd.isna(now.get("rsi")) else 50.0
        macd_diff = now.get("macd_diff")

        macd_buy_ok = True if macd_diff is None or pd.isna(macd_diff) else macd_diff > -MACD_TOLERANCE
        macd_sell_ok = True if macd_diff is None or pd.isna(macd_diff) else macd_diff < MACD_TOLERANCE

        # rel_sep current
        rel_sep = abs(ema_mid - ema_slow) / (close if close != 0 else 1e-12)

        # basic conditions
        buy_ok = triple_up and (bullish or perto_lower) and rsi_now <= RSI_BUY_MAX and macd_buy_ok
        sell_ok = triple_down and (bearish or perto_upper) and rsi_now >= RSI_SELL_MIN and macd_sell_ok

        # allow looser EMAs if relative separation small but other forces present
        # se rel_sep muito pequena, aplicamos heurística de 'micro force' para liberar sinais mais frequentes
        if not (buy_ok or sell_ok):
            # micro-force: se diferenciação rápida entre fast e mid for alta relativa, permitir
            micro_sep = abs(ema_fast - ema_mid) / (close if close != 0 else 1e-12)
            # transformar thresholds em valores interpretáveis:
            # MICRO_FORCE_ALLOW_THRESHOLD é um inteiro; transformamos numa escala pequena
            micro_thresh = MICRO_FORCE_ALLOW_THRESHOLD * 1e-05
            force_thresh = FORCE_MIN * 1e-04

            if rel_sep < REL_EMA_SEP_PCT:
                # se micro_sep excede micro_thresh e a vela é direcional, liberar conforme fallback
                if micro_sep > micro_thresh and (bullish or bearish):
                    if bullish:
                        buy_ok = True
                    if bearish:
                        sell_ok = True
                # ou se rel_sep supera force_thresh (força maior) liberar
                if rel_sep > force_thresh:
                    if bullish:
                        buy_ok = True
                    if bearish:
                        sell_ok = True

        if is_fallback_active():
            if not buy_ok and ema_mid > ema_slow and bullish:
                buy_ok = True
            if not sell_ok and ema_mid < ema_slow and bearish:
                sell_ok = True

        if not (buy_ok or sell_ok):
            log(f"[{symbol}] Sem condição para sinal (triple_up={triple_up} triple_down={triple_down} rsi={rsi_now:.2f} rel_sep={rel_sep:.2e}).", "info")
            return None

        tipo = "COMPRA" if buy_ok else "VENDA"
        return {"tipo": tipo, "candle_id": candle_id}

    except Exception as e:
        log(f"[{symbol}] Erro gerar_sinal: {e} | {traceback.format_exc()}", "error")
        return None

# ---------------- Monitor WebSocket (robusto) ----------------
async def monitor_symbol(symbol: str):
    columns = ["epoch", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(columns=columns)

    csv_path = DATA_DIR / f"candles_{symbol}.csv"
    if csv_path.exists():
        try:
            tmp = pd.read_csv(csv_path)
            if not tmp.empty:
                tmp = tmp.loc[:, tmp.columns.intersection(columns)]
                if not tmp.empty:
                    df = pd.DataFrame(tmp, columns=columns)
                    df = calcular_indicadores(df)
                    if len(df) > MAX_CANDLES:
                        df = df.tail(MAX_CANDLES).reset_index(drop=True)
                    log(f"{symbol} | Histórico carregado do CSV ({len(df)} candles).", "info")
        except Exception as e:
            log(f"{symbol} | ERRO ao ler CSV histórico: {e}", "warning")

    connect_attempt = 0
    backoff_base = 2.0

    while True:
        try:
            connect_attempt += 1
            if connect_attempt > 1:
                delay = min(120, backoff_base ** min(connect_attempt, 6)) + random.random()
                log(f"{symbol} | Aguardando {delay:.1f}s antes de nova tentativa (attempt {connect_attempt}).", "info")
                await asyncio.sleep(delay)

            log(f"{symbol} | Conectando ao WS (attempt {connect_attempt})...", "info")
            async with websockets.connect(WS_URL, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT, max_size=None) as ws:
                log(f"{symbol} | WS conectado.", "info")

                # authorize
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                try:
                    auth_raw = await asyncio.wait_for(ws.recv(), timeout=20)
                    try:
                        auth_msg = json.loads(auth_raw)
                        log(f"{symbol} | Autorize response: {auth_msg.get('msg_type', 'NA')}", "info")
                    except Exception:
                        log(f"{symbol} | Autorize raw recebido (não JSON).", "info")
                except asyncio.TimeoutError:
                    log(f"{symbol} | Timeout aguardando authorize response.", "warning")

                # ---------------------------
                # 1) solicita histórico (sem subscribe) - retry até 3 vezes se vier muito curto
                # ---------------------------
                history_attempts = 0
                history_ok = False
                while history_attempts < 3 and not history_ok:
                    history_attempts += 1
                    hist_req = {
                        "ticks_history": symbol,
                        "adjust_start_time": 1,
                        "count": INITIAL_HISTORY_COUNT,
                        "end": "latest",
                        "style": "candles",
                        "granularity": GRANULARITY_SECONDS
                    }
                    await ws.send(json.dumps(hist_req))
                    log(f"{symbol} | Histórico solicitado ({INITIAL_HISTORY_COUNT} candles) (attempt {history_attempts}).", "info")
                    # aguardar resposta de history (tempo curto)
                    try:
                        raw_hist = await asyncio.wait_for(ws.recv(), timeout=20)
                    except asyncio.TimeoutError:
                        log(f"{symbol} | Timeout aguardando resposta de histórico (attempt {history_attempts}).", "warning")
                        await asyncio.sleep(0.5 + random.random()*0.5)
                        continue

                    try:
                        msg_hist = json.loads(raw_hist)
                    except Exception:
                        log(f"{symbol} | Resposta histórico não-JSON (attempt {history_attempts}).", "warning")
                        await asyncio.sleep(0.5 + random.random()*0.5)
                        continue

                    candles_candidate = None
                    if "history" in msg_hist and isinstance(msg_hist.get("history"), dict) and "candles" in msg_hist["history"]:
                        candles_candidate = msg_hist["history"]["candles"]
                    elif "candles" in msg_hist and isinstance(msg_hist.get("candles"), list):
                        candles_candidate = msg_hist["candles"]

                    if isinstance(candles_candidate, list) and len(candles_candidate) > 0:
                        hist = pd.DataFrame(candles_candidate)
                        hist = hist.loc[:, hist.columns.intersection(columns)]
                        if not hist.empty:
                            hist = hist.sort_values("epoch").reset_index(drop=True)
                            df = pd.DataFrame(hist, columns=columns)
                            df = calcular_indicadores(df)
                            if len(df) > MAX_CANDLES:
                                df = df.tail(MAX_CANDLES).reset_index(drop=True)
                            save_last_candles(df, symbol)
                            log(f"{symbol} | Histórico recebido ({len(df)} candles).", "info")
                            # se veio muito curto (ex.: 1 vela), retry para aumentar chance de histórico completo
                            if len(df) < max(EMA_SLOW, 30):
                                log(f"{symbol} | Histórico curto ({len(df)}). Tentando solicitar novamente para completar...", "warning")
                                await asyncio.sleep(0.6 + random.random()*0.6)
                                continue
                            history_ok = True
                            # treinar ML inicial se possível
                            if ML_ENABLED:
                                try:
                                    loop = asyncio.get_event_loop()
                                    await loop.run_in_executor(None, train_ml_for_symbol, df.copy(), symbol)
                                    ml_trained_samples[symbol] = len(df)
                                except Exception as e:
                                    log(f"{symbol} | Erro retrain ML inicial: {e}", "warning")
                    else:
                        log(f"{symbol} | Histórico recebido sem candles (attempt {history_attempts}).", "warning")
                        await asyncio.sleep(0.6 + random.random()*0.6)

                if not history_ok:
                    log(f"{symbol} | Histórico inicial incompleto após retries; continuaremos e aguardaremos candles ao vivo.", "warning")

                # ---------------------------
                # 2) agora envia subscribe para atualizações ao vivo (separado)
                # ---------------------------
                try:
                    subscribe_msg = {
                        "ticks_history": symbol,
                        "style": "candles",
                        "granularity": GRANULARITY_SECONDS,
                        "subscribe": 1
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    log(f"{symbol} | Subscribe enviado para updates ao vivo.", "info")
                except Exception as e:
                    log(f"{symbol} | Erro ao enviar subscribe: {e}", "warning")

                connect_attempt = 0
                last_msg_ts = time.time()

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                        last_msg_ts = time.time()
                    except asyncio.TimeoutError:
                        raise Exception("Timeout prolongado, reconectar")

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        log(f"{symbol} | Mensagem WS inválida (não JSON).", "warning")
                        continue

                    # tratar histórico (caso ainda venha) - mesma lógica
                    if "history" in msg and isinstance(msg.get("history"), dict) and "candles" in msg["history"]:
                        try:
                            candles = msg["history"]["candles"]
                            if isinstance(candles, list) and len(candles) > 0:
                                hist = pd.DataFrame(candles)
                                hist = hist.loc[:, hist.columns.intersection(columns)]
                                hist = hist.rename(columns={c: c for c in hist.columns})
                                if not hist.empty:
                                    hist = hist.sort_values("epoch").reset_index(drop=True)
                                    df = pd.DataFrame(hist, columns=columns)
                                    df = calcular_indicadores(df)
                                    if len(df) > MAX_CANDLES:
                                        df = df.tail(MAX_CANDLES).reset_index(drop=True)
                                    save_last_candles(df, symbol)
                                    log(f"{symbol} | Histórico recebido via WS ({len(df)} candles).", "info")
                                    if ML_ENABLED:
                                        try:
                                            loop = asyncio.get_event_loop()
                                            await loop.run_in_executor(None, train_ml_for_symbol, df.copy(), symbol)
                                            ml_trained_samples[symbol] = len(df)
                                        except Exception as e:
                                            log(f"{symbol} | Erro retrain ML inicial: {e}", "warning")
                        except Exception as e:
                            log(f"{symbol} | Erro processando histórico WS: {e}", "warning")
                        continue

                    # candle / ohlc / candles updates
                    candle = None
                    if "candle" in msg and isinstance(msg.get("candle"), dict):
                        candle = msg["candle"]
                    elif "ohlc" in msg and isinstance(msg.get("ohlc"), dict):
                        candle = msg["ohlc"]
                    elif "candles" in msg and isinstance(msg.get("candles"), list) and msg["candles"]:
                        candle = msg["candles"][-1]
                    elif "tick" in msg and isinstance(msg.get("tick"), dict):
                        continue

                    if not candle:
                        continue

                    # parse candle and accept only closed candles aligned to granularity
                    try:
                        epoch = int(candle.get("epoch"))
                        if epoch % GRANULARITY_SECONDS != 0:
                            continue
                        open_p = float(candle.get("open", 0.0))
                        high_p = float(candle.get("high", 0.0))
                        low_p = float(candle.get("low", 0.0))
                        close_p = float(candle.get("close", 0.0))
                        volume_p = float(candle.get("volume", 0.0) if candle.get("volume") else 0.0)
                    except Exception:
                        log(f"{symbol} | Erro ao parsear candle: {traceback.format_exc()}", "warning")
                        continue

                    new_row = {"epoch": epoch, "open": open_p, "high": high_p, "low": low_p, "close": close_p, "volume": volume_p}
                    try:
                        if df.empty:
                            df = pd.DataFrame([new_row], columns=columns)
                        elif set(new_row.keys()) <= set(df.columns):
                            df.loc[len(df)] = new_row
                        else:
                            df = pd.concat([df, pd.DataFrame([new_row], columns=columns)], ignore_index=True)
                    except Exception:
                        df = pd.concat([df, pd.DataFrame([new_row], columns=columns)], ignore_index=True)

                    if len(df) > MAX_CANDLES:
                        df = df.tail(MAX_CANDLES).reset_index(drop=True)

                    df = calcular_indicadores(df)
                    save_last_candles(df, symbol)

                    log(f"🕯 {symbol} | Vela fechada recebida: epoch={epoch} O={open_p} H={high_p} L={low_p} C={close_p}", "info")

                    # incremental ML retrain
                    try:
                        samples = len(df)
                        last_trained = ml_trained_samples.get(symbol, 0)
                        if ML_ENABLED and samples >= ML_MIN_TRAINED_SAMPLES and samples >= last_trained + ML_RETRAIN_INTERVAL:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, train_ml_for_symbol, df.copy(), symbol)
                            ml_trained_samples[symbol] = samples
                    except Exception as e:
                        log(f"{symbol} | Erro retrain ML: {e}", "warning")

                    # gerar sinal
                    sinal = gerar_sinal(df, symbol)
                    if sinal:
                        ml_prob = None
                        if ML_ENABLED and ml_model_ready.get(symbol):
                            ml_prob = ml_predict_prob(symbol, df.iloc[-1])
                            log(f"[ML {symbol}] Prob_up={ml_prob}", "info")
                        # filtrar por ML
                        if ml_prob is not None:
                            if sinal["tipo"] == "COMPRA" and ml_prob < ML_CONF_THRESHOLD:
                                log(f"⛔ [{symbol}] ML bloqueou sinal COMPRA (prob_up={ml_prob:.3f})", "warning")
                                continue
                            elif sinal["tipo"] == "VENDA" and (1.0 - ml_prob) < ML_CONF_THRESHOLD:
                                log(f"⛔ [{symbol}] ML bloqueou sinal VENDA (prob_up={ml_prob:.3f})", "warning")
                                continue

                        next_candle_epoch = epoch + GRANULARITY_SECONDS
                        entry_dt_utc = datetime.fromtimestamp(next_candle_epoch, tz=timezone.utc)
                        msg_text = format_signal_message(symbol, sinal["tipo"], entry_dt_utc, ml_prob)
                        send_telegram(msg_text, symbol)
                        log(f"✅ [{symbol}] Sinal enviado: {sinal['tipo']} (entrada {convert_utc_to_brasilia(entry_dt_utc)})", "info")

                        last_signal_candle[symbol] = sinal["candle_id"]
                        last_signal_time[symbol] = time.time()
                        sent_timestamps.append(time.time())
                        prune_sent_timestamps()
                        check_and_activate_fallback()

        except Exception as e:
            log(f"{symbol} | ERRO WS (reconectando): {e}", "error")
            await asyncio.sleep(3 + random.random() * 2)

# ---------------- Loop principal ----------------
async def main():
    # Startup message + telegram health check
    start_msg = format_start_message()
    log("Iniciando bot...", "info")
    if TELEGRAM_TOKEN and CHAT_ID:
        log("📡 Telegram config detectada — tentando enviar mensagem de startup...", "info")
        send_telegram(start_msg, bypass_throttle=True)
    else:
        log("⚠️ TELEGRAM_TOKEN/CHAT_ID não presente — não será possível enviar sinais até configurar.", "warning")

    log("Iniciando tasks para todos os símbolos...", "info")
    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

# ---------------- Flask health-check ----------------
app = Flask(__name__)

@app.get("/")
def home():
    return "BOT ONLINE", 200

def run_flask():
    port = int(os.getenv("PORT", 10000))
    log(f"🔎 Flask HTTP health-check iniciado na porta {port}", "info")
    app.run(host="0.0.0.0", port=port)

# ---------------- STARTUP ----------------
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando por KeyboardInterrupt.", "info")
    except Exception as e:
        log(f"Erro fatal no loop principal: {e}", "error")
