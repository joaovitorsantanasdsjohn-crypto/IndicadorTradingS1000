# deriv_telegram_bot.py — LÓGICA A (Opção A — Precisão Profissional para FOREX M5)
# Ajustes técnicos: limites de memória, GC, robustez WS, limites ML.
# Modificações: histórico adaptado para Render FREE (count=500), ML com mais amostras e retrain mais frequente,
# parâmetros mais permissivos para aumentar volume de sinais (com proteções).
# Removido envio para Telegram de mensagens "sinal bloqueado pelo ML" — agora só log.
# Histórico inicial: tenta 5 vezes; se falhar, faz log e aguarda 10 minutos antes de tentar novamente (Opção 1).

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback
import math
from collections import deque
import html  # para escapar texto antes de enviar via HTML para Telegram
import gc

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

# ---------------- Parâmetros (ajustáveis) ----------------
BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

# Anti-spam / anti-duplicate (tempo)
MIN_SECONDS_BETWEEN_SIGNALS = 3           # cooldown entre envios (aplicado APÓS ML aprovar)
MIN_SECONDS_BETWEEN_OPPOSITE = 45        # evita compra->venda imediata (opposite cooldown)

# Anti-duplicate (velas): aguarda N velas entre sinais para o mesmo par
MIN_CANDLES_BETWEEN_SIGNALS = 4           # aumentado para 4 conforme solicitado

# EMA separation relative threshold (fraction of price)
REL_EMA_SEP_PCT = 3e-06                   # relaxado para aumentar volume
MICRO_FORCE_ALLOW_THRESHOLD = 15          # permite sinais menores

# Força mínima absoluta para envio
FORCE_MIN = 30                            # reduzido para aumentar volume

# ML params
ML_ENABLED = SKLEARN_AVAILABLE
ML_N_ESTIMATORS = 40
ML_MAX_DEPTH = 4
ML_MIN_TRAINED_SAMPLES = 50               # reduzido para treinar cedo (mais prob)
ML_CONF_THRESHOLD = 0.55                  # 55% conforme solicitado

# ML memory safety: cap de amostras para treinar
ML_MAX_SAMPLES = 2000                     # limite para não estourar memória
ML_RETRAIN_INTERVAL = 50                  # retrain quando houver X samples a mais

# Fallback adaptativo (garantir sinais/hora)
MIN_SIGNALS_PER_HOUR = 3
FALLBACK_WINDOW_SEC = 3600
FALLBACK_FORCE_MIN = 30
FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD = 20
FALLBACK_REL_EMA_SEP_PCT = 2e-05
FALLBACK_DURATION_SECONDS = 15 * 60

DEFAULT_EMA_SEP_SCALE = 0.01

# Initial history fetch count (ajustado para Render FREE; evita vazio mas mantém segurança)
INITIAL_HISTORY_COUNT = 500  # recomendado para Render FREE

# Histórico inicial: tentativas até N antes de pausar (por par)
HISTORY_MAX_TRIES = 5
HISTORY_RETRY_PAUSE_SEC = 10 * 60  # 10 minutos

# --- Novos parâmetros Profissionais (Opção A)
EMA_FAST = 9
EMA_MID = 20
EMA_SLOW = 200

ATR_WINDOW = 14
ATR_PCT_MIN = 0.00025
ATR_PCT_MAX = 0.02

CONSOL_WINDOW = 12
CONSOL_RANGE_PCT = 0.00018

EMA_ALIGNMENT_STRICTNESS = 0.00005

MIN_BODY_PCT = 0.00012
MAX_WICK_BODY_RATIO = 0.6
REQUIRE_CLOSE_OUTSIDE_BB = False

# ---------------- Estado ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}

# ML models per symbol
ml_models = {}
ml_model_ready = {}

# track sent timestamps globally to enforce MIN_SIGNALS_PER_HOUR fallback
sent_timestamps = deque()
fallback_active_until = 0.0

# pending signals
pending_signals = {}

# ---------- flags per símbolo para evitar repetidos heavy ops ----------
historical_loaded = {s: False for s in SYMBOLS}
live_subscribed = {s: False for s in SYMBOLS}
ml_trained_samples = {s: 0 for s in SYMBOLS}
notify_flags = {s: {"connected": False, "history": False, "ml": False, "subscribed": False} for s in SYMBOLS}
if "global" not in notify_flags:
    notify_flags["global"] = {}

# Contador de falhas consecutivas de histórico por símbolo (para logging claro)
history_fail_count = {s: 0 for s in SYMBOLS}

# ---------------- Limites para memória ----------------
MAX_CANDLES = 300   # mantém só as últimas N velas na memória

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    # mensagens importantes via logger + print (flush) para compatibilidade com render logs
    if level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)
    print(msg, flush=True)

# ---------------- Telegram helper (bypass option) ----------------
def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False):
    """
    Envia mensagem para o chat.
    - Se 'symbol' informado, aplica throttle por símbolo (3s) a menos que bypass_throttle=True.
    - Usa parse_mode HTML.
    """
    now = time.time()

    if symbol and not bypass_throttle:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            log(f"[TG] throttle skip for {symbol}", "warning")
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("⚠️ Telegram não configurado.", "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
    except Exception as e:
        log(f"[TG] Erro ao enviar: {e}\n{traceback.format_exc()}", "error")

# ---------- Helper de notificação controlada ----------
def notify_once(symbol: str, key: str, message: str, bypass=False):
    """
    Notifica via Telegram apenas uma vez por chave (connected/history/ml/subscribed).
    Aceita symbol "global".
    """
    if symbol not in notify_flags:
        notify_flags[symbol] = {}
    flags = notify_flags.get(symbol, {})
    if flags.get(key):
        return
    try:
        send_telegram(html.escape(message), bypass_throttle=bypass)
    except Exception:
        log(f"[{symbol}] Falha ao notificar Telegram (notify_once).", "warning")
    flags[key] = True
    notify_flags[symbol] = flags

# ---------------- Utilitários específicos por símbolo ----------------
def ema_sep_scale_for_symbol(symbol: str) -> float:
    if "JPY" in symbol or any(x in symbol for x in ["USDNOK", "USDSEK", "USDJPY", "GBPJPY", "EURJPY"]):
        return 0.5
    return DEFAULT_EMA_SEP_SCALE

def human_pair(symbol: str) -> str:
    return symbol.replace("frx", "")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("epoch").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
        else:
            df[c] = 0.0
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0

    df[f"ema{EMA_FAST}"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df[f"ema{EMA_MID}"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df[f"ema{EMA_SLOW}"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], 14).rsi()
    try:
        macd = MACD(df["close"], 26, 12, 9)
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd_diff"] = pd.NA

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    try:
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=ATR_WINDOW)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = df["atr"] / df["close"].replace(0, 1e-12)
    except Exception:
        df["atr"] = pd.NA
        df["atr_pct"] = pd.NA

    df["body"] = (df["close"] - df["open"]).abs()
    df["body_pct"] = df["body"] / df["close"].replace(0, 1e-12)
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["upper_wick_pct"] = df["upper_wick"] / df["close"].replace(0, 1e-12)
    df["lower_wick_pct"] = df["lower_wick"] / df["close"].replace(0, 1e-12)

    df["rel_sep"] = (df[f"ema{EMA_MID}"] - df[f"ema{EMA_SLOW}"]).abs() / df["close"].replace(0, 1e-12)

    df["range"] = df["high"] - df["low"]
    df["range_pct"] = df["range"] / df["close"].replace(0, 1e-12)
    df["range_sma"] = df["range_pct"].rolling(CONSOL_WINDOW, min_periods=1).mean()

    return df

# ---------------- ML: treino e predição ----------------
def _build_ml_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    df2 = df.copy().reset_index(drop=True)
    features = [
        "open", "high", "low", "close", "volume",
        f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}",
        "rsi", "macd_diff",
        "bb_upper", "bb_lower", "bb_mavg", "bb_width",
        "atr", "atr_pct", "body", "body_pct", "upper_wick", "lower_wick", "rel_sep", "range_pct", "range_sma"
    ]
    for c in features:
        if c not in df2.columns:
            df2[c] = 0.0
    df2 = df2[features].fillna(0.0)

    y = (df2["close"].shift(-1) > df2["close"]).astype(int)
    X = df2.iloc[:-1].copy()
    y = y.iloc[:-1].copy()

    if len(X) > ML_MAX_SAMPLES:
        X = X.tail(ML_MAX_SAMPLES).reset_index(drop=True)
        y = y.tail(ML_MAX_SAMPLES).reset_index(drop=True)

    return X, y

def train_ml_for_symbol(df: pd.DataFrame, symbol: str):
    """
    Treina ML somente se necessário. Atualiza ml_models, ml_model_ready e ml_trained_samples.
    """
    if not ML_ENABLED:
        log(f"[ML {symbol}] scikit-learn não disponível — ML desabilitado.", "warning")
        ml_model_ready[symbol] = False
        return False

    try:
        X, y = _build_ml_dataset(df)
        samples = len(X)
        if samples < ML_MIN_TRAINED_SAMPLES or len(y.unique()) < 2:
            log(f"[ML {symbol}] Dados insuficientes para treinar ML (samples={samples}, classes={list(y.unique())}).", "info")
            ml_model_ready[symbol] = False
            return False

        last_trained = ml_trained_samples.get(symbol, 0)
        if last_trained > 0 and samples < last_trained + ML_RETRAIN_INTERVAL:
            log(f"[ML {symbol}] Já treinado recentemente (samples={last_trained}); pula retrain (samples atuais={samples}).", "info")
            ml_model_ready[symbol] = True
            return True

        model = RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH, random_state=42)
        model.fit(X, y)

        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True
        ml_trained_samples[symbol] = samples

        log(f"[ML {symbol}] Modelo treinado (samples={samples}).", "info")
        try:
            notify_once(symbol, "ml", f"🤖 ML treinado para {human_pair(symbol)} com {samples} amostras.", bypass=True)
        except Exception:
            log(f"[ML {symbol}] Falha ao notificar Telegram sobre treino.", "warning")

        gc.collect()
        return True

    except Exception as e:
        log(f"[ML {symbol}] Erro ao treinar ML: {e}\n{traceback.format_exc()}", "error")
        ml_model_ready[symbol] = False
        return False

def ml_predict_prob(symbol: str, last_row: pd.Series) -> float:
    try:
        if not ml_model_ready.get(symbol):
            return None
        model, cols = ml_models.get(symbol, (None, None))
        if model is None or cols is None:
            return None

        Xrow = []
        for c in cols:
            Xrow.append(float(last_row.get(c, 0.0) if pd.notna(last_row.get(c, None)) else 0.0))

        probs = model.predict_proba([Xrow])[0]
        prob_up = float(probs[1])
        return prob_up

    except Exception as e:
        log(f"[ML {symbol}] Erro em ml_predict_prob: {e}\n{traceback.format_exc()}", "error")
        return None

# ---------------- Helpers de fallback / contagem ----------------
def prune_sent_timestamps():
    cutoff = time.time() - FALLBACK_WINDOW_SEC
    while sent_timestamps and sent_timestamps[0] < cutoff:
        sent_timestamps.popleft()

def check_and_activate_fallback():
    prune_sent_timestamps()
    if len(sent_timestamps) < MIN_SIGNALS_PER_HOUR:
        global fallback_active_until
        fallback_active_until = time.time() + FALLBACK_DURATION_SECONDS
        log(f"[FALLBACK] Ativado fallback adaptativo por pouca atividade (sinais última hora={len(sent_timestamps)}).", "warning")

def is_fallback_active():
    return time.time() < fallback_active_until

# ---------------- Lógica principal de geração de sinal (Opção A) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < max(EMA_SLOW, ATR_WINDOW) + 5:
            log(f"[{symbol}] Dados insuficientes para gerar sinal.", "info")
            return None

        now = df.iloc[-1]
        prev = df.iloc[-2]

        epoch = int(now["epoch"])
        close = float(now["close"])
        open_p = float(now["open"])
        high = float(now["high"])
        low = float(now["low"])
        candle_id = epoch - (epoch % GRANULARITY_SECONDS)

        if last_signal_candle.get(symbol) == candle_id:
            log(f"[{symbol}] Já houve sinal nesse candle ({candle_id}), ignorando.", "debug")
            return None

        last_candle = last_signal_candle.get(symbol)
        if last_candle is not None:
            candles_passed = (candle_id - last_candle) // GRANULARITY_SECONDS
            if candles_passed < MIN_CANDLES_BETWEEN_SIGNALS:
                log(f"[{symbol}] Ignorando: só passaram {candles_passed} velas desde o último sinal (< {MIN_CANDLES_BETWEEN_SIGNALS}).", "info")
                return None

        for col in [f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}", "atr", "atr_pct", "range_sma", "body_pct", "upper_wick_pct", "lower_wick_pct", "rsi"]:
            if col not in now or pd.isna(now.get(col, None)):
                log(f"[{symbol}] Indicador {col} incompleto (NaN) — aguardando mais candles.", "warning")
                return None

        ema_fast = float(now[f"ema{EMA_FAST}"])
        ema_mid = float(now[f"ema{EMA_MID}"])
        ema_slow = float(now[f"ema{EMA_SLOW}"])

        rel_diff_fast_mid = abs(ema_fast - ema_mid) / max(1e-12, abs(close))
        rel_diff_mid_slow = abs(ema_mid - ema_slow) / max(1e-12, abs(close))

        triple_up = (ema_fast > ema_mid) and (ema_mid > ema_slow) and (ema_mid - ema_slow) / max(1e-12, abs(close)) >= EMA_ALIGNMENT_STRICTNESS
        triple_down = (ema_fast < ema_mid) and (ema_mid < ema_slow) and (ema_slow - ema_mid) / max(1e-12, abs(close)) >= EMA_ALIGNMENT_STRICTNESS

        if is_fallback_active():
            triple_up = (ema_fast >= ema_mid) and (ema_mid >= ema_slow)
            triple_down = (ema_fast <= ema_mid) and (ema_mid <= ema_slow)

        atr = float(now["atr"])
        atr_pct = float(now["atr_pct"]) if not pd.isna(now["atr_pct"]) else 0.0

        if atr_pct < ATR_PCT_MIN and not is_fallback_active():
            log(f"[{symbol}] ATR muito baixo (atr_pct={atr_pct:.6f} < {ATR_PCT_MIN}) — evita sinais em mercado fraco.", "info")
            return None
        if atr_pct > ATR_PCT_MAX:
            log(f"[{symbol}] ATR muito alto (atr_pct={atr_pct:.6f} > {ATR_PCT_MAX}) — mercado volátil demais, evita.", "warning")
            return None

        range_sma = float(now.get("range_sma", 0.0))
        if range_sma < CONSOL_RANGE_PCT and not is_fallback_active():
            log(f"[{symbol}] Zona de consolidação detectada (range_sma={range_sma:.6f} < {CONSOL_RANGE_PCT}) — evitando sinais.", "info")
            return None

        body = float(now["body"])
        body_pct = float(now["body_pct"])
        upper_wick_pct = float(now["upper_wick_pct"])
        lower_wick_pct = float(now["lower_wick_pct"])

        if body_pct < MIN_BODY_PCT and not is_fallback_active():
            log(f"[{symbol}] Candle fraco (body_pct={body_pct:.6f} < {MIN_BODY_PCT}) — ignora.", "debug")
            return None

        if body > 0:
            max_wick = max(upper_wick_pct, lower_wick_pct)
            if max_wick / (body_pct + 1e-12) > (1.0 / MAX_WICK_BODY_RATIO) and not is_fallback_active():
                log(f"[{symbol}] Candle com wick grande vs body (wick/body ratio alto) — ignora.", "debug")
                return None

        bb_upper = float(now["bb_upper"])
        bb_lower = float(now["bb_lower"])
        range_bb = bb_upper - bb_lower
        if range_bb <= 0 or math.isclose(range_bb, 0.0):
            log(f"[{symbol}] Bollinger range zero — ignorando.", "warning")
            return None
        lim_inf = bb_lower + range_bb * BB_PROXIMITY_PCT
        lim_sup = bb_upper - range_bb * BB_PROXIMITY_PCT
        perto_lower = close <= lim_inf
        perto_upper = close >= lim_sup

        candle_bullish = now["close"] > now["open"]
        candle_bearish = now["close"] < now["open"]

        rsi_now = float(now["rsi"]) if not pd.isna(now["rsi"]) else 50.0
        macd_diff = now.get("macd_diff")
        macd_buy_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff > -MACD_TOLERANCE)
        macd_sell_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff < MACD_TOLERANCE)
        buy_rsi_ok = rsi_now <= RSI_BUY_MAX
        sell_rsi_ok = rsi_now >= RSI_SELL_MIN

        ema_fast_prev = float(prev.get(f"ema{EMA_FAST}", ema_fast))
        ema_fast_mom = ema_fast - ema_fast_prev

        strong_momentum_buy = ema_fast_mom > (atr * 0.1)
        strong_momentum_sell = ema_fast_mom < -(atr * 0.1)

        cond_buy = (
            triple_up
            and (candle_bullish or strong_momentum_buy)
            and (perto_lower or strong_momentum_buy)
            and buy_rsi_ok
            and macd_buy_ok
        )

        cond_sell = (
            triple_down
            and (candle_bearish or strong_momentum_sell)
            and (perto_upper or strong_momentum_sell)
            and sell_rsi_ok
            and macd_sell_ok
        )

        if is_fallback_active():
            if not cond_buy and (ema_mid > ema_slow) and candle_bullish:
                cond_buy = True
            if not cond_sell and (ema_mid < ema_slow) and candle_bearish:
                cond_sell = True

        if not (cond_buy or cond_sell):
            log(f"[{symbol}] Condições buy/sell não satisfeitas (triple_up={triple_up}, triple_down={triple_down}, perto_lower={perto_lower}, perto_upper={perto_upper}, buy_rsi_ok={buy_rsi_ok}, sell_rsi_ok={sell_rsi_ok}).", "debug")
            return None

        last_state = last_signal_state.get(symbol)
        last_time = last_signal_time.get(symbol, 0)
        now_ts = time.time()
        if last_state is not None and last_state != ("COMPRA" if cond_buy else "VENDA"):
            if now_ts - last_time < MIN_SECONDS_BETWEEN_OPPOSITE:
                log(f"[{symbol}] Sinal oposto detectado mas dentro do cooldown oposto ({now_ts-last_time:.1f}s) — skip.", "warning")
                return None

        def calc_forca(is_buy: bool):
            score = 0.0
            score += min(25.0, rel_diff_mid_slow / (EMA_ALIGNMENT_STRICTNESS + 1e-12) * 25.0)
            score += min(20.0, (atr_pct / max(1e-12, ATR_PCT_MIN)) * 10.0)
            score += min(30.0, body_pct / (MIN_BODY_PCT + 1e-12) * 30.0)
            if is_buy:
                dist = max(0.0, min(1.0, (lim_inf - close) / range_bb))
                score += dist * 15.0
            else:
                dist = max(0.0, min(1.0, (close - lim_sup) / range_bb))
                score += dist * 15.0
            if macd_diff is not None and not pd.isna(macd_diff):
                macd_strength = max(0.0, min(1.0, abs(macd_diff) / (MACD_TOLERANCE * 5)))
                score += macd_strength * 10.0
            return int(max(0, min(100, round(score))))

        if cond_buy:
            force = calc_forca(is_buy=True)
            rel_thresh = FALLBACK_REL_EMA_SEP_PCT if is_fallback_active() else REL_EMA_SEP_PCT
            micro_force_thresh = FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD if is_fallback_active() else MICRO_FORCE_ALLOW_THRESHOLD
            force_min_effective = FALLBACK_FORCE_MIN if is_fallback_active() else FORCE_MIN

            if now["rel_sep"] < rel_thresh and force < micro_force_thresh:
                log(f"[{symbol}] Bloqueado por micro-ruído: rel_sep={now['rel_sep']:.3e} < {rel_thresh:.3e} e força={force} < {micro_force_thresh}.", "info")
                return None

            if force < force_min_effective:
                log(f"[{symbol}] Força {force}% abaixo do mínimo efetivo {force_min_effective}% — ignorando.", "debug")
                return None

            log(f"[{symbol}] SINAL GERADO (pré-pendência - Profissional): COMPRA (força={force}%, atr_pct={atr_pct:.5f}, body_pct={body_pct:.5f})", "info")
            return {"tipo": "COMPRA", "forca": force, "candle_id": candle_id, "rel_sep": now["rel_sep"]}

        if cond_sell:
            force = calc_forca(is_buy=False)
            rel_thresh = FALLBACK_REL_EMA_SEP_PCT if is_fallback_active() else REL_EMA_SEP_PCT
            micro_force_thresh = FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD if is_fallback_active() else MICRO_FORCE_ALLOW_THRESHOLD
            force_min_effective = FALLBACK_FORCE_MIN if is_fallback_active() else FORCE_MIN

            if now["rel_sep"] < rel_thresh and force < micro_force_thresh:
                log(f"[{symbol}] Bloqueado por micro-ruído: rel_sep={now['rel_sep']:.3e} < {rel_thresh:.3e} e força={force} < {micro_force_thresh}.", "info")
                return None

            if force < force_min_effective:
                log(f"[{symbol}] Força {force}% abaixo do mínimo efetivo {force_min_effective}% — ignorando.", "debug")
                return None

            log(f"[{symbol}] SINAL GERADO (pré-pendência - Profissional): VENDA (força={force}%, atr_pct={atr_pct:.5f}, body_pct={body_pct:.5f})", "info")
            return {"tipo": "VENDA", "forca": force, "candle_id": candle_id, "rel_sep": now["rel_sep"]}

        return None

    except Exception as e:
        log(f"[{symbol}] Erro em gerar_sinal: {e}\n{traceback.format_exc()}", "error")
        return None

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    try:
        df.tail(MAX_CANDLES).to_csv(path, index=False)
    except Exception as e:
        log(f"[{symbol}] Erro ao salvar candles: {e}", "warning")

# ---------------- Monitor WebSocket (com backoff, validação e ML) ----------------
async def monitor_symbol(symbol: str):
    reconnect_attempt = 0
    df = pd.DataFrame()
    csv_path = DATA_DIR / f"candles_{symbol}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = calcular_indicadores(df)
            if len(df) > MAX_CANDLES:
                df = df.tail(MAX_CANDLES).reset_index(drop=True)
            historical_loaded[symbol] = True
            log(f"[{symbol}] Histórico carregado do disco ({len(df)} candles).")
            notify_once(symbol, "history", f"📥 [{human_pair(symbol)}] Histórico carregado do disco ({len(df)} candles).", bypass=True)
        except Exception:
            log(f"[{symbol}] Falha ao carregar histórico do disco, continuará solicitando via WS.", "warning")

    while True:
        try:
            reconnect_attempt += 1
            log(f"[{symbol}] Conectando ao WS (attempt {reconnect_attempt})...")
            try:
                async with websockets.connect(WS_URL, ping_interval=30, ping_timeout=10) as ws:
                    log(f"[{symbol}] WS conectado.")
                    notify_once(symbol, "connected", f"🔌 [{human_pair(symbol)}] WebSocket conectado.", bypass=True)
                    reconnect_attempt = 0

                    # authorize
                    try:
                        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                        auth_raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    except asyncio.CancelledError:
                        log(f"[{symbol}] asyncio.CancelledError durante authorize — reconectar.", "warning")
                        raise
                    except Exception as e:
                        log(f"[{symbol}] Falha ao autorizar/receber authorize: {e}", "error")
                        raise

                    # HISTÓRICO INICIAL: tentativas controladas
                    if not historical_loaded.get(symbol, False):
                        history_tries = 0
                        while True:
                            history_tries += 1
                            try:
                                req_hist = {
                                    "ticks_history": symbol,
                                    "count": INITIAL_HISTORY_COUNT,
                                    "end": "latest",
                                    "granularity": GRANULARITY_SECONDS,
                                    "style": "candles"
                                }
                                await ws.send(json.dumps(req_hist))
                                raw = await asyncio.wait_for(ws.recv(), timeout=60)
                                data = json.loads(raw)

                                if isinstance(data, dict) and "history" in data and isinstance(data["history"], dict):
                                    history = data["history"]
                                    if isinstance(history.get("candles"), list) and len(history["candles"]) > 0:
                                        df = pd.DataFrame(history["candles"])
                                        break

                                if isinstance(data, dict) and "candles" in data and isinstance(data["candles"], list) and len(data["candles"]) > 0:
                                    df = pd.DataFrame(data["candles"])
                                    break

                                log(f"[{symbol}] Histórico inicial sem candles (tentativa {history_tries}), resposta keys: {list(data.keys()) if isinstance(data, dict) else type(data)}", "warning")
                                await asyncio.sleep(1.0 + random.random() * 1.5)

                            except asyncio.TimeoutError:
                                log(f"[{symbol}] Timeout ao solicitar histórico (tentativa {history_tries}).", "warning")
                                # não lançar direto — controla via contador
                                if history_tries >= HISTORY_MAX_TRIES:
                                    # registra falha, pausa e tenta novamente depois de HISTORY_RETRY_PAUSE_SEC
                                    history_fail_count[symbol] += 1
                                    log(f"[{symbol}] Erro no histórico inicial — servidor Deriv retornou TIMEOUT. Tentando novamente em {HISTORY_RETRY_PAUSE_SEC//60} minutos. (fail #{history_fail_count[symbol]})", "warning")
                                    await asyncio.sleep(HISTORY_RETRY_PAUSE_SEC)
                                    # reiniciar tentativas após pausa
                                    history_tries = 0
                                    continue
                                await asyncio.sleep(1.0 + random.random() * 2.0)
                            except asyncio.CancelledError:
                                log(f"[{symbol}] asyncio.CancelledError ao obter histórico — reconectar.", "warning")
                                raise
                            except Exception as e:
                                log(f"[{symbol}] Erro ao obter histórico: {e}", "error")
                                # se atingiu o máximo de tentativas, faz pausa longa e recomeça (sem matar task)
                                if history_tries >= HISTORY_MAX_TRIES:
                                    history_fail_count[symbol] += 1
                                    log(f"[{symbol}] Erro no histórico inicial — {str(e)}. Tentando novamente em {HISTORY_RETRY_PAUSE_SEC//60} minutos. (fail #{history_fail_count[symbol]})", "warning")
                                    # pausa 10 minutos antes de reiniciar tentativas (não spamma Telegram)
                                    await asyncio.sleep(HISTORY_RETRY_PAUSE_SEC)
                                    history_tries = 0
                                    continue
                                await asyncio.sleep(1.0 + random.random() * 2.0)

                        # ao sair do loop com df válido
                        df = calcular_indicadores(df)
                        if len(df) > MAX_CANDLES:
                            df = df.tail(MAX_CANDLES).reset_index(drop=True)
                        save_last_candles(df, symbol)
                        historical_loaded[symbol] = True
                        log(f"[{symbol}] Histórico inicial carregado ({len(df)} candles).")
                        notify_once(symbol, "history", f"📥 [{human_pair(symbol)}] Histórico inicial ({len(df)} candles) carregado.", bypass=True)

                    # initial train if possible
                    try:
                        await asyncio.get_event_loop().run_in_executor(None, train_ml_for_symbol, df, symbol)
                    except Exception as e:
                        log(f"[ML {symbol}] Erro ao treinar inicial: {e}", "error")

                    # subscribe: only once per process
                    if not live_subscribed.get(symbol, False):
                        sub_req = {
                            "ticks_history": symbol,
                            "style": "candles",
                            "granularity": GRANULARITY_SECONDS,
                            "end": "latest",
                            "subscribe": 1
                        }
                        try:
                            await ws.send(json.dumps(sub_req))
                            live_subscribed[symbol] = True
                            log(f"[{symbol}] Inscrito em candles ao vivo.")
                            notify_once(symbol, "subscribed", f"🔔 [{human_pair(symbol)}] Inscrito em candles ao vivo (M{CANDLE_INTERVAL}).", bypass=True)
                        except Exception as e:
                            log(f"[{symbol}] Falha ao enviar subscribe: {e}", "warning")

                    ultimo_candle_time = time.time()

                    # Main receive loop
                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=600)
                        except asyncio.TimeoutError:
                            if time.time() - ultimo_candle_time > 300:
                                log(f"[{symbol}] Nenhum candle por >5min, forçando reconexão.", "warning")
                                raise Exception("Timeout prolongado, reconectar")
                            else:
                                log(f"[{symbol}] Timeout curto aguardando mensagem, mantendo conexão...", "info")
                                continue
                        except asyncio.CancelledError:
                            log(f"[{symbol}] asyncio.CancelledError no recv — reconectar.", "warning")
                            raise

                        try:
                            msg = json.loads(raw)
                        except Exception:
                            logger.debug(f"[{symbol}] Mensagem não JSON recebida, ignorando.")
                            continue

                        candle = None
                        if isinstance(msg, dict):
                            if "candle" in msg and isinstance(msg["candle"], dict):
                                candle = msg["candle"]
                            elif "ohlc" in msg and isinstance(msg["ohlc"], dict):
                                candle = msg["ohlc"]
                            elif "history" in msg and isinstance(msg["history"], dict) and isinstance(msg["history"].get("candles"), list):
                                last = msg["history"]["candles"][-1]
                                candle = last
                            elif "candles" in msg and isinstance(msg["candles"], list) and len(msg["candles"]) > 0:
                                candle = msg["candles"][-1]

                        if candle is None:
                            if isinstance(msg, dict) and msg.get("msg_type"):
                                logger.debug(f"[{symbol}] msg_type recebida: {msg.get('msg_type')}")
                            continue

                        try:
                            epoch = int(candle.get("epoch"))
                            open_p = float(candle.get("open"))
                            high_p = float(candle.get("high"))
                            low_p = float(candle.get("low"))
                            close_p = float(candle.get("close"))
                            volume_p = float(candle.get("volume")) if candle.get("volume") is not None else 0.0
                        except Exception:
                            log(f"[{symbol}] Candle com campos inválidos, ignorando: {candle}", "warning")
                            continue

                        # Consider only closed candles aligned with granularity
                        if epoch % GRANULARITY_SECONDS != 0:
                            continue

                        candle_time_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
                        log(f"[{symbol}] Novo candle recebido: epoch={epoch} UTC | close={close_p}")
                        ultimo_candle_time = time.time()

                        last_epoch_in_df = int(df.iloc[-1]["epoch"]) if not df.empty else None
                        is_new_candle = df.empty or last_epoch_in_df != epoch
                        if is_new_candle:
                            # append and trim to MAX_CANDLES for memory safety
                            df.loc[len(df)] = {
                                "epoch": epoch,
                                "open": open_p,
                                "high": high_p,
                                "low": low_p,
                                "close": close_p,
                                "volume": volume_p
                            }

                            # trim in-place
                            if len(df) > MAX_CANDLES:
                                df = df.tail(MAX_CANDLES).reset_index(drop=True)

                            df = calcular_indicadores(df)
                            save_last_candles(df, symbol)

                            # incremental retrain: retrain if we gained ML_RETRAIN_INTERVAL new samples
                            try:
                                samples = len(df)
                                last_trained = ml_trained_samples.get(symbol, 0)
                                if ML_ENABLED and (samples >= ML_MIN_TRAINED_SAMPLES) and (samples >= last_trained + ML_RETRAIN_INTERVAL):
                                    # run in executor to avoid blocking
                                    await asyncio.get_event_loop().run_in_executor(None, train_ml_for_symbol, df, symbol)
                                    # try to free memory after heavy op
                                    gc.collect()
                            except Exception:
                                log(f"[ML {symbol}] Erro no retrain incremental.", "warning")

                            # Antes de gerar sinal, verificar fallback global
                            prune_sent_timestamps()
                            if len(sent_timestamps) < MIN_SIGNALS_PER_HOUR:
                                check_and_activate_fallback()

                            # --- 1) Primeiro: checar se existe um pending signal para este symbol aguardando a abertura desta vela.
                            pending = pending_signals.get(symbol)
                            if pending:
                                # Se o novo candle (atual) tem epoch > candle_id do pending, então é a vela seguinte:
                                pending_candle_id = pending["sinal"]["candle_id"]
                                if epoch > pending_candle_id:
                                    # avaliar ML e cooldown usando os dados já atualizados (usamos a nova vela como entrada)
                                    tipo = pending["sinal"]["tipo"]
                                    forca = pending["sinal"]["forca"]
                                    # entry price = abertura da vela atual
                                    entry_price = open_p
                                    # somente horário (Brasília) — sem data
                                    entrada_br = candle_time_utc.astimezone(timezone(timedelta(hours=-3)))
                                    entrada_time_only = entrada_br.strftime("%H:%M:%S")

                                    # ML filter (avaliado no momento da abertura da vela seguinte para maior precisão)
                                    ml_ok = True
                                    ml_prob = None

                                    # if ML not ready, try to train quickly (to avoid N/A)
                                    if ML_ENABLED and not ml_model_ready.get(symbol, False):
                                        try:
                                            log(f"[ML {symbol}] ML não pronto (pending) — tentando treinar rápido antes de avaliar prob.")
                                            await asyncio.get_event_loop().run_in_executor(None, train_ml_for_symbol, df, symbol)
                                        except Exception:
                                            log(f"[ML {symbol}] Falha ao tentar treinar rápido (pending).", "warning")

                                    if ML_ENABLED and ml_model_ready.get(symbol, False):
                                        try:
                                            last_row = df.iloc[-1]
                                            prob_up = ml_predict_prob(symbol, last_row)
                                            ml_prob = prob_up
                                            if prob_up is None:
                                                ml_ok = True
                                            else:
                                                if tipo == "COMPRA":
                                                    ml_ok = prob_up >= ML_CONF_THRESHOLD
                                                else:
                                                    ml_ok = (1.0 - prob_up) >= ML_CONF_THRESHOLD
                                        except Exception as e:
                                            log(f"[ML {symbol}] Erro ao avaliar ML (pending): {e}", "error")
                                            ml_ok = True  # fail-open

                                    if not ml_ok:
                                        # **IMPORTANTE**: NÃO enviar mensagem para Telegram sobre bloqueio.
                                        # Apenas logamos (render logs) para diagnóstico.
                                        log(f"[{symbol}] ❌ ML bloqueou o pending sinal {tipo} (prob_up={ml_prob}) — descartando (somente log).", "warning")
                                        # remove pending and continue (não marcar last_signal_time)
                                        del pending_signals[symbol]
                                    else:
                                        # cooldown global APÓS ML aprovar
                                        now_ts = time.time()
                                        if now_ts - last_signal_time.get(symbol, 0) < MIN_SECONDS_BETWEEN_SIGNALS:
                                            log(f"[{symbol}] Cooldown global ainda ativo (pending) ({now_ts-last_signal_time.get(symbol,0):.1f}s) — descarta pending.", "warning")
                                            del pending_signals[symbol]
                                        else:
                                            # enviar o sinal usando entry_price (abertura da vela atual)
                                            last_signal_time[symbol] = now_ts
                                            last_signal_candle[symbol] = pending_candle_id  # sinal originou da vela anterior
                                            last_signal_state[symbol] = tipo

                                            sent_timestamps.append(time.time())
                                            prune_sent_timestamps()

                                            # construir mensagem no modelo pedido (sem data, só horário)
                                            pair = html.escape(human_pair(symbol))
                                            direction = "COMPRA" if tipo == "COMPRA" else "VENDA"
                                            strength = forca
                                            price = f"{entry_price:.5f}"
                                            entry_time = entrada_time_only
                                            if ml_prob is not None:
                                                prob_pct = int(round(ml_prob * 100))
                                            else:
                                                prob_pct = "N/A"

                                            message = f"""
📊 NOVO SINAL — M{CANDLE_INTERVAL}
• Par: {pair}
• Direção: {direction}
• Força do sinal: {strength}%
• Preço: {price}
• Horário de entrada: {entry_time}
• ML prob: {prob_pct}%
"""
                                            try:
                                                send_telegram(message, symbol=symbol, bypass_throttle=False)
                                                log(f"[{symbol}] Pending: mensagem enviada ao Telegram (entry open).", "info")
                                            except Exception:
                                                log(f"[{symbol}] Falha ao enviar sinal pending ao Telegram.", "warning")

                                            # remove pending after send
                                            del pending_signals[symbol]

                            # --- 2) Em seguida: avaliar se geramos um novo sinal a partir da vela que acabou de fechar
                            novo_sinal = gerar_sinal(df, symbol)
                            if novo_sinal:
                                # armazena como pending — será enviado quando a próxima vela abrir
                                pending_signals[symbol] = {
                                    "sinal": novo_sinal,
                                    "created_at": time.time(),
                                    "max_age_candles": 2  # expira se não enviado em 2 velas
                                }
                                log(f"[{symbol}] Pending signal criado para candle_id={novo_sinal['candle_id']} (aguardando próxima vela para enviar).", "info")

                            # --- 3) limpeza de pendings antigos (expiração)
                            to_remove = []
                            for sym, p in list(pending_signals.items()):
                                age_candles = (epoch - p["sinal"]["candle_id"]) // GRANULARITY_SECONDS
                                if age_candles > p.get("max_age_candles", 2):
                                    to_remove.append(sym)
                            for sym in to_remove:
                                log(f"[{sym}] Pending sinal expirou (não enviado em tempo) — removendo.", "warning")
                                del pending_signals[sym]

            except websockets.InvalidStatusCode as e:
                log(f"[WS {symbol}] InvalidStatusCode ao conectar: {e}", "warning")
            except Exception as e:
                # qualquer exceção no contexto de conexão cai aqui
                log(f"[WS {symbol}] erro na sessão WS: {e}\n{traceback.format_exc()}", "error")

        except Exception as e:
            log(f"[{symbol}] erro no ciclo principal: {e}\n{traceback.format_exc()}", "error")

        # backoff antes de tentar reconectar — menos agressivo e com jitter
        reconnect_attempt = min(reconnect_attempt + 1, 10)
        base = [3, 8, 15, 30]
        idx = min(reconnect_attempt - 1, len(base) - 1)
        backoff = base[idx]
        jitter = random.uniform(0.8, 1.2)
        sleep_time = backoff * jitter
        log(f"[{symbol}] Reconectando em {sleep_time:.1f}s (attempt {reconnect_attempt})...", "info")
        # small sleep before reconnect
        await asyncio.sleep(sleep_time)
        # hint GC between reconnects
        gc.collect()

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Lógica A (Precisão Profissional) — Triple EMA + ATR + Price Action + ML leve (hist=500, ML samples up to 2000)"

# ---------------- Execução ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run("0.0.0.0", port, debug=False, use_reloader=False)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    notify_once("global", "started", "✅ Bot iniciado — Lógica A (Precisão Profissional) — Triple EMA + ATR + Price Action + ML leve (hist=500)", bypass=True)
    if not SKLEARN_AVAILABLE:
        log("⚠️ scikit-learn não encontrado. ML desabilitado. Instale scikit-learn para habilitar ML.", "warning")
    await asyncio.gather(*(monitor_symbol(s) for s in SYMBOLS))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
