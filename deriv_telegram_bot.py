# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (AJUSTADA) + ML LEVE (RandomForest)
# (Opção B — Filtro Moderado + ML leve + 3 velas entre sinais + trend momentum)
# Correções: não marcar candle antes de ML, cooldown aplicado APÓS ML,
# tendência menos restritiva, micro-ruído mais permissivo.
# ===============================================================

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
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback
import math

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

# Anti-spam / anti-duplicate (tempo)
MIN_SECONDS_BETWEEN_SIGNALS = 5           # cooldown entre envios (aplicado APÓS ML aprovar)
MIN_SECONDS_BETWEEN_OPPOSITE = 60         # evita compra->venda imediata (opposite cooldown)

# Anti-duplicate (velas): aguarda N velas entre sinais para o mesmo par
MIN_CANDLES_BETWEEN_SIGNALS = 3           # 3 velas = 15min se CANDLE_INTERVAL=5

# EMA separation relative threshold (fraction of price)
# Ajustado para ser mais permissivo (menos bloqueios por micro-ruído)
REL_EMA_SEP_PCT = 8e-06                   # menos restritivo que 2e-05/5e-05

# Soft rule: se rel_sep < REL_EMA_SEP_PCT, permita sinal apenas se força >= threshold
MICRO_FORCE_ALLOW_THRESHOLD = 40          # reduzido para permitir mais sinais

# Require minimum EMA separation scale baseline (kept for fallback)
DEFAULT_EMA_SEP_SCALE = 0.01

# ML params
ML_ENABLED = SKLEARN_AVAILABLE            # habilitado apenas se sklearn presente
ML_N_ESTIMATORS = 40
ML_MAX_DEPTH = 4
ML_MIN_TRAINED_SAMPLES = 50               # mínimo de amostras para considerar o modelo válido
ML_CONF_THRESHOLD = 0.55                  # probabilidade mínima para aceitar a predição do ML

# ---------------- Estado ----------------
last_signal_state = {s: None for s in SYMBOLS}        # "COMPRA"/"VENDA"
last_signal_candle = {s: None for s in SYMBOLS}      # candle_id (epoch aligned)
last_signal_time = {s: 0 for s in SYMBOLS}           # timestamp last signal (set WHEN SENT)
last_notify_time = {}                                 # throttle per-symbol for normal messages

# ML models per symbol
ml_models = {}               # symbol -> sklearn model
ml_model_ready = {}          # symbol -> bool (trained & valid)

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
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

# ---------------- Telegram helper (bypass option) ----------------
def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False):
    """
    Envia mensagem para o chat.
    - Se 'symbol' informado, aplica throttle por símbolo (3s) a menos que bypass_throttle=True.
    - Use bypass_throttle=True para avisos de conexão/histórico/ML.
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
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
    except Exception as e:
        log(f"[TG] Erro ao enviar: {e}\n{traceback.format_exc()}", "error")

# ---------------- Utilitários específicos por symbol ----------------
def ema_sep_scale_for_symbol(symbol: str) -> float:
    """
    Retorna uma escala heurística para normalizar separação EMA20-EMA50.
    Mantive para compatibilidade, mas agora usamos checagem relativa (ema_sep/price).
    """
    if "JPY" in symbol or any(x in symbol for x in ["USDNOK", "USDSEK", "USDJPY", "GBPJPY", "EURJPY"]):
        return 0.5
    return DEFAULT_EMA_SEP_SCALE

def human_pair(symbol: str) -> str:
    return symbol.replace("frx", "")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
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

    # additional helpers
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["rel_sep"] = (df["ema20"] - df["ema50"]).abs() / df["close"].replace(0, 1e-12)

    return df

# ---------------- ML: treino e predição ----------------
def _build_ml_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Constrói X,y para treinar o modelo:
      - X: features do candle t
      - y: direção do candle t+1 (1 se close_t+1 > close_t, else 0)
    Usa colunas de indicadores já calculadas.
    """
    df2 = df.copy().reset_index(drop=True)
    # require numeric columns present
    features = [
        "open", "high", "low", "close", "volume",
        "ema20", "ema50", "rsi", "macd_diff",
        "bb_upper", "bb_lower", "bb_mavg", "bb_width", "rel_sep"
    ]
    # fillna
    for c in features:
        if c not in df2.columns:
            df2[c] = 0.0
    df2 = df2[features].fillna(0.0)

    # label: next close > current close
    y = (df2["close"].shift(-1) > df2["close"]).astype(int)
    X = df2.iloc[:-1].copy()
    y = y.iloc[:-1].copy()

    return X, y

def train_ml_for_symbol(df: pd.DataFrame, symbol: str):
    """
    Treina (ou re-treina) um RandomForest leve para 'symbol'.
    Retorna True se o modelo for treinado com sucesso e tem amostras suficientes.
    """
    if not ML_ENABLED:
        log(f"[ML {symbol}] scikit-learn não disponível — ML desabilitado.", "warning")
        ml_model_ready[symbol] = False
        return False

    try:
        X, y = _build_ml_dataset(df)
        if len(X) < ML_MIN_TRAINED_SAMPLES or len(y.unique()) < 2:
            log(f"[ML {symbol}] Dados insuficientes para treinar ML (samples={len(X)}, classes={y.unique()}).", "info")
            ml_model_ready[symbol] = False
            return False

        model = RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH, random_state=42)
        model.fit(X, y)

        ml_models[symbol] = (model, X.columns.tolist())
        ml_model_ready[symbol] = True

        log(f"[ML {symbol}] Modelo treinado (samples={len(X)}).", "info")
        # notify once that ML is ready for this symbol
        try:
            send_telegram(f"🤖 ML treinado para {human_pair(symbol)} com {len(X)} amostras.", bypass_throttle=True)
        except Exception:
            log(f"[ML {symbol}] Falha ao notificar Telegram sobre treino.", "warning")
        return True

    except Exception as e:
        log(f"[ML {symbol}] Erro ao treinar ML: {e}\n{traceback.format_exc()}", "error")
        ml_model_ready[symbol] = False
        return False

def ml_predict_prob(symbol: str, last_row: pd.Series) -> float:
    """
    Retorna probabilidade prevista de 'alta' (float 0..1) para o próximo candle.
    Se modelo não pronto, retorna None.
    """
    try:
        if not ml_model_ready.get(symbol):
            return None
        model, cols = ml_models.get(symbol, (None, None))
        if model is None or cols is None:
            return None

        # build feature vector in same order
        Xrow = []
        for c in cols:
            # if col missing in last_row, use 0
            Xrow.append(float(last_row.get(c, 0.0) if pd.notna(last_row.get(c, None)) else 0.0))

        # model.predict_proba expects 2D
        probs = model.predict_proba([Xrow])[0]
        # probs: [prob_class0, prob_class1]
        prob_up = float(probs[1])
        return prob_up

    except Exception as e:
        log(f"[ML {symbol}] Erro em ml_predict_prob: {e}\n{traceback.format_exc()}", "error")
        return None

# ---------------- Lógica — CORRIGIDA + FORÇA DO SINAL + MELHORIA DE TENDÊNCIA ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """
    Retorna None ou dict {"tipo": "COMPRA"/"VENDA", "forca": int(0-100), "candle_id": ...}
    Regras principais implementadas:
      - 1 sinal por candle (last_signal_candle)
      - cooldown de N velas entre sinais do mesmo par (MIN_CANDLES_BETWEEN_SIGNALS)
      - cooldown global (MIN_SECONDS_BETWEEN_SIGNALS) aplicado SOMENTE APÓS ML aprovar
      - bloqueio de sinais opostos por MIN_SECONDS_BETWEEN_OPPOSITE (usa last_signal_time)
      - filtro antirruído moderado (soft): se rel_sep baixo, só permite se força >= MICRO_FORCE_ALLOW_THRESHOLD
      - tendência aceita OR (tendencia OR cruzamento) para ser menos restritivo
      - NÃO atualiza estado do sinal aqui (apenas retorna). Estado só é atualizado no monitor_symbol após ML + envio.
    """
    try:
        if len(df) < 3:
            log(f"[{symbol}] Dados insuficientes para gerar sinal.", "info")
            return None

        now = df.iloc[-1]
        prev = df.iloc[-2]

        epoch = int(now["epoch"])
        close = float(now["close"])
        candle_id = epoch - (epoch % GRANULARITY_SECONDS)

        # 1 sinal por candle
        if last_signal_candle.get(symbol) == candle_id:
            log(f"[{symbol}] Já houve sinal nesse candle ({candle_id}), ignorando.", "debug")
            return None

        # ------------------ nova checagem: gap de velas desde último sinal ------------------
        last_candle = last_signal_candle.get(symbol)
        if last_candle is not None:
            # compute number of candles passed since last_candle
            candles_passed = (candle_id - last_candle) // GRANULARITY_SECONDS
            if candles_passed < MIN_CANDLES_BETWEEN_SIGNALS:
                log(f"[{symbol}] Ignorando: só passaram {candles_passed} velas desde o último sinal (< {MIN_CANDLES_BETWEEN_SIGNALS}).", "info")
                return None

        # indicadores
        ema20_now, ema50_now = now["ema20"], now["ema50"]
        ema20_prev, ema50_prev = prev["ema20"], prev["ema50"]
        rsi_now = now["rsi"]
        bb_upper, bb_lower = now["bb_upper"], now["bb_lower"]
        macd_diff = now.get("macd_diff")

        if any(pd.isna([ema20_now, ema50_now, ema20_prev, ema50_prev, rsi_now, bb_upper, bb_lower])):
            log(f"[{symbol}] Indicadores incompletos (NaN) — aguardando mais candles.", "warning")
            return None

        # Tendência / cruzamentos (mais permissivo: aceita tendência OU cruzamento)
        cruzou_up = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
        cruzou_down = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)
        tendencia_up = ema20_now > ema50_now
        tendencia_down = ema20_now < ema50_now

        # EMA20 momentum (sinaliza que EMA20 está subindo/descendo)
        ema20_momentum = ema20_now - ema20_prev

        # Bollinger proximidade
        range_bb = bb_upper - bb_lower
        if range_bb == 0 or math.isclose(range_bb, 0.0):
            log(f"[{symbol}] Bollinger range zero — ignorando.", "warning")
            return None
        lim_inf = bb_lower + range_bb * BB_PROXIMITY_PCT
        lim_sup = bb_upper - range_bb * BB_PROXIMITY_PCT

        perto_lower = close <= lim_inf
        perto_upper = close >= lim_sup

        # RSI + MACD ok flags
        buy_rsi_ok = rsi_now <= RSI_BUY_MAX
        sell_rsi_ok = rsi_now >= RSI_SELL_MIN
        macd_buy_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff > -MACD_TOLERANCE)
        macd_sell_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff < MACD_TOLERANCE)

        # Evitar micro-ruído: calcular rel_sep (mas não bloquear imediatamente)
        ema_sep = abs(ema20_now - ema50_now)
        rel_sep = ema_sep / max(1e-12, abs(close))  # razão relativa

        # --------- reforço da assertividade: aceitar tendencia OR cruzamento ----------
        buy_trend_ok = tendencia_up or cruzou_up
        sell_trend_ok = tendencia_down or cruzou_down

        # Condições combinadas (agora usando buy_trend_ok / sell_trend_ok)
        cond_buy = buy_trend_ok and perto_lower and buy_rsi_ok and macd_buy_ok
        cond_sell = sell_trend_ok and perto_upper and sell_rsi_ok and macd_sell_ok

        if not (cond_buy or cond_sell):
            log(f"[{symbol}] Condições buy/sell não satisfeitas (buy_trend_ok={buy_trend_ok}, sell_trend_ok={sell_trend_ok}, perto_lower={perto_lower}, perto_upper={perto_upper}, buy_rsi_ok={buy_rsi_ok}, sell_rsi_ok={sell_rsi_ok}).", "debug")
            return None

        # Evitar sinais opostos muito próximos (usa last_signal_time, que é o momento do ÚLTIMO ENVIO)
        last_state = last_signal_state.get(symbol)
        last_time = last_signal_time.get(symbol, 0)
        now_ts = time.time()
        if last_state is not None and last_state != ("COMPRA" if cond_buy else "VENDA"):
            if now_ts - last_time < MIN_SECONDS_BETWEEN_OPPOSITE:
                log(f"[{symbol}] Sinal oposto detectado mas dentro do cooldown oposto ({now_ts-last_time:.1f}s) — skip.", "warning")
                return None

        # Calcular força combinada (0..100) — adaptativa
        def calc_forca(is_buy: bool):
            score = 0.0

            # Bollinger: proximidade até 40 pontos
            if is_buy:
                dist = max(0.0, min(1.0, (lim_inf - close) / range_bb))
                score += dist * 40.0
            else:
                dist = max(0.0, min(1.0, (close - lim_sup) / range_bb))
                score += dist * 40.0

            # RSI: até 25 pontos
            if is_buy:
                rsi_strength = max(0.0, min(1.0, (RSI_BUY_MAX - rsi_now) / 20.0))
                score += rsi_strength * 25.0
            else:
                rsi_strength = max(0.0, min(1.0, (rsi_now - RSI_SELL_MIN) / 20.0))
                score += rsi_strength * 25.0

            # EMA separation: até 25 pontos (normalizamos por preço)
            # evitar divisão por zero (REL_EMA_SEP_PCT pode ser muito pequeno)
            denom = max(REL_EMA_SEP_PCT * 10, 1e-12)
            sep_strength = max(0.0, min(1.0, rel_sep / denom))
            score += sep_strength * 25.0

            # MACD: até 10 pontos
            if macd_diff is not None and not pd.isna(macd_diff):
                macd_strength = max(0.0, min(1.0, abs(macd_diff) / (MACD_TOLERANCE * 5)))
                score += macd_strength * 10.0

            return int(max(0, min(100, round(score))))

        # Build result (IMPORTANT: NÃO atualiza last_signal_candle/state/time aqui)
        if cond_buy:
            force = calc_forca(is_buy=True)

            # Soft micro-ruído rule (moderada):
            if rel_sep < REL_EMA_SEP_PCT and force < MICRO_FORCE_ALLOW_THRESHOLD:
                log(f"[{symbol}] Bloqueado por micro-ruído moderado: rel_sep={rel_sep:.3e} < {REL_EMA_SEP_PCT:.3e} e força={force} < {MICRO_FORCE_ALLOW_THRESHOLD}.", "info")
                return None

            log(f"[{symbol}] SINAL GERADO (pré-ML): COMPRA (força={force}%, rel_sep={rel_sep:.3e})", "info")
            return {"tipo": "COMPRA", "forca": force, "candle_id": candle_id, "rel_sep": rel_sep}

        if cond_sell:
            force = calc_forca(is_buy=False)

            if rel_sep < REL_EMA_SEP_PCT and force < MICRO_FORCE_ALLOW_THRESHOLD:
                log(f"[{symbol}] Bloqueado por micro-ruído moderado: rel_sep={rel_sep:.3e} < {REL_EMA_SEP_PCT:.3e} e força={force} < {MICRO_FORCE_ALLOW_THRESHOLD}.", "info")
                return None

            log(f"[{symbol}] SINAL GERADO (pré-ML): VENDA (força={force}%, rel_sep={rel_sep:.3e})", "info")
            return {"tipo": "VENDA", "forca": force, "candle_id": candle_id, "rel_sep": rel_sep}

        return None

    except Exception as e:
        log(f"[{symbol}] Erro em gerar_sinal: {e}\n{traceback.format_exc()}", "error")
        return None

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    try:
        df.tail(200).to_csv(path, index=False)
    except Exception as e:
        log(f"[{symbol}] Erro ao salvar candles: {e}", "warning")

# ---------------- Monitor WebSocket (com backoff, validação e ML) ----------------
async def monitor_symbol(symbol: str):
    reconnect_attempt = 0
    while True:
        try:
            reconnect_attempt += 1
            log(f"[{symbol}] Conectando ao WS (attempt {reconnect_attempt})...")
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                log(f"[{symbol}] WS conectado.")
                # enviar notificação de conexão sem throttle (bypass_throttle=True)
                try:
                    send_telegram(f"🔌 [{human_pair(symbol)}] WebSocket conectado.", bypass_throttle=True)
                except Exception:
                    log(f"[{symbol}] Falha ao notificar Telegram sobre conexão.", "warning")

                reconnect_attempt = 0  # reset on success

                # authorize
                try:
                    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                    auth_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except Exception as e:
                    log(f"[{symbol}] Falha ao autorizar/receber authorize: {e}", "error")
                    raise

                # initial history request
                df = None
                history_tries = 0
                while True:
                    history_tries += 1
                    try:
                        req_hist = {
                            "ticks_history": symbol,
                            "count": 200,
                            "end": "latest",
                            "granularity": GRANULARITY_SECONDS,
                            "style": "candles"
                        }
                        await ws.send(json.dumps(req_hist))
                        raw = await asyncio.wait_for(ws.recv(), timeout=10)
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
                        if history_tries >= 3:
                            raise Exception("Falha ao obter histórico após múltiplas tentativas")
                        await asyncio.sleep(1.0 + random.random() * 2.0)
                    except Exception as e:
                        log(f"[{symbol}] Erro ao obter histórico: {e}", "error")
                        if history_tries >= 3:
                            raise
                        await asyncio.sleep(1.0 + random.random() * 2.0)

                df = calcular_indicadores(df)
                save_last_candles(df, symbol)
                log(f"[{symbol}] Histórico inicial carregado ({len(df)} candles).")
                # notificar histórico carregado (sem throttle)
                try:
                    send_telegram(f"📥 [{human_pair(symbol)}] Histórico inicial ({len(df)} candles) carregado.", bypass_throttle=True)
                except Exception:
                    log(f"[{symbol}] Falha ao notificar Telegram sobre histórico.", "warning")

                # Train ML now with the initial history (non-blocking enough; fast)
                try:
                    trained = train_ml_for_symbol(df, symbol)
                    if not trained and ML_ENABLED:
                        log(f"[ML {symbol}] Modelo NÃO treinado (dados insuficientes).", "info")
                except Exception as e:
                    log(f"[ML {symbol}] Erro ao treinar inicial: {e}", "error")

                # subscribe
                sub_req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": GRANULARITY_SECONDS,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))
                log(f"[{symbol}] Inscrito em candles ao vivo.")
                try:
                    send_telegram(f"🔔 [{human_pair(symbol)}] Inscrito em candles ao vivo (M{CANDLE_INTERVAL}).", bypass_throttle=True)
                except Exception:
                    log(f"[{symbol}] Falha ao notificar Telegram sobre inscrição.", "warning")

                ultimo_candle_time = time.time()

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    except asyncio.TimeoutError:
                        if time.time() - ultimo_candle_time > 300:
                            log(f"[{symbol}] Nenhum candle por >5min, forçando reconexão.", "warning")
                            raise Exception("Timeout prolongado, reconectar")
                        else:
                            log(f"[{symbol}] Timeout curto aguardando mensagem, mantendo conexão...", "info")
                            continue

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        log(f"[{symbol}] Mensagem não JSON recebida, ignorando.", "warning")
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
                            log(f"[{symbol}] msg_type recebida: {msg.get('msg_type')}", "info")
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
                    if df.empty or last_epoch_in_df != epoch:
                        df.loc[len(df)] = {
                            "epoch": epoch,
                            "open": open_p,
                            "high": high_p,
                            "low": low_p,
                            "close": close_p,
                            "volume": volume_p
                        }

                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        # incremental retrain if needed
                        try:
                            if len(df) >= 200 and (not ml_model_ready.get(symbol, False)):
                                train_ml_for_symbol(df, symbol)
                        except Exception:
                            log(f"[ML {symbol}] Erro no retrain incremental.", "warning")

                        sinal = gerar_sinal(df, symbol)

                        if sinal:
                            tipo = sinal["tipo"]
                            forca = sinal["forca"]
                            arrow = "🟢" if tipo == "COMPRA" else "🔴"
                            price = close_p
                            entrada_br = candle_time_utc.astimezone(timezone(timedelta(hours=-3)))
                            entrada_str = entrada_br.strftime("%Y-%m-%d %H:%M:%S")

                            # ---------- ML filter (se habilitado) ----------
                            ml_ok = True
                            ml_prob = None
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
                                    log(f"[ML {symbol}] Erro ao avaliar ML: {e}", "error")
                                    ml_ok = True  # fail-open: se der erro no ML, não bloquear

                            if not ml_ok:
                                # NÃO marcar last_signal_candle — liberar para tentar novamente
                                log(f"[{symbol}] ❌ ML bloqueou o sinal {tipo} (prob_up={ml_prob}). Sinal NÃO marcado — aguarda próxima vela.", "warning")
                                try:
                                    send_telegram(f"❌ [{human_pair(symbol)}] Sinal {tipo} bloqueado pelo ML (prob_up={ml_prob:.3f}).", bypass_throttle=True)
                                except Exception:
                                    log(f"[{symbol}] Falha ao notificar Telegram sobre bloqueio ML.", "warning")
                                continue  # não marcar; não atualizar last_signal_time

                            # ---------- cooldown global APÓS ML aprovar ----------
                            now_ts = time.time()
                            if now_ts - last_signal_time.get(symbol, 0) < MIN_SECONDS_BETWEEN_SIGNALS:
                                log(f"[{symbol}] Cooldown global ainda ativo ({now_ts-last_signal_time.get(symbol,0):.1f}s) — skip.", "warning")
                                continue

                            # ---------- tudo ok: marcar estado e enviar ----------
                            last_signal_time[symbol] = now_ts
                            last_signal_candle[symbol] = sinal["candle_id"]
                            last_signal_state[symbol] = tipo

                            msg_final = (
                                f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                f"• Par: {human_pair(symbol)}\n"
                                f"• Direção: {arrow} *{tipo}*\n"
                                f"• Força do sinal: *{forca}%*\n"
                                f"• Preço: {price:.5f}\n"
                                f"• Horário de entrada (Brasília): {entrada_str}"
                            )

                            if ml_prob is not None:
                                prob_pct = int(round(ml_prob * 100))
                                if tipo == "COMPRA":
                                    msg_final += f"\n• ML prob subida: *{prob_pct}%* (threshold {int(ML_CONF_THRESHOLD*100)}%)"
                                else:
                                    msg_final += f"\n• ML prob descida: *{100-prob_pct}%* (threshold {int(ML_CONF_THRESHOLD*100)}%)"

                            try:
                                send_telegram(msg_final, symbol=symbol, bypass_throttle=False)
                                log(f"[{symbol}] Mensagem enviada ao Telegram.", "info")
                            except Exception:
                                log(f"[{symbol}] Falha ao enviar sinal ao Telegram.", "warning")

        except websockets.exceptions.ConnectionClosed as e:
            log(f"[WS {symbol}] ConnectionClosed: {e}", "warning")
        except Exception as e:
            log(f"[WS {symbol}] erro: {e}\n{traceback.format_exc()}", "error")

        # backoff antes de tentar reconectar
        reconnect_attempt = min(reconnect_attempt + 1, 10)
        backoff = min(60, (2 ** (reconnect_attempt if reconnect_attempt > 0 else 1)) * 0.5)
        jitter = random.uniform(0.5, 1.5)
        sleep_time = backoff * jitter
        log(f"[{symbol}] Reconectando em {sleep_time:.1f}s (attempt {reconnect_attempt})...", "info")
        await asyncio.sleep(sleep_time)

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Lógica B (ajustada) — com força do sinal + ML leve (Filtro Moderado)"

# ---------------- Execução ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run("0.0.0.0", port, debug=False, use_reloader=False)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    # startup notification (bypass so you always get it)
    send_telegram("✅ Bot iniciado — Lógica B ajustada + Força do Sinal + ML leve (Filtro Moderado)", bypass_throttle=True)
    if not SKLEARN_AVAILABLE:
        log("⚠️ scikit-learn não encontrado. ML desabilitado. Instale scikit-learn para habilitar ML.", "warning")
    await asyncio.gather(*(monitor_symbol(s) for s in SYMBOLS))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
