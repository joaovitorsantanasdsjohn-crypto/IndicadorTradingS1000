===============================================================

deriv_telegram_bot.py — LÓGICA C (Opção C — Filtro Leve + ML atualizado por candle)

Ajustes aplicados:

1) Micro-ruído menos restritivo

2) Força mínima reduzida (FORCE_MIN default 50)

3) ML leve re-treinado incrementalmente por candle (retrain quando >= ML_MIN_TRAINED_SAMPLES)

4) Tendência moderada permitida (aceita tendência OU cruzamento)

5) Não marcar candle antes do ML; cooldown aplicado APÓS ML aprovar

6) Modo fallback adaptativo para garantir mínimo de sinais/hora

7) Precisão de entrada: sinal gerado no fechamento da vela, enviado na abertura da vela seguinte

===============================================================

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
from collections import deque
import html  # para escapar texto antes de enviar via HTML para Telegram

ML imports

try:
from sklearn.ensemble import RandomForestClassifier
SKLEARN_AVAILABLE = True
except Exception:
SKLEARN_AVAILABLE = False

---------------- Inicialização ----------------

load_dotenv()

---------------- Configurações principais ----------------

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

---------------- Parâmetros ----------------

BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

Anti-spam / anti-duplicate (tempo)

MIN_SECONDS_BETWEEN_SIGNALS = 5           # cooldown entre envios (aplicado APÓS ML aprovar)
MIN_SECONDS_BETWEEN_OPPOSITE = 60        # evita compra->venda imediata (opposite cooldown)

Anti-duplicate (velas): aguarda N velas entre sinais para o mesmo par

MIN_CANDLES_BETWEEN_SIGNALS = 3           # 3 velas = 15min se CANDLE_INTERVAL=5

EMA separation relative threshold (fraction of price)

REL_EMA_SEP_PCT = 8e-06                   # permissivo

Soft rule: se rel_sep < REL_EMA_SEP_PCT, permita sinal apenas se força >= threshold

MICRO_FORCE_ALLOW_THRESHOLD = 40

Força mínima absoluta para envio

FORCE_MIN = 50

ML params

ML_ENABLED = SKLEARN_AVAILABLE
ML_N_ESTIMATORS = 40
ML_MAX_DEPTH = 4
ML_MIN_TRAINED_SAMPLES = 300
ML_CONF_THRESHOLD = 0.55

Fallback adaptativo (garantir sinais/hora)

MIN_SIGNALS_PER_HOUR = 4
FALLBACK_WINDOW_SEC = 3600
FALLBACK_FORCE_MIN = 40
FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD = 25
FALLBACK_REL_EMA_SEP_PCT = 2e-05
FALLBACK_DURATION_SECONDS = 15 * 60

DEFAULT_EMA_SEP_SCALE = 0.01

---------------- Estado ----------------

last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}

ML models per symbol

ml_models = {}
ml_model_ready = {}

track sent timestamps globally to enforce MIN_SIGNALS_PER_HOUR fallback

sent_timestamps = deque()
fallback_active_until = 0.0

pending signals: quando gerar_sinal() detectar um sinal ele será guardado aqui

e enviado somente quando a próxima vela (candle) abrir — para melhorar precisão da entrada

estrutura: pending_signals[symbol] = {"sinal": sinal_dict, "created_at": ts, "max_age_candles": 2}

pending_signals = {}

---------------- Logging ----------------

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

---------------- Telegram helper (bypass option) ----------------

def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False):
"""
Envia mensagem para o chat.
- Se 'symbol' informado, aplica throttle por símbolo (3s) a menos que bypass_throttle=True.
- Usa parse_mode HTML (mais robusto) e escapa o texto.
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
    # escape message for HTML  
    safe_msg = message  # message should already be HTML-escaped where necessary  
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"  
    payload = {"chat_id": CHAT_ID, "text": safe_msg, "parse_mode": "HTML"}  
    r = requests.post(url, data=payload, timeout=10)  
    if r.status_code != 200:  
        log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")  
except Exception as e:  
    log(f"[TG] Erro ao enviar: {e}\n{traceback.format_exc()}", "error")

---------------- Utilitários específicos por symbol ----------------

def ema_sep_scale_for_symbol(symbol: str) -> float:
if "JPY" in symbol or any(x in symbol for x in ["USDNOK", "USDSEK", "USDJPY", "GBPJPY", "EURJPY"]):
return 0.5
return DEFAULT_EMA_SEP_SCALE

def human_pair(symbol: str) -> str:
return symbol.replace("frx", "")

---------------- Indicadores ----------------

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

df["bb_width"] = df["bb_upper"] - df["bb_lower"]  
df["rel_sep"] = (df["ema20"] - df["ema50"]).abs() / df["close"].replace(0, 1e-12)  

return df

---------------- ML: treino e predição ----------------

def _build_ml_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
df2 = df.copy().reset_index(drop=True)
features = [
"open", "high", "low", "close", "volume",
"ema20", "ema50", "rsi", "macd_diff",
"bb_upper", "bb_lower", "bb_mavg", "bb_width", "rel_sep"
]
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
if not ML_ENABLED:
log(f"[ML {symbol}] scikit-learn não disponível — ML desabilitado.", "warning")
ml_model_ready[symbol] = False
return False

try:  
    X, y = _build_ml_dataset(df)  
    if len(X) < ML_MIN_TRAINED_SAMPLES or len(y.unique()) < 2:  
        log(f"[ML {symbol}] Dados insuficientes para treinar ML (samples={len(X)}, classes={list(y.unique())}).", "info")  
        ml_model_ready[symbol] = False  
        return False  

    model = RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH, random_state=42)  
    model.fit(X, y)  

    ml_models[symbol] = (model, X.columns.tolist())  
    ml_model_ready[symbol] = True  

    log(f"[ML {symbol}] Modelo treinado (samples={len(X)}).", "info")  
    try:  
        send_telegram(html.escape(f"🤖 ML treinado para {human_pair(symbol)} com {len(X)} amostras."), bypass_throttle=True)  
    except Exception:  
        log(f"[ML {symbol}] Falha ao notificar Telegram sobre treino.", "warning")  
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

---------------- Helpers de fallback / contagem ----------------

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

---------------- Lógica principal de geração de sinal ----------------

def gerar_sinal(df: pd.DataFrame, symbol: str):
try:
if len(df) < 3:
log(f"[{symbol}] Dados insuficientes para gerar sinal.", "info")
return None

now = df.iloc[-1]  
    prev = df.iloc[-2]  

    epoch = int(now["epoch"])  
    close = float(now["close"])  
    candle_id = epoch - (epoch % GRANULARITY_SECONDS)  

    # 1 sinal por candle (não marcar aqui; só evitar gerar duas vezes no mesmo candle)  
    if last_signal_candle.get(symbol) == candle_id:  
        log(f"[{symbol}] Já houve sinal nesse candle ({candle_id}), ignorando.", "debug")  
        return None  

    # gap de velas desde último sinal  
    last_candle = last_signal_candle.get(symbol)  
    if last_candle is not None:  
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

    # Tendência / cruzamentos  
    cruzou_up = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)  
    cruzou_down = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)  
    tendencia_up = ema20_now > ema50_now  
    tendencia_down = ema20_now < ema50_now  

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

    # rel_sep  
    ema_sep = abs(ema20_now - ema50_now)  
    rel_sep = ema_sep / max(1e-12, abs(close))  

    buy_trend_ok = tendencia_up or cruzou_up  
    sell_trend_ok = tendencia_down or cruzou_down  

    cond_buy = buy_trend_ok and perto_lower and buy_rsi_ok and macd_buy_ok  
    cond_sell = sell_trend_ok and perto_upper and sell_rsi_ok and macd_sell_ok  

    if not (cond_buy or cond_sell):  
        log(f"[{symbol}] Condições buy/sell não satisfeitas (buy_trend_ok={buy_trend_ok}, sell_trend_ok={sell_trend_ok}, perto_lower={perto_lower}, perto_upper={perto_upper}, buy_rsi_ok={buy_rsi_ok}, sell_rsi_ok={sell_rsi_ok}).", "debug")  
        return None  

    # Evitar sinais opostos muito próximos (usa last_signal_time)  
    last_state = last_signal_state.get(symbol)  
    last_time = last_signal_time.get(symbol, 0)  
    now_ts = time.time()  
    if last_state is not None and last_state != ("COMPRA" if cond_buy else "VENDA"):  
        if now_ts - last_time < MIN_SECONDS_BETWEEN_OPPOSITE:  
            log(f"[{symbol}] Sinal oposto detectado mas dentro do cooldown oposto ({now_ts-last_time:.1f}s) — skip.", "warning")  
            return None  

    # Calcular força combinada (0..100)  
    def calc_forca(is_buy: bool):  
        score = 0.0  
        if is_buy:  
            dist = max(0.0, min(1.0, (lim_inf - close) / range_bb))  
            score += dist * 40.0  
        else:  
            dist = max(0.0, min(1.0, (close - lim_sup) / range_bb))  
            score += dist * 40.0  

        if is_buy:  
            rsi_strength = max(0.0, min(1.0, (RSI_BUY_MAX - rsi_now) / 20.0))  
            score += rsi_strength * 25.0  
        else:  
            rsi_strength = max(0.0, min(1.0, (rsi_now - RSI_SELL_MIN) / 20.0))  
            score += rsi_strength * 25.0  

        denom = max(REL_EMA_SEP_PCT * 10, 1e-12)  
        sep_strength = max(0.0, min(1.0, rel_sep / denom))  
        score += sep_strength * 25.0  

        if macd_diff is not None and not pd.isna(macd_diff):  
            macd_strength = max(0.0, min(1.0, abs(macd_diff) / (MACD_TOLERANCE * 5)))  
            score += macd_strength * 10.0  

        return int(max(0, min(100, round(score))))  

    # Build result (NÃO atualiza estado aqui)  
    if cond_buy:  
        force = calc_forca(is_buy=True)  
        rel_thresh = FALLBACK_REL_EMA_SEP_PCT if is_fallback_active() else REL_EMA_SEP_PCT  
        micro_force_thresh = FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD if is_fallback_active() else MICRO_FORCE_ALLOW_THRESHOLD  
        force_min_effective = FALLBACK_FORCE_MIN if is_fallback_active() else FORCE_MIN  

        if rel_sep < rel_thresh and force < micro_force_thresh:  
            log(f"[{symbol}] Bloqueado por micro-ruído moderado: rel_sep={rel_sep:.3e} < {rel_thresh:.3e} e força={force} < {micro_force_thresh}.", "info")  
            return None  

        if force < force_min_effective:  
            log(f"[{symbol}] Força {force}% abaixo do mínimo efetivo {force_min_effective}% — ignorando.", "debug")  
            return None  

        log(f"[{symbol}] SINAL GERADO (pré-pendência): COMPRA (força={force}%, rel_sep={rel_sep:.3e})", "info")  
        return {"tipo": "COMPRA", "forca": force, "candle_id": candle_id, "rel_sep": rel_sep}  

    if cond_sell:  
        force = calc_forca(is_buy=False)  
        rel_thresh = FALLBACK_REL_EMA_SEP_PCT if is_fallback_active() else REL_EMA_SEP_PCT  
        micro_force_thresh = FALLBACK_MICRO_FORCE_ALLOW_THRESHOLD if is_fallback_active() else MICRO_FORCE_ALLOW_THRESHOLD  
        force_min_effective = FALLBACK_FORCE_MIN if is_fallback_active() else FORCE_MIN  

        if rel_sep < rel_thresh and force < micro_force_thresh:  
            log(f"[{symbol}] Bloqueado por micro-ruído moderado: rel_sep={rel_sep:.3e} < {rel_thresh:.3e} e força={force} < {micro_force_thresh}.", "info")  
            return None  

        if force < force_min_effective:  
            log(f"[{symbol}] Força {force}% abaixo do mínimo efetivo {force_min_effective}% — ignorando.", "debug")  
            return None  

        log(f"[{symbol}] SINAL GERADO (pré-pendência): VENDA (força={force}%, rel_sep={rel_sep:.3e})", "info")  
        return {"tipo": "VENDA", "forca": force, "candle_id": candle_id, "rel_sep": rel_sep}  

    return None  

except Exception as e:  
    log(f"[{symbol}] Erro em gerar_sinal: {e}\n{traceback.format_exc()}", "error")  
    return None

---------------- Persistência ----------------

def save_last_candles(df: pd.DataFrame, symbol: str):
path = DATA_DIR / f"candles_{symbol}.csv"
try:
df.tail(200).to_csv(path, index=False)
except Exception as e:
log(f"[{symbol}] Erro ao salvar candles: {e}", "warning")

---------------- Monitor WebSocket (com backoff, validação e ML) ----------------

async def monitor_symbol(symbol: str):
reconnect_attempt = 0
while True:
try:
reconnect_attempt += 1
log(f"[{symbol}] Conectando ao WS (attempt {reconnect_attempt})...")
async with websockets.connect(WS_URL, ping_interval=None) as ws:
log(f"[{symbol}] WS conectado.")
try:
send_telegram(html.escape(f"🔌 [{human_pair(symbol)}] WebSocket conectado."), bypass_throttle=True)
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
                        "count": 500,  
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
            try:  
                send_telegram(html.escape(f"📥 [{human_pair(symbol)}] Histórico inicial ({len(df)} candles) carregado."), bypass_throttle=True)  
            except Exception:  
                log(f"[{symbol}] Falha ao notificar Telegram sobre histórico.", "warning")  

            # initial train if possible  
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
                send_telegram(html.escape(f"🔔 [{human_pair(symbol)}] Inscrito em candles ao vivo (M{CANDLE_INTERVAL})."), bypass_throttle=True)  
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
                is_new_candle = df.empty or last_epoch_in_df != epoch  
                if is_new_candle:  
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

                    # incremental retrain  
                    try:  
                        if ML_ENABLED:  
                            if len(df) >= ML_MIN_TRAINED_SAMPLES and (not ml_model_ready.get(symbol, False)):  
                                train_ml_for_symbol(df, symbol)  
                            elif len(df) >= ML_MIN_TRAINED_SAMPLES and len(df) % ML_MIN_TRAINED_SAMPLES == 0:  
                                train_ml_for_symbol(df, symbol)  
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
                            entrada_br = candle_time_utc.astimezone(timezone(timedelta(hours=-3)))  
                            entrada_str = entrada_br.strftime("%Y-%m-%d %H:%M:%S")  

                            # ML filter (avaliado no momento da abertura da vela seguinte para maior precisão)  
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
                                    log(f"[ML {symbol}] Erro ao avaliar ML (pending): {e}", "error")  
                                    ml_ok = True  # fail-open  

                            if not ml_ok:  
                                log(f"[{symbol}] ❌ ML bloqueou o pending sinal {tipo} (prob_up={ml_prob}). Pending descartado.", "warning")  
                                try:  
                                    send_telegram(html.escape(f"❌ [{human_pair(symbol)}] Sinal {tipo} bloqueado pelo ML (prob_up={ml_prob:.3f})."), bypass_throttle=True)  
                                except Exception:  
                                    log(f"[{symbol}] Falha ao notificar Telegram sobre bloqueio ML (pending).", "warning")  
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

                                    # construir mensagem HTML (escape)  
                                    pair = html.escape(human_pair(symbol))  
                                    tipo_str = "COMPRA" if tipo == "COMPRA" else "VENDA"  
                                    arrow = "🟢" if tipo == "COMPRA" else "🔴"  
                                    msg_final = (  
                                        f"📊 <b>NOVO SINAL — M{CANDLE_INTERVAL}</b>\n"  
                                        f"• Par: <b>{pair}</b>\n"  
                                        f"• Direção: {arrow} <b>{tipo_str}</b>\n"  
                                        f"• Força do sinal: <b>{forca}%</b>\n"  
                                        f"• Preço de entrada (open próxima vela): <b>{entry_price:.5f}</b>\n"  
                                        f"• Horário de entrada (Brasília): <b>{entrada_str}</b>"  
                                    )  

                                    if ml_prob is not None:  
                                        prob_pct = int(round(ml_prob * 100))  
                                        if tipo == "COMPRA":  
                                            msg_final += f"\n• ML prob subida: <b>{prob_pct}%</b> (threshold {int(ML_CONF_THRESHOLD*100)}%)"  
                                        else:  
                                            msg_final += f"\n• ML prob descida: <b>{100-prob_pct}%</b> (threshold {int(ML_CONF_THRESHOLD*100)}%)"  

                                    try:  
                                        send_telegram(msg_final, symbol=symbol, bypass_throttle=False)  
                                        log(f"[{symbol}] Pending: mensagem enviada ao Telegram (entry open).", "info")  
                                    except Exception:  
                                        log(f"[{symbol}] Falha ao enviar sinal pending ao Telegram.", "warning")  

                                    # remove pending after send  
                                    del pending_signals[symbol]  

                    # --- 2) Em seguida: avaliar se geramos um novo sinal a partir da vela que acabou de fechar  
                    # (o sinal ficará pendente até a próxima vela abrir)  
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
                    # remove pendings com mais de `max_age_candles` passadas  
                    to_remove = []  
                    for sym, p in pending_signals.items():  
                        age_candles = (epoch - p["sinal"]["candle_id"]) // GRANULARITY_SECONDS  
                        if age_candles > p.get("max_age_candles", 2):  
                            to_remove.append(sym)  
                    for sym in to_remove:  
                        log(f"[{sym}] Pending sinal expirou (não enviado em tempo) — removendo.", "warning")  
                        del pending_signals[sym]  

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

---------------- Flask ----------------

app = Flask(name)

@app.route("/")
def index():
return "Bot ativo — Lógica C (ajustada) — Força do Sinal + ML leve (retrain incremental) — envio na abertura da próxima vela"

---------------- Execução ----------------

def run_flask():
port = int(os.getenv("PORT", 10000))
app.run("0.0.0.0", port, debug=False, use_reloader=False)

async def main():
threading.Thread(target=run_flask, daemon=True).start()
send_telegram(html.escape("✅ Bot iniciado — Lógica C (Filtro Leve) + ML leve (retrain incremental) — envio na abertura da próxima vela"), bypass_throttle=True)
if not SKLEARN_AVAILABLE:
log("⚠️ scikit-learn não encontrado. ML desabilitado. Instale scikit-learn para habilitar ML.", "warning")
await asyncio.gather(*(monitor_symbol(s) for s in SYMBOLS))

if name == "main":
try:
asyncio.run(main())
except KeyboardInterrupt:
log("Encerrando...", "info")
