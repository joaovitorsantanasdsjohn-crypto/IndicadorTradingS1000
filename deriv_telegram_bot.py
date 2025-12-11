# Indicador Trading S1000 #




import asyncio 
import json 
import logging 
import os 
import random 
import time 
import traceback 
from collections 
import deque 
from datetime import datetime, timedelta, timezone 
from pathlib import Path

import pandas as pd 
import requests 
import websockets 
from dotenv import load_dotenv 
from flask import Flask

indicadores TA

from ta.momentum import RSIIndicator 
from ta.trend import EMAIndicator, MACD 
from ta.volatility import BollingerBands

ML opcional

try: from sklearn.ensemble import RandomForestClassifier 
    SKLEARN_AVAILABLE = True 
except Exception: SKLEARN_AVAILABLE = False

---------------- Inicialização ----------------

load_dotenv()

---------------- Configurações ----------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") CHAT_ID = os.getenv("CHAT_ID") DERIV_TOKEN = os.getenv("DERIV_TOKEN") CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5")) APP_ID = os.getenv("DERIV_APP_ID", "111022") WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}" GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SYMBOLS = [ "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD", "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP", "frxEURAUD", "frxAUDJPY", "frxGBPAUD", "frxGBPCAD", "frxAUDNZD", "frxEURCAD", ]

DATA_DIR = Path("./candles_data") DATA_DIR.mkdir(exist_ok=True)

---------------- Parâmetros ----------------

BB_PROXIMITY_PCT = 0.20 RSI_BUY_MAX = 52 RSI_SELL_MIN = 48 MACD_TOLERANCE = 0.002 MIN_CANDLES_BETWEEN_SIGNALS = int(os.getenv("MIN_CANDLES_BETWEEN_SIGNALS", "4")) EMA_FAST = 9 EMA_MID = 20 EMA_SLOW = 200

ML_ENABLED = SKLEARN_AVAILABLE ML_N_ESTIMATORS = 40 ML_MAX_DEPTH = 4 ML_MIN_TRAINED_SAMPLES = 200 ML_MAX_SAMPLES = 2000 ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55")) ML_RETRAIN_INTERVAL = 50

MIN_SIGNALS_PER_HOUR = 5 FALLBACK_WINDOW_SEC = 3600 FALLBACK_DURATION_SECONDS = 15 * 60 INITIAL_HISTORY_COUNT = int(os.getenv("INITIAL_HISTORY_COUNT", "500")) MAX_CANDLES = 300

---------------- Estado ----------------

last_signal_candle = {s: None for s in SYMBOLS} last_signal_time = {s: 0 for s in SYMBOLS} last_notify_time = {} ml_models = {} ml_model_ready = {} sent_timestamps = deque() fallback_active_until = 0.0 ml_trained_samples = {s: 0 for s in SYMBOLS}

---------------- Logging ----------------

logger = logging.getLogger("indicador") logger.setLevel(logging.INFO) handler = logging.StreamHandler() formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S") handler.setFormatter(formatter) logger.addHandler(handler)

def log(msg: str, level: str = "info"): if level == "info": logger.info(msg) elif level == "warning": logger.warning(msg) elif level == "error": logger.error(msg) else: logger.debug(msg)

---------------- Telegram ----------------

def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False): """Envia mensagem para o Telegram (API Bot).""" now_ts = time.time() if symbol and not bypass_throttle: last = last_notify_time.get(symbol, 0) if now_ts - last < 3: log(f"[TG] throttle skip for {symbol}", "warning") return last_notify_time[symbol] = now_ts

if not TELEGRAM_TOKEN or not CHAT_ID:
    log("⚠️ Telegram não configurado (TELEGRAM_TOKEN/CHAT_ID faltando).", "warning")
    return

try:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=10)
    if r.status_code != 200:
        log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
except Exception as e:
    log(f"[TG] Erro ao enviar: {e}", "error")

---------------- Utilitários ----------------

def human_pair(symbol: str) -> str: return symbol.replace("frx", "")

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame: if df is None or df.empty: return pd.DataFrame() df = df.sort_values("epoch").reset_index(drop=True) for c in ["open", "high", "low", "close", "volume"]: df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)

# EMAs
df[f"ema{EMA_FAST}"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
df[f"ema{EMA_MID}"] = EMAIndicator(df["close"], EMA_MID).ema_indicator()
df[f"ema{EMA_SLOW}"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()

# RSI
df["rsi"] = RSIIndicator(df["close"], 14).rsi().fillna(50.0)

# MACD diff
try:
    macd = MACD(df["close"], 26, 12, 9)
    df["macd_diff"] = macd.macd_diff().fillna(0.0)
except Exception:
    df["macd_diff"] = 0.0

# Bollinger
bb = BollingerBands(df["close"], window=20, window_dev=2)
df["bb_upper"] = bb.bollinger_hband().fillna(df["close"])
df["bb_lower"] = bb.bollinger_lband().fillna(df["close"])
df["bb_mavg"] = bb.bollinger_mavg().fillna(df["close"])
df["bb_width"] = df["bb_upper"] - df["bb_lower"]

# relative separation
df["rel_sep"] = (df[f"ema{EMA_MID}"] - df[f"ema{EMA_SLOW}"]).abs() / df["close"].replace(0, 1e-12)
return df

---------------- ML ----------------

def _build_ml_dataset(df: pd.DataFrame): df2 = df.copy().reset_index(drop=True) features = [ "open", "high", "low", "close", "volume", f"ema{EMA_FAST}", f"ema{EMA_MID}", f"ema{EMA_SLOW}", "rsi", "macd_diff", "bb_upper", "bb_lower", "bb_mavg", "bb_width", "rel_sep", ] for c in features: df2[c] = df2.get(c, 0.0)

y = (df2["close"].shift(-1) > df2["close"]).astype(int)
X = df2.iloc[:-1].copy()
y = y.iloc[:-1].copy()

if len(X) > ML_MAX_SAMPLES:
    X = X.tail(ML_MAX_SAMPLES).reset_index(drop=True)
    y = y.tail(ML_MAX_SAMPLES).reset_index(drop=True)

return X, y

def train_ml_for_symbol(df: pd.DataFrame, symbol: str): if not ML_ENABLED: ml_model_ready[symbol] = False return False try: X, y = _build_ml_dataset(df) if len(X) < ML_MIN_TRAINED_SAMPLES or len(y.unique()) < 2: ml_model_ready[symbol] = False log(f"[ML {symbol}] Dados insuficientes para treinar ({len(X)} samples).", "info") return False model = RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH, random_state=42) model.fit(X, y) ml_models[symbol] = (model, X.columns.tolist()) ml_model_ready[symbol] = True log(f"[ML {symbol}] Modelo treinado ({len(X)} samples).", "info") return True except Exception as e: ml_model_ready[symbol] = False log(f"[ML {symbol}] Erro ao treinar ML: {e}", "error") return False

def ml_predict_prob(symbol: str, last_row: pd.Series) -> float | None: try: if not ml_model_ready.get(symbol): return None model, cols = ml_models.get(symbol, (None, None)) if model is None or cols is None: return None Xrow = [float(last_row.get(c, 0.0)) for c in cols] return float(model.predict_proba([Xrow])[0][1]) except Exception as e: log(f"[ML {symbol}] Erro ao prever prob: {e}", "warning") return None

---------------- Fallback ----------------

def prune_sent_timestamps(): cutoff = time.time() - FALLBACK_WINDOW_SEC while sent_timestamps and sent_timestamps[0] < cutoff: sent_timestamps.popleft()

def check_and_activate_fallback(): global fallback_active_until prune_sent_timestamps() if len(sent_timestamps) < MIN_SIGNALS_PER_HOUR: fallback_active_until = time.time() + FALLBACK_DURATION_SECONDS log("⚠️ Fallback ativado (poucos sinais recentes).", "warning")

def is_fallback_active() -> bool: return time.time() < fallback_active_until

---------------- Persistência ----------------

def save_last_candles(df: pd.DataFrame, symbol: str): try: df.tail(MAX_CANDLES).to_csv(DATA_DIR / f"candles_{symbol}.csv", index=False) except Exception as e: log(f"[{symbol}] Erro ao salvar candles: {e}", "error")

---------------- Mensagens ----------------

def convert_utc_to_brasilia(dt_utc: datetime) -> str: brasilia = dt_utc - timedelta(hours=3) return brasilia.strftime("%H:%M:%S") + " BRT"

def format_signal_message(symbol: str, tipo: str, entry_dt_utc: datetime, ml_prob: float | None) -> str: pair = human_pair(symbol) direction_emoji = "🟢" if tipo == "COMPRA" else "🔴" direction_label = "COMPRA" if tipo == "COMPRA" else "VENDA" entry_brasilia = convert_utc_to_brasilia(entry_dt_utc) ml_text = "N/A" if ml_prob is None else f"{int(round(ml_prob * 100))}%" text = ( f"💱 <b>{pair}</b>\n\n" f"📈 DIREÇÃO: <b>{direction_emoji} {direction_label}</b>\n" f"⏱ ENTRADA: <b>{entry_brasilia}</b>\n\n" f"🤖 ML: <b>{ml_text}</b>" ) return text

def format_start_message() -> str: return ( "🟢 <b>BOT INICIADO!</b>\n\n" "O sistema está ativo e monitorando os pares configurados.\n" "Os horários enviados serão ajustados automaticamente para <b>Horário de Brasília</b>.\n" "Entradas serão disparadas para a <b>abertura da próxima vela</b> (M5)." )

---------------- Geração de sinais ----------------

def gerar_sinal(df: pd.DataFrame, symbol: str): try: if len(df) < EMA_SLOW + 5: log(f"[{symbol}] Dados insuficientes ({len(df)} candles).", "info") return None

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

    log(
        f"[{symbol}] EMA{EMA_FAST}={ema_fast:.6f} EMA{EMA_MID}={ema_mid:.6f} EMA{EMA_SLOW}={ema_slow:.6f} RSI={rsi_now:.2f} MACD_diff={macd_diff}",
        "info",
    )

    if not (buy_ok or sell_ok):
        log(f"[{symbol}] Sem condição para sinal.", "info")
        return None

    tipo = "COMPRA" if buy_ok else "VENDA"
    return {"tipo": tipo, "candle_id": candle_id}
except Exception as e:
    log(f"[{symbol}] Erro gerar_sinal: {e}", "error")
    return None

---------------- Monitor WebSocket (robusto) ----------------

async def monitor_symbol(symbol: str): """ Monitor individual symbol com reconexão/backoff e logs. Envia subscribe:1 junto ao pedido de ticks_history para receber atualizações ao vivo. """ columns = ["epoch", "open", "high", "low", "close", "volume"] df = pd.DataFrame(columns=columns)

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
        log(f"{symbol} | ERRO ao ler CSV histórico: {e}", "error")

connect_attempt = 0
backoff_base = 2.0

while True:
    try:
        connect_attempt += 1
        if connect_attempt > 1:
            delay = min(120, backoff_base ** min(connect_attempt, 6)) + random.random()
            log(
                f"{symbol} | Aguardando {delay:.1f}s antes de nova tentativa (attempt {connect_attempt}).",
                "info",
            )
            await asyncio.sleep(delay)

        log(f"{symbol} | Conectando ao WS (attempt {connect_attempt})...", "info")
        async with websockets.connect(WS_URL, ping_interval=40, ping_timeout=20, max_size=None) as ws:
            log(f"{symbol} | WS conectado.", "info")

            # autoriza
            await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
            try:
                auth_raw = await asyncio.wait_for(ws.recv(), timeout=30)
                try:
                    auth_msg = json.loads(auth_raw)
                    log(f"{symbol} | Autorize response: {auth_msg.get('msg_type', 'NA')}", "info")
                except Exception:
                    log(f"{symbol} | Autorize raw recebido (não JSON).", "info")
            except asyncio.TimeoutError:
                log(f"{symbol} | Timeout aguardando authorize response.", "warning")

            # solicita histórico (candles) + subscribe para updates ao vivo
            subscribe_msg = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": INITIAL_HISTORY_COUNT,
                "end": "latest",
                "style": "candles",
                "granularity": GRANULARITY_SECONDS,
                "subscribe": 1,
            }
            await ws.send(json.dumps(subscribe_msg))
            log(
                f"{symbol} | Histórico solicitado ({INITIAL_HISTORY_COUNT} candles) + subscribe para updates ao vivo.",
                "info",
            )

            connect_attempt = 0

            last_msg_ts = time.time()

            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=900)
                    last_msg_ts = time.time()
                except asyncio.TimeoutError:
                    raise Exception("Timeout prolongado, reconectar")

                try:
                    msg = json.loads(raw)
                except Exception:
                    log(f"{symbol} | Mensagem WS inválida (não JSON).", "warning")
                    continue

                # extrai candle de várias estruturas possíveis
                candle = None
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

                                # treinar ML ao carregar histórico
                                try:
                                    if ML_ENABLED:
                                        loop = asyncio.get_event_loop()
                                        await loop.run_in_executor(None, train_ml_for_symbol, df.copy(), symbol)
                                        ml_trained_samples[symbol] = len(df)
                                except Exception as e:
                                    log(f"{symbol} | Erro retrain ML inicial: {e}", "warning")
                    except Exception as e:
                        log(f"{symbol} | Erro processando histórico WS: {e}", "warning")
                    continue

                if "candle" in msg and isinstance(msg.get("candle"), dict):
                    candle = msg["candle"]
                elif "ohlc" in msg and isinstance(msg.get("ohlc"), dict):
                    candle = msg["ohlc"]
                elif "candles" in msg and isinstance(msg.get("candles"), list) and msg["candles"]:
                    candle = msg["candles"][-1]
                elif "tick" in msg and isinstance(msg.get("tick"), dict):
                    # ignorar ticks brutos
                    continue

                if not candle:
                    continue

                try:
                    epoch = int(candle.get("epoch"))
                    # aceitar apenas candles fechadas alinhadas à granularidade
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
                if df.empty:
                    df = pd.DataFrame([new_row], columns=columns)
                elif set(new_row.keys()) <= set(df.columns):
                    try:
                        df.loc[len(df)] = new_row
                    except Exception:
                        df = pd.concat([df, pd.DataFrame([new_row], columns=columns)], ignore_index=True)
                else:
                    log(f"{symbol} | Linha do candle inválida: {new_row}", "warning")

                if len(df) > MAX_CANDLES:
                    df = df.tail(MAX_CANDLES).reset_index(drop=True)

                # calcula indicadores e salva
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)

                log(f"🕯 {symbol} | Vela fechada recebida: epoch={epoch} O={open_p} H={high_p} L={low_p} C={close_p}", "info")

                # ML incremental (retrain em background se necessário)
                try:
                    samples = len(df)
                    last_trained = ml_trained_samples.get(symbol, 0)
                    if ML_ENABLED and samples >= ML_MIN_TRAINED_SAMPLES and samples >= last_trained + ML_RETRAIN_INTERVAL:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, train_ml_for_symbol, df.copy(), symbol)
                        ml_trained_samples[symbol] = samples
                except Exception as e:
                    log(f"{symbol} | Erro retrain ML: {e}", "warning")

                # geração de sinal
                sinal = gerar_sinal(df, symbol)
                if sinal:
                    ml_prob = None
                    if ML_ENABLED and ml_model_ready.get(symbol):
                        ml_prob = ml_predict_prob(symbol, df.iloc[-1])
                        log(f"[ML {symbol}] Prob_up={ml_prob}", "info")

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
        log(f"{symbol} | ERRO WS: {e} | reconectando em 3s", "error")
        await asyncio.sleep(3 + random.random() * 2)

---------------- LOOP PRINCIPAL ----------------

async def main(): start_msg = format_start_message() send_telegram(start_msg, bypass_throttle=True) log("Iniciando tasks para todos os símbolos...", "info") tasks = [monitor_symbol(s) for s in SYMBOLS] await asyncio.gather(*tasks)

---------------- Flask ----------------

app = Flask(name)

@app.get("/") def home(): return "BOT ONLINE", 200

def run_flask(): port = int(os.getenv("PORT", 10000)) log(f"🔎 Flask HTTP health-check iniciado na porta {port}", "info") app.run(host="0.0.0.0", port=port)

---------------- STARTUP ----------------

if name == "main": flask_thread = threading.Thread(target=run_flask, daemon=True) flask_thread.start() try: asyncio.run(main()) except KeyboardInterrupt: log("Encerrando por KeyboardInterrupt.", "info") except Exception as e: log(f"Erro fatal no loop principal: {e}", "error")
