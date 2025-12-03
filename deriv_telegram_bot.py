# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (AJUSTADA) — COMPLETO
# (com: anti-duplicação reforçada, timezone-aware (Brasília),
#  backoff/reconexão, validação robusta do histórico, Força do Sinal,
#  notificação Telegram quando cada WebSocket conectar — sem throttle)
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
    "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Parâmetros ----------------
BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

MIN_SECONDS_BETWEEN_SIGNALS = 10
MIN_SECONDS_BETWEEN_OPPOSITE = 60
DEFAULT_EMA_SEP_SCALE = 0.01

# ---------------- Estado ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}

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

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None, bypass_throttle: bool = False):
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

# ---------------- Utilitários ----------------
def ema_sep_scale_for_symbol(symbol: str) -> float:
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
    return df

# ---------------- Lógica do sinal ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < 3: return None
        now = df.iloc[-1]
        prev = df.iloc[-2]
        epoch = int(now["epoch"])
        close = float(now["close"])
        candle_id = epoch - (epoch % GRANULARITY_SECONDS)
        if last_signal_candle.get(symbol) == candle_id: return None

        ema20_now, ema50_now = now["ema20"], now["ema50"]
        ema20_prev, ema50_prev = prev["ema20"], prev["ema50"]
        rsi_now = now["rsi"]
        bb_upper, bb_lower = now["bb_upper"], now["bb_lower"]
        macd_diff = now.get("macd_diff")
        if any(pd.isna([ema20_now, ema50_now, ema20_prev, ema50_prev, rsi_now, bb_upper, bb_lower])):
            return None

        cruzou_up = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
        cruzou_down = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)
        tendencia_up = ema20_now > ema50_now
        tendencia_down = ema20_now < ema50_now

        range_bb = bb_upper - bb_lower
        if range_bb == 0 or math.isclose(range_bb, 0.0): return None
        lim_inf = bb_lower + range_bb * BB_PROXIMITY_PCT
        lim_sup = bb_upper - range_bb * BB_PROXIMITY_PCT
        perto_lower = close <= lim_inf
        perto_upper = close >= lim_sup

        buy_rsi_ok = rsi_now <= RSI_BUY_MAX
        sell_rsi_ok = rsi_now >= RSI_SELL_MIN
        macd_buy_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff > -MACD_TOLERANCE)
        macd_sell_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff < MACD_TOLERANCE)

        # Micro-ruído EMA
        ema_sep = abs(ema20_now - ema50_now)
        sep_scale = ema_sep_scale_for_symbol(symbol)
        required_sep = sep_scale if sep_scale >= 0.01 else 0.001
        if ema_sep < required_sep * 0.1:
            return None

        cond_buy = (cruzou_up or tendencia_up) and perto_lower and buy_rsi_ok and macd_buy_ok
        cond_sell = (cruzou_down or tendencia_down) and perto_upper and sell_rsi_ok and macd_sell_ok
        if not (cond_buy or cond_sell): return None

        last_state = last_signal_state.get(symbol)
        last_time = last_signal_time.get(symbol, 0)
        now_ts = time.time()
        if last_state is not None and last_state != ("COMPRA" if cond_buy else "VENDA"):
            if now_ts - last_time < MIN_SECONDS_BETWEEN_OPPOSITE:
                return None

        def calc_forca(is_buy: bool):
            score = 0.0
            if is_buy:
                dist = max(0.0, min(1.0, (lim_inf - close) / range_bb))
                score += dist * 40.0
                rsi_strength = max(0.0, min(1.0, (RSI_BUY_MAX - rsi_now) / 20.0))
            else:
                dist = max(0.0, min(1.0, (close - lim_sup) / range_bb))
                score += dist * 40.0
                rsi_strength = max(0.0, min(1.0, (rsi_now - RSI_SELL_MIN) / 20.0))
            score += rsi_strength * 25.0
            scale = ema_sep_scale_for_symbol(symbol)
            if scale <= 0: scale = DEFAULT_EMA_SEP_SCALE
            sep_strength = max(0.0, min(1.0, ema_sep / (scale * 1.0)))
            score += sep_strength * 25.0
            if macd_diff is not None and not pd.isna(macd_diff):
                macd_strength = max(0.0, min(1.0, abs(macd_diff) / (MACD_TOLERANCE * 5)))
                score += macd_strength * 10.0
            return int(max(0, min(100, round(score))))

        if cond_buy:
            force = calc_forca(True)
            last_signal_candle[symbol] = candle_id
            last_signal_time[symbol] = time.time()
            last_signal_state[symbol] = "COMPRA"
            return {"tipo": "COMPRA", "forca": force, "candle_id": candle_id}
        if cond_sell:
            force = calc_forca(False)
            last_signal_candle[symbol] = candle_id
            last_signal_time[symbol] = time.time()
            last_signal_state[symbol] = "VENDA"
            return {"tipo": "VENDA", "forca": force, "candle_id": candle_id}

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

# ---------------- WebSocket ----------------
async def monitor_symbol(symbol: str):
    reconnect_attempt = 0
    while True:
        try:
            reconnect_attempt += 1
            log(f"[{symbol}] Conectando ao WS (attempt {reconnect_attempt})...")
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                log(f"[{symbol}] WS conectado.")
                send_telegram(f"🔌 [{human_pair(symbol)}] WebSocket conectado.", bypass_throttle=True)
                reconnect_attempt = 0

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth_raw = await asyncio.wait_for(ws.recv(), timeout=10)

                df = None
                history_tries = 0
                while True:
                    history_tries += 1
                    try:
                        req_hist = {"ticks_history": symbol, "count": 200, "end": "latest",
                                    "granularity": GRANULARITY_SECONDS, "style": "candles"}
                        await ws.send(json.dumps(req_hist))
                        raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(raw)
                        if "history" in data and "candles" in data["history"] and len(data["history"]["candles"]) > 0:
                            df = pd.DataFrame(data["history"]["candles"])
                            break
                        if "candles" in data and len(data["candles"]) > 0:
                            df = pd.DataFrame(data["candles"])
                            break
                        await asyncio.sleep(1.0 + random.random() * 1.5)
                    except:
                        if history_tries >= 3: raise
                        await asyncio.sleep(1.0 + random.random() * 2.0)

                df = calcular_indicadores(df)
                save_last_candles(df, symbol)
                send_telegram(f"📥 [{human_pair(symbol)}] Histórico inicial ({len(df)} candles) carregado.", bypass_throttle=True)

                await ws.send(json.dumps({"ticks_history": symbol, "style": "candles",
                                          "granularity": GRANULARITY_SECONDS, "end": "latest", "subscribe": 1}))
                send_telegram(f"🔔 [{human_pair(symbol)}] Inscrito em candles ao vivo (M{CANDLE_INTERVAL}).", bypass_throttle=True)
                ultimo_candle_time = time.time()

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    except asyncio.TimeoutError:
                        if time.time() - ultimo_candle_time > 300: raise Exception("Timeout prolongado, reconectar")
                        continue

                    try: msg = json.loads(raw)
                    except: continue

                    candle = None
                    if isinstance(msg, dict):
                        if "candle" in msg: candle = msg["candle"]
                        elif "ohlc" in msg: candle = msg["ohlc"]
                        elif "history" in msg and "candles" in msg["history"]:
                            candle = msg["history"]["candles"][-1]
                        elif "candles" in msg and len(msg["candles"]) > 0:
                            candle = msg["candles"][-1]

                    if candle is None: continue

                    try:
                        epoch = int(candle.get("epoch"))
                        open_p = float(candle.get("open"))
                        high_p = float(candle.get("high"))
                        low_p = float(candle.get("low"))
                        close_p = float(candle.get("close"))
                        volume_p = float(candle.get("volume", 0))
                    except: continue

                    if epoch % GRANULARITY_SECONDS != 0: continue

                    candle_time_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    ultimo_candle_time = time.time()
                    last_epoch_in_df = int(df.iloc[-1]["epoch"]) if not df.empty else None
                    if df.empty or last_epoch_in_df != epoch:
                        df.loc[len(df)] = {"epoch": epoch, "open": open_p, "high": high_p, "low": low_p,
                                           "close": close_p, "volume": volume_p}
                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        sinal = gerar_sinal(df, symbol)
                        if sinal:
                            now_ts = time.time()
                            last_ts = last_signal_time.get(symbol, 0)
                            if now_ts - last_ts < MIN_SECONDS_BETWEEN_SIGNALS: continue
                            tipo = sinal["tipo"]
                            forca = sinal["forca"]
                            arrow = "🟢" if tipo == "COMPRA" else "🔴"
                            price = close_p
                            entrada_br = candle_time_utc.astimezone(timezone(timedelta(hours=-3)))
                            entrada_str = entrada_br.strftime("%Y-%m-%d %H:%M:%S")
                            msg_final = (f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                         f"• Par: {human_pair(symbol)}\n"
                                         f"• Direção: {arrow} *{tipo}*\n"
                                         f"• Força do sinal: *{forca}%*\n"
                                         f"• Preço: {price:.5f}\n"
                                         f"• Horário de entrada (Brasília): {entrada_str}")
                            last_signal_time[symbol] = time.time()
                            send_telegram(msg_final, symbol=symbol)

        except:
            await asyncio.sleep(min(60, (2 ** reconnect_attempt) * 0.5 * random.uniform(0.5,1.5)))

# ---------------- Flask ----------------
app = Flask(__name__)
@app.route("/")
def index():
    return "Bot ativo — Lógica B (ajustada) — com força do sinal"

def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run("0.0.0.0", port, debug=False, use_reloader=False)

# ---------------- Main ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado — Lógica B ajustada + Força do Sinal", bypass_throttle=True)
    await asyncio.gather(*(monitor_symbol(s) for s in SYMBOLS))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
