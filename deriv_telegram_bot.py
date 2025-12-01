# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (ajustada) — 24/7, reconexão infinita
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback
import pytz

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações principais (ajustáveis) ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos (FTT 5m)
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxGBPAUD", "frxGBPCAD", "frxAUDNZD",
    "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Sinal / frequência (tune aqui) ----------------
SIGNAL_MIN_INTERVAL_SECONDS = 1800  # 30 minutos entre sinais do mesmo par

RSI_STRICT_BUY = 40
RSI_STRICT_SELL = 60
RSI_RELAX = 45

USE_MACD_CONFIRMATION = True
MACD_DIFF_RELAX = 0.0001

BB_PROXIMITY_PCT = 0.02

# ---------------- Estado / controle ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
sent_download_message = {s: False for s in SYMBOLS}
last_notify_time = {}

# ---------------- Logging (Render-friendly) ----------------
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
        print(msg, flush=True)
    elif level == "warning":
        logger.warning(msg)
        print(msg, flush=True)
    elif level == "error":
        logger.error(msg)
        print(msg, flush=True)
    else:
        logger.debug(msg)
        print(msg, flush=True)

# ---------------- Telegram helper ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()
    if symbol:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            log(f"Telegram rate limit skipped for {symbol}", "warning")
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("⚠️ Telegram não configurado. Mensagem: " + message, "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
        else:
            log(f"Telegram enviado: {message}")
    except Exception as e:
        log(f"❌ Erro ao enviar Telegram: {e}\n{traceback.format_exc()}", "error")


# ---------------- Função Modelo A — Mensagem formatada ----------------
def montar_mensagem_sinal(symbol, direction, price):
    tz = pytz.timezone("America/Sao_Paulo")

    agora = datetime.now(tz)
    detectado_str = agora.strftime("%H:%M:%S")

    minuto = agora.minute
    proximo_m5 = (minuto // 5 + 1) * 5
    entrada = agora.replace(second=0, microsecond=0)

    if proximo_m5 >= 60:
        entrada = entrada.replace(minute=0) + timedelta(hours=1)
    else:
        entrada = entrada.replace(minute=proximo_m5)

    entrada_str = entrada.strftime("%H:%M")

    symbol_clear = symbol.replace("frx", "")

    msg = (
        f"📊 *SINAL — M5*\n"
        f"🪙 Par: {symbol_clear}\n"
        f"📈 Direção: {direction}\n"
        f"💰 Preço: {price:.5f}\n"
        f"⏰ Detectado: {detectado_str} (UTC-3)\n"
        f"🎯 Entrada: {entrada_str} (UTC-3)"
    )

    return msg


# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    try:
        macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
    except:
        df["macd"] = pd.NA
        df["macd_signal"] = pd.NA
        df["macd_diff"] = pd.NA

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df


# ---------------- Lógica B ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < 2:
            return None

        now_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        epoch = int(now_row["epoch"])
        close = float(now_row["close"])

        ema20_now = now_row["ema20"]
        ema50_now = now_row["ema50"]
        ema20_prev = prev_row["ema20"]
        ema50_prev = prev_row["ema50"]
        rsi_now = now_row["rsi"]
        bb_lower = now_row["bb_lower"]
        bb_upper = now_row["bb_upper"]
        bb_mavg = now_row["bb_mavg"]
        macd_diff = now_row.get("macd_diff")

        if any(pd.isna([ema20_prev, ema50_prev, ema20_now, ema50_now, rsi_now, bb_lower, bb_upper])):
            return None

        cruzou_para_cima = ema20_prev <= ema50_prev and ema20_now > ema50_now
        cruzou_para_baixo = ema20_prev >= ema50_prev and ema20_now < ema50_now

        bb_gap_buy = bb_mavg - bb_lower
        bb_gap_sell = bb_upper - bb_mavg

        prox_buy = bb_lower + BB_PROXIMITY_PCT * bb_gap_buy
        prox_sell = bb_upper - BB_PROXIMITY_PCT * bb_gap_sell

        close_below_lower = close <= bb_lower or close <= prox_buy
        close_above_upper = close >= bb_upper or close >= prox_sell

        buy_rsi_ok = rsi_now <= RSI_STRICT_BUY or (
            rsi_now <= RSI_RELAX and USE_MACD_CONFIRMATION and macd_diff > MACD_DIFF_RELAX
        )

        sell_rsi_ok = rsi_now >= RSI_STRICT_SELL or (
            rsi_now >= RSI_RELAX and USE_MACD_CONFIRMATION and macd_diff < -MACD_DIFF_RELAX
        )

        macd_confirms_buy = True
        macd_confirms_sell = True
        if USE_MACD_CONFIRMATION and not pd.isna(macd_diff):
            macd_confirms_buy = macd_diff > -MACD_DIFF_RELAX
            macd_confirms_sell = macd_diff < MACD_DIFF_RELAX

        cond_buy = cruzou_para_cima and close_below_lower and buy_rsi_ok and macd_confirms_buy
        cond_sell = cruzou_para_baixo and close_above_upper and sell_rsi_ok and macd_confirms_sell

        last_time = last_signal_time.get(symbol, 0)
        now_ts = time.time()
        if now_ts - last_time < SIGNAL_MIN_INTERVAL_SECONDS:
            return None

        last_state = last_signal_state.get(symbol)
        if cond_buy:
            if last_state == "COMPRA" and last_signal_candle.get(symbol) == epoch:
                return None

            last_signal_state[symbol] = "COMPRA"
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = now_ts
            return "COMPRA"

        if cond_sell:
            if last_state == "VENDA" and last_signal_candle.get(symbol) == epoch:
                return None

            last_signal_state[symbol] = "VENDA"
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = now_ts
            return "VENDA"

        last_signal_state[symbol] = None
        last_signal_candle[symbol] = None
        return None

    except Exception:
        return None


# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)


# ---------------- MONITOR — 24/7 & reconexão infinita ----------------
async def monitor_symbol(symbol: str):
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                send_telegram(f"🔌 [{symbol}] Conectado ao WebSocket.", symbol)
                log(f"[{symbol}] WebSocket conectado.")

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth = json.loads(await ws.recv())
                if "authorize" not in auth:
                    raise Exception("Falha na autorização")

                # Histórico inicial
                hist_req = {
                    "ticks_history": symbol,
                    "count": 200,
                    "end": "latest",
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(hist_req))
                data = json.loads(await ws.recv())

                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)

                if not sent_download_message[symbol]:
                    send_telegram(f"📥 [{symbol}] Download de velas completo ({len(df)} candles).", symbol)
                    sent_download_message[symbol] = True

                sub_req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))

                ultimo_candle_time = time.time()

                # ====================== LOOP DOS CANDLES AO VIVO ======================
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    except asyncio.TimeoutError:
                        if time.time() - ultimo_candle_time > 300:
                            raise Exception("Reconexão por inatividade")
                        continue

                    msg = json.loads(raw)
                    candle = msg.get("candle")
                    if not candle:
                        continue

                    if df.empty or df.iloc[-1]["epoch"] != candle["epoch"]:
                        df.loc[len(df)] = candle
                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        sinal = gerar_sinal(df, symbol)
                        if sinal:
                            price = float(candle["close"])
                            mensagem = montar_mensagem_sinal(symbol, sinal, price)
                            send_telegram(mensagem, symbol)

                    ultimo_candle_time = time.time()

        except Exception as e:
            log(f"[{symbol}] ERRO: {e}")
            await asyncio.sleep(random.uniform(2, 6))
            continue


# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Lógica B — 24/7"


# ---------------- Execução principal ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


async def main():
    threading.Thread(target=run_flask, daemon=True).start()

    send_telegram("✅ Bot iniciado com LÓGICA B (EMA20/EMA50 + BB20 + RSI14).")
    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
