# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (ajustada) — 1 WS por par (Opção A) — COMPLETO + CORREÇÕES
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações principais ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos (5m)
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

# ---------------- Parâmetros ajustados (CORREÇÃO 3 – mais assertivo) ----------------
SIGNAL_MIN_INTERVAL_SECONDS = 0
BB_PROXIMITY_PCT = 0.20     # porcentagem mais próxima das bandas → mais assertivo
RSI_BUY_MAX = 52            # antes: 55
RSI_SELL_MIN = 48           # antes: 45
MACD_TOLERANCE = 0.002      # macd mais restrito

# ---------------- Estado / controle ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
sent_download_message = {s: False for s in SYMBOLS}
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
        log(f"⚠️ Telegram não configurado. Mensagem: {message}", "warning")
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

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    try:
        macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd_diff"] = pd.NA

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df

# -------------- Lógica de sinal (OPÇÃO 1 — corrigida) --------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < 3:
            return None

        now = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(now["close"])
        epoch = int(now["epoch"])

        ema20_now = now["ema20"]
        ema50_now = now["ema50"]
        ema20_prev = prev["ema20"]
        ema50_prev = prev["ema50"]
        rsi_now = now["rsi"]
        bb_upper = now["bb_upper"]
        bb_lower = now["bb_lower"]
        macd_diff = now.get("macd_diff")

        if any(pd.isna([ema20_now, ema50_now, ema20_prev, ema50_prev, rsi_now, bb_upper, bb_lower])):
            return None

        # ---------------- Tendência e cruzamentos ----------------
        cruzou_up = ema20_prev <= ema50_prev and ema20_now > ema50_now
        cruzou_down = ema20_prev >= ema50_prev and ema20_now < ema50_now
        tendencia_up = ema20_now > ema50_now
        tendencia_down = ema20_now < ema50_now

        # ---------------- Bollinger ----------------
        banda_range = bb_upper - bb_lower
        if banda_range == 0:
            return None

        lim_inf = bb_lower + banda_range * BB_PROXIMITY_PCT
        lim_sup = bb_upper - banda_range * BB_PROXIMITY_PCT

        perto_lower = close <= lim_inf
        perto_upper = close >= lim_sup

        # ---------------- RSI ----------------
        buy_rsi_ok = rsi_now <= RSI_BUY_MAX
        sell_rsi_ok = rsi_now >= RSI_SELL_MIN

        # ---------------- MACD ----------------
        macd_buy_ok = True
        macd_sell_ok = True
        if macd_diff is not None and not pd.isna(macd_diff):
            macd_buy_ok = macd_diff > -MACD_TOLERANCE
            macd_sell_ok = macd_diff < MACD_TOLERANCE

        # ---------------- Condições ----------------
        cond_buy = (cruzou_up or tendencia_up) and perto_lower and buy_rsi_ok and macd_buy_ok
        cond_sell = (cruzou_down or tendencia_down) and perto_upper and sell_rsi_ok and macd_sell_ok

        # -------------- CORREÇÃO 2 — evitar duplicidade --------------
        if last_signal_candle.get(symbol) == epoch:
            return None

        # ------------ resultado ------------
        if cond_buy:
            last_signal_candle[symbol] = epoch
            return "COMPRA"

        if cond_sell:
            last_signal_candle[symbol] = epoch
            return "VENDA"

        return None

    except Exception as e:
        log(f"[{symbol}] Erro em gerar_sinal: {e}\n{traceback.format_exc()}", "error")
        return None

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)

# ---------------- Monitor por símbolo ----------------
async def monitor_symbol(symbol: str):
    reconnect_attempt = 0
    while True:
        try:
            reconnect_attempt += 1
            async with websockets.connect(WS_URL, ping_interval=None) as ws:

                # authorize
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                await ws.recv()

                # request history
                req_hist = {
                    "ticks_history": symbol,
                    "count": 200,
                    "end": "latest",
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req_hist))
                data = json.loads(await ws.recv())

                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)

                save_last_candles(df, symbol)

                # subscribe
                sub_req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))

                # evento
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)

                    candle = None

                    # normalização
                    if "candle" in msg:
                        candle = msg["candle"]
                    if "ohlc" in msg:
                        candle = msg["ohlc"]
                    if "candles" in msg:
                        candle = msg["candles"][-1]

                    if not candle:
                        continue

                    epoch = int(candle["epoch"])

                    if df.empty or int(df.iloc[-1]["epoch"]) != epoch:
                        df.loc[len(df)] = {
                            "epoch": candle["epoch"],
                            "open": candle["open"],
                            "high": candle["high"],
                            "low": candle["low"],
                            "close": candle["close"],
                            "volume": candle.get("volume", 0.0)
                        }

                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        sinal = gerar_sinal(df, symbol)

                        if sinal:
                            arrow = "🟢" if sinal == "COMPRA" else "🔴"
                            close_price = float(candle["close"])

                            # --------- CORREÇÃO 1 — Horário de Brasília ------------
                            entrada_brasilia = datetime.utcfromtimestamp(epoch) - timedelta(hours=3)
                            entrada_brasilia_str = entrada_brasilia.strftime("%Y-%m-%d %H:%M:%S")

                            mensagem = (
                                f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                f"• Par: {symbol.replace('frx','')}\n"
                                f"• Direção: {arrow} *{sinal}*\n"
                                f"• Preço: {close_price:.5f}\n"
                                f"• Horário de entrada (Brasília): {entrada_brasilia_str}"
                            )

                            send_telegram(mensagem, symbol)

        except Exception as e:
            log(f"[{symbol}] ERRO no WS: {e}")
            await asyncio.sleep(random.uniform(2, 6))
            continue

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Opção A — Lógica B Ajustada"

# ---------------- Execução ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("Bot iniciado — Lógica B ajustada")

    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
