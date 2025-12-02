# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (ajustada) — 1 WS por par (Opção A)
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timezone
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

# ---------------- Parâmetros de sinal/freq ----------------
SIGNAL_MIN_INTERVAL_SECONDS = 1800  # cooldown por par (ajustável)
RSI_STRICT_BUY = 40
RSI_STRICT_SELL = 60
RSI_RELAX = 45
USE_MACD_CONFIRMATION = True
MACD_DIFF_RELAX = 0.0001
BB_PROXIMITY_PCT = 0.02  # 2%

# ---------------- Estado / controle ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
sent_download_message = {s: False for s in SYMBOLS}
last_notify_time = {}  # throttle telegram por par

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

    # EMAs
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    # RSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    # MACD (opcional)
    try:
        macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd"] = pd.NA
        df["macd_signal"] = pd.NA
        df["macd_diff"] = pd.NA

    # Bollinger
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df

# ---------------- Lógica de Sinal (Lógica B) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < 2:
            return None

        now_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        epoch = int(now_row["epoch"])
        close = float(now_row["close"])

        ema20_now = now_row.get("ema20")
        ema50_now = now_row.get("ema50")
        ema20_prev = prev_row.get("ema20")
        ema50_prev = prev_row.get("ema50")
        rsi_now = now_row.get("rsi")
        bb_lower = now_row.get("bb_lower")
        bb_upper = now_row.get("bb_upper")
        bb_mavg = now_row.get("bb_mavg")
        macd_diff = now_row.get("macd_diff")

        if any(pd.isna([ema20_prev, ema50_prev, ema20_now, ema50_now, rsi_now, bb_lower, bb_upper])):
            log(f"[{symbol}] Indicadores incompletos (NaN) — aguardando mais candles.")
            return None

        # logs
        try:
            log(f"[{symbol}] close={close:.5f} EMA20_prev={ema20_prev:.5f} EMA50_prev={ema50_prev:.5f} EMA20_now={ema20_now:.5f} EMA50_now={ema50_now:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
        except Exception:
            log(f"[{symbol}] (log indicadores: possível NaN)")

        cruzou_para_cima = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
        cruzou_para_baixo = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)

        bb_gap_buy = bb_mavg - bb_lower
        bb_gap_sell = bb_upper - bb_mavg
        prox_buy_threshold = bb_lower + BB_PROXIMITY_PCT * bb_gap_buy
        prox_sell_threshold = bb_upper - BB_PROXIMITY_PCT * bb_gap_sell

        close_below_lower = close <= bb_lower or close <= prox_buy_threshold
        close_above_upper = close >= bb_upper or close >= prox_sell_threshold

        # RSI rules with MACD relax
        buy_rsi_ok = False
        sell_rsi_ok = False

        if rsi_now <= RSI_STRICT_BUY:
            buy_rsi_ok = True
        elif RSI_STRICT_BUY < rsi_now <= RSI_RELAX:
            if USE_MACD_CONFIRMATION and (not pd.isna(macd_diff)) and macd_diff > MACD_DIFF_RELAX:
                buy_rsi_ok = True

        if rsi_now >= RSI_STRICT_SELL:
            sell_rsi_ok = True
        elif RSI_RELAX <= rsi_now < RSI_STRICT_SELL:
            if USE_MACD_CONFIRMATION and (not pd.isna(macd_diff)) and macd_diff < -MACD_DIFF_RELAX:
                sell_rsi_ok = True

        macd_confirms_buy = True
        macd_confirms_sell = True
        if USE_MACD_CONFIRMATION:
            if pd.isna(macd_diff):
                macd_confirms_buy = False
                macd_confirms_sell = False
            else:
                macd_confirms_buy = macd_diff > -MACD_DIFF_RELAX
                macd_confirms_sell = macd_diff < MACD_DIFF_RELAX

        cond_buy = cruzou_para_cima and close_below_lower and buy_rsi_ok and macd_confirms_buy
        cond_sell = cruzou_para_baixo and close_above_upper and sell_rsi_ok and macd_confirms_sell

        # cooldown
        last_time = last_signal_time.get(symbol, 0)
        now_ts = time.time()
        if now_ts - last_time < SIGNAL_MIN_INTERVAL_SECONDS:
            if cond_buy or cond_sell:
                log(f"[{symbol}] Condição satisfeita, mas em cooldown ({int(now_ts-last_time)}s desde último sinal).")
            return None

        current_state = last_signal_state.get(symbol)
        if cond_buy:
            if current_state == "COMPRA" and last_signal_candle.get(symbol) == epoch:
                log(f"[{symbol}] COMPRA já enviada neste candle (skip).")
                return None
            last_signal_state[symbol] = "COMPRA"
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = now_ts
            log(f"✅ [{symbol}] SINAL COMPRA gerado. close={close:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
            return "COMPRA"

        if cond_sell:
            if current_state == "VENDA" and last_signal_candle.get(symbol) == epoch:
                log(f"[{symbol}] VENDA já enviada neste candle (skip).")
                return None
            last_signal_state[symbol] = "VENDA"
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = now_ts
            log(f"✅ [{symbol}] SINAL VENDA gerado. close={close:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
            return "VENDA"

        if last_signal_state.get(symbol) is not None:
            last_signal_state[symbol] = None
            last_signal_candle[symbol] = None
            log(f"[{symbol}] Estado de sinal limpo (nenhuma condição ativa).")

        return None

    except Exception as e:
        log(f"[{symbol}] Erro em gerar_sinal: {e}\n{traceback.format_exc()}", "error")
        return None

# ---------------- Persistência candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    log(f"[{symbol}] Últimos candles salvos em {path}")

# ---------------- Monitor por símbolo (1 WS por par) ----------------
async def monitor_symbol(symbol: str):
    reconnect_attempt = 0
    while True:
        try:
            reconnect_attempt += 1
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                log(f"[{symbol}] WebSocket conectado (attempt {reconnect_attempt}).")
                send_telegram(f"🔌 [{symbol}] Conectado ao WebSocket.", symbol)

                # Authorize
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth_raw = await ws.recv()
                try:
                    auth = json.loads(auth_raw)
                except Exception:
                    log(f"[{symbol}] Não foi possível parsear authorize response: {auth_raw}", "error")
                    raise Exception("Authorize parse error")

                if "error" in auth:
                    log(f"[{symbol}] Resposta de authorize contém error: {auth}", "error")
                    raise Exception("Authorize error from Deriv")
                if "authorize" not in auth:
                    log(f"[{symbol}] Falha na autorização (payload inesperado): {auth}", "error")
                    raise Exception("Falha na autorização Deriv (campo 'authorize' ausente).")

                log(f"[{symbol}] Autorizado na Deriv.")

                # Request initial candles (200)
                req_hist = {
                    "ticks_history": symbol,
                    "count": 200,
                    "end": "latest",
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req_hist))
                data_raw = await ws.recv()
                try:
                    data = json.loads(data_raw)
                except Exception:
                    log(f"[{symbol}] Erro ao parsear histórico inicial: {data_raw}", "error")
                    raise

                if "candles" not in data:
                    log(f"[{symbol}] Histórico inicial sem candles: {data}", "warning")
                    await asyncio.sleep(5)
                    continue

                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)

                if not sent_download_message[symbol]:
                    send_telegram(f"📥 [{symbol}] Download de velas completo ({len(df)} candles).", symbol)
                    sent_download_message[symbol] = True

                # Subscribe to live candles (style candles)
                sub_req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))
                log(f"[{symbol}] Inscrito para candles ao vivo (subscribe enviado).")

                ultimo_candle_time = time.time()

                # Loop de recebimento
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    except asyncio.TimeoutError:
                        # reconecta se nada por 5 minutos
                        if time.time() - ultimo_candle_time > 300:
                            log(f"[{symbol}] Nenhum candle há 5 minutos — forçando reconexão.", "warning")
                            raise Exception("Reconexão por inatividade")
                        else:
                            log(f"[{symbol}] Timeout aguardando mensagem, mantendo conexão...", "info")
                            continue

                    # parse
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        log(f"[{symbol}] Mensagem não-JSON recebida: {raw}", "warning")
                        continue

                    # quando Deriv retorna candles em subscription, a mensagem contém "candle" (ou "history" no inicial)
                    candle = msg.get("candle")
                    history = msg.get("history")
                    # Log qualquer mensagem para debug (mas cuidado com volume)
                    #log(f"[{symbol}] Mensagem recebida: {msg}")

                    if history:
                        # historico incremental / updates — ignorar pois já baixamos initial
                        log(f"[{symbol}] Histórico recebido via subscription (tamanho={len(history.get('candles', []))}).")
                        ultimo_candle_time = time.time()
                        continue

                    if candle:
                        # debug: marque que recebeu um candle ao vivo
                        candle_time = datetime.utcfromtimestamp(candle['epoch']).strftime('%Y-%m-%d %H:%M:%S')
                        log(f"[{symbol}] Novo candle recebido às {candle_time} UTC | close={candle['close']}")
                        ultimo_candle_time = time.time()

                        # adiciona novo candle (fechado)
                        if df.empty or df.iloc[-1]["epoch"] != candle["epoch"]:
                            df.loc[len(df)] = candle
                            df = calcular_indicadores(df)
                            save_last_candles(df, symbol)

                            # gera sinal (apenas no fechamento do candle)
                            sinal = gerar_sinal(df, symbol)
                            if sinal:
                                arrow = "🟢" if sinal == "COMPRA" else "🔴"
                                close_price = float(candle["close"])
                                # horário de entrada = epoch do candle (UTC)
                                entrada_utc = datetime.utcfromtimestamp(candle["epoch"]).strftime("%Y-%m-%d %H:%M:%S")
                                mensagem_sinal = (
                                    f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                    f"• Par: {symbol.replace('frx','')}\n"
                                    f"• Direção: {arrow} *{sinal}*\n"
                                    f"• Preço: {close_price:.5f}\n"
                                    f"• Horário de entrada (UTC): {entrada_utc}"
                                )
                                send_telegram(mensagem_sinal, symbol)

                    else:
                        # mensagens sem candle (pong, subscription ack etc) — opcional log
                        # para debug ocasional:
                        if msg.get("echo_req") and msg.get("msg_type"):
                            log(f"[{symbol}] Recebeu msg de controle: {msg.get('msg_type')}")

        except Exception as e:
            log(f"[{symbol}] ERRO no WebSocket / loop: {e}\n{traceback.format_exc()}", "error")
            # backoff curto com jitter
            await asyncio.sleep(random.uniform(2.0, 6.0))
            continue

# ---------------- Flask (diagnóstico) ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Opção A (1 WS por par) — Lógica B (EMA20/50 + BB20 + RSI14)"

# ---------------- Execução ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    log(f"🌐 Iniciando Flask na porta {port} (no thread).")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado (Opção A — 1 WS por par) — Lógica B (EMA20/50 + BB20 + RSI14).")
    log("▶ Iniciando monitoramento paralelo por par (1 WS por par) ...")

    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
