# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (ajustada) — 1 WS por par (Opção A) — COMPLETO
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime
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

# ---------------- Parâmetros de sinal/freq (Config B - Volume alto) ----------------
# Configuração agressiva para gerar 30-50 sinais / dia
SIGNAL_MIN_INTERVAL_SECONDS = 0    # cooldown por par (0 = sem cooldown para volume alto)
BB_PROXIMITY_PCT = 0.45           # 45% -> mais proximidade aceita
RSI_STRICT_BUY = 60
RSI_STRICT_SELL = 40
RSI_RELAX = 70
USE_MACD_CONFIRMATION = True
MACD_DIFF_RELAX = 0.004

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

# ---------------- Lógica de Sinal — OPÇÃO 1 (Lógica B mais leve/agressiva) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < 3:
            return None
        
        now = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(now["close"])
        epoch = int(now["epoch"])

        ema20_now = now.get("ema20")
        ema50_now = now.get("ema50")
        ema20_prev = prev.get("ema20")
        ema50_prev = prev.get("ema50")
        rsi_now = now.get("rsi")

        bb_upper = now.get("bb_upper")
        bb_lower = now.get("bb_lower")
        bb_mavg  = now.get("bb_mavg")

        macd_diff = now.get("macd_diff", None)

        # Verifica NaNs
        if any(pd.isna([ema20_now, ema50_now, ema20_prev, ema50_prev, rsi_now, bb_upper, bb_lower])):
            log(f"[{symbol}] Indicadores incompletos (NaN) — aguardando mais candles.")
            return None

        # ---------------- Cruzamentos / direção ----------------
        # Opção 1: aceita direção (ou cruzamento imediato)
        cruzou_para_cima  = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
        cruzou_para_baixo = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)
        # também aceita simplesmente EMA20 > EMA50 (tendência de alta) / EMA20 < EMA50 (tendência de baixa)
        tendencia_alta = ema20_now > ema50_now
        tendencia_baixa = ema20_now < ema50_now

        # ---------------- Proximidade Bollinger (mais leve) ----------------
        banda_range = bb_upper - bb_lower
        if banda_range == 0:
            return None
        lim_inf = bb_lower + banda_range * 0.15
        lim_sup = bb_upper - banda_range * 0.15

        perto_da_lower = close <= lim_inf
        perto_da_upper = close >= lim_sup

        # ---------------- RSI leve (mais permissivo) ----------------
        buy_rsi_ok  = rsi_now <= 55
        sell_rsi_ok = rsi_now >= 45

        # ---------------- MACD opcional ----------------
        macd_buy_ok = True
        macd_sell_ok = True
        if macd_diff is not None and not pd.isna(macd_diff):
            macd_buy_ok  = macd_diff > -MACD_DIFF_RELAX
            macd_sell_ok = macd_diff <  MACD_DIFF_RELAX

        # ---------------- Condições finais ----------------
        # Aceita cruzamento imediato OU direção (tendência) para aumentar volume de sinais
        cond_buy = (cruzou_para_cima or tendencia_alta) and perto_da_lower and buy_rsi_ok and macd_buy_ok
        cond_sell = (cruzou_para_baixo or tendencia_baixa) and perto_da_upper and sell_rsi_ok and macd_sell_ok

        # Cooldown (usa constante configurável)
        now_ts = time.time()
        last_time = last_signal_time.get(symbol, 0)
        if SIGNAL_MIN_INTERVAL_SECONDS and (now_ts - last_time < SIGNAL_MIN_INTERVAL_SECONDS):
            # se está em cooldown, não envia -- mas como em config B normalmente é 0, não bloqueará
            if cond_buy or cond_sell:
                log(f"[{symbol}] Condição satisfeita, mas em cooldown ({int(now_ts-last_time)}s desde último sinal).")
            return None

        # ------------ Geração do sinal ------------
        if cond_buy:
            last_signal_time[symbol] = now_ts
            last_signal_state[symbol] = "COMPRA"
            last_signal_candle[symbol] = epoch
            log(f"📈 [{symbol}] SINAL COMPRA — close={close:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
            return "COMPRA"

        if cond_sell:
            last_signal_time[symbol] = now_ts
            last_signal_state[symbol] = "VENDA"
            last_signal_candle[symbol] = epoch
            log(f"📉 [{symbol}] SINAL VENDA — close={close:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
            return "VENDA"

        return None

    except Exception as e:
        log(f"[{symbol}] Erro em gerar_sinal (opção 1): {e}\n{traceback.format_exc()}", "error")
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

                    # parse robusto e extração do candle (aceita candle/ohlc/history/candles)
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        log(f"[{symbol}] Mensagem não-JSON recebida (raw): {raw}", "warning")
                        continue

                    # função local para normalizar/retornar um candle dict ou None
                    def _extract_candle_from_msg(obj):
                        # 1) mensagem com chave "candle" (padrão usado antes)
                        if isinstance(obj, dict) and "candle" in obj and isinstance(obj["candle"], dict):
                            c = obj["candle"]
                            try:
                                c_norm = {
                                    "epoch": int(c.get("epoch")),
                                    "open": float(c.get("open")),
                                    "high": float(c.get("high")),
                                    "low": float(c.get("low")),
                                    "close": float(c.get("close")),
                                    "volume": float(c.get("volume")) if c.get("volume") is not None else 0.0
                                }
                                return c_norm
                            except Exception:
                                return None

                        # 2) mensagens com "ohlc"
                        if isinstance(obj, dict) and "ohlc" in obj:
                            o = obj["ohlc"]
                            if isinstance(o, dict):
                                try:
                                    c_norm = {
                                        "epoch": int(o.get("epoch")),
                                        "open": float(o.get("open")),
                                        "high": float(o.get("high")),
                                        "low": float(o.get("low")),
                                        "close": float(o.get("close")),
                                        "volume": float(o.get("volume")) if o.get("volume") is not None else 0.0
                                    }
                                    return c_norm
                                except Exception:
                                    return None

                        # 3) mensagens com "history" que contenham "candles" array
                        if isinstance(obj, dict) and "history" in obj and isinstance(obj["history"], dict):
                            history = obj["history"]
                            if isinstance(history.get("candles"), list) and len(history["candles"]) > 0:
                                last = history["candles"][-1]
                                try:
                                    c_norm = {
                                        "epoch": int(last.get("epoch")),
                                        "open": float(last.get("open")),
                                        "high": float(last.get("high")),
                                        "low": float(last.get("low")),
                                        "close": float(last.get("close")),
                                        "volume": float(last.get("volume")) if last.get("volume") is not None else 0.0
                                    }
                                    return c_norm
                                except Exception:
                                    return None

                        # 4) mensagem contendo "candles" top-level
                        if isinstance(obj, dict) and "candles" in obj and isinstance(obj["candles"], list):
                            try:
                                last = obj["candles"][-1]
                                c_norm = {
                                    "epoch": int(last.get("epoch")),
                                    "open": float(last.get("open")),
                                    "high": float(last.get("high")),
                                    "low": float(last.get("low")),
                                    "close": float(last.get("close")),
                                    "volume": float(last.get("volume")) if last.get("volume") is not None else 0.0
                                }
                                return c_norm
                            except Exception:
                                return None

                        # 5) tick messages — ignoradas para candles fechados
                        if isinstance(obj, dict) and "tick" in obj and isinstance(obj["tick"], dict):
                            return None

                        return None

                    candle = _extract_candle_from_msg(msg)

                    # para debug: log de controle (quando msg tiver msg_type)
                    if candle is None:
                        if isinstance(msg, dict) and msg.get("msg_type"):
                            log(f"[{symbol}] Recebeu msg de controle: {msg.get('msg_type')}")
                        # else: desconhecido — silencioso para não encher logs

                    # se temos candle normalizado, use-o
                    if candle:
                        candle_time = datetime.utcfromtimestamp(candle['epoch']).strftime('%Y-%m-%d %H:%M:%S')
                        log(f"[{symbol}] Novo candle (normalizado) recebido às {candle_time} UTC | close={candle['close']}")
                        ultimo_candle_time = time.time()

                        # adiciona novo candle (fechado) ao df
                        if df.empty or int(df.iloc[-1]["epoch"]) != int(candle["epoch"]):
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
                                entrada_utc = datetime.utcfromtimestamp(candle["epoch"]).strftime("%Y-%m-%d %H:%M:%S")
                                mensagem_sinal = (
                                    f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                    f"• Par: {symbol.replace('frx','')}\n"
                                    f"• Direção: {arrow} *{sinal}*\n"
                                    f"• Preço: {close_price:.5f}\n"
                                    f"• Horário de entrada (UTC): {entrada_utc}"
                                )
                                send_telegram(mensagem_sinal, symbol)

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
