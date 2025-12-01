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
# Alvo aproximado: ajustar SIGNAL_MIN_INTERVAL_SECONDS para controlar volume.
# Ex: 3600 = 1 sinal por hora por par (reduz sinais); 1800 = ~2/h por par (aumenta sinais).
SIGNAL_MIN_INTERVAL_SECONDS = 1800  # 30 minutos entre sinais do mesmo par (ajustável)

# Relaxamento RSI: valores padrão (ajustáveis)
RSI_STRICT_BUY = 40   # se RSI <= 40 => compra imediata (mais criterioso)
RSI_STRICT_SELL = 60  # se RSI >= 60 => venda imediata
RSI_RELAX = 45        # se MACD forte, aceitar RSI até 45/55

# MACD confirmar sinal? (True/False)
USE_MACD_CONFIRMATION = True
MACD_DIFF_RELAX = 0.0001  # threshold pequeno para considerar "MACD forte"

# Bollinger proximity relaxation: se close estiver *próximo* (pct) da banda, ainda aceitar
BB_PROXIMITY_PCT = 0.02  # 2% do spread mavg->band como "próximo"

# ---------------- Estado / controle ----------------
last_signal_state = {s: None for s in SYMBOLS}  # None / "COMPRA" / "VENDA"
last_signal_candle = {s: None for s in SYMBOLS}  # epoch do candle do último sinal
last_signal_time = {s: 0 for s in SYMBOLS}  # timestamp do último sinal (para cooldown)
sent_download_message = {s: False for s in SYMBOLS}
last_notify_time = {}  # throttle telegram por par

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
        # evita flood telegram para o mesmo par (3 segundos)
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

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    # EMAs
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    # RSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    # MACD (para confirmação opcional)
    try:
        macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd"] = pd.NA
        df["macd_signal"] = pd.NA
        df["macd_diff"] = pd.NA

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df

# ---------------- Função que decide sinal (Lógica B ajustada) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """
    Regras (LÓGICA B - ajustada):
      - SINAL APENAS NO FECHAMENTO DO CANDLE (chamado quando chega novo candle)
      - COMPRA:
          * EMA20 prev <= EMA50 prev  AND EMA20 now > EMA50 now  (cruzou pra cima)
          * close now <= bb_lower  OR close "próximo" à bb_lower (porcentagem)
          * RSI now abaixo de threshold (flexível com MACD)
          * (opcional) MACD confirma (macd_diff > 0)
      - VENDA:
          * EMA20 prev >= EMA50 prev  AND EMA20 now < EMA50 now  (cruzou pra baixo)
          * close now >= bb_upper OR "próximo" à bb_upper
          * RSI now acima de threshold
          * (opcional) MACD confirma (macd_diff < 0)
      - Evita reenviar se:
          * mesmo candle já enviado
          * cooldown (SIGNAL_MIN_INTERVAL_SECONDS) não expirou
    """
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

        # validar NaNs
        if any(pd.isna([ema20_prev, ema50_prev, ema20_now, ema50_now, rsi_now, bb_lower, bb_upper])):
            log(f"[{symbol}] Indicadores incompletos (NaN) — aguardando mais candles.")
            return None

        # logs úteis
        try:
            log(f"[{symbol}] close={close:.5f} EMA20_prev={ema20_prev:.5f} EMA50_prev={ema50_prev:.5f} EMA20_now={ema20_now:.5f} EMA50_now={ema50_now:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
        except Exception:
            log(f"[{symbol}] (log indicadores: possível NaN)")

        # cruzamentos
        cruzou_para_cima = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
        cruzou_para_baixo = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)

        # proximidade com banda (permissão de "próximo")
        bb_gap_buy = bb_mavg - bb_lower
        bb_gap_sell = bb_upper - bb_mavg
        # distância relativa para considerar "próximo"
        prox_buy_threshold = bb_lower + BB_PROXIMITY_PCT * bb_gap_buy
        prox_sell_threshold = bb_upper - BB_PROXIMITY_PCT * bb_gap_sell

        close_below_lower = close <= bb_lower or close <= prox_buy_threshold
        close_above_upper = close >= bb_upper or close >= prox_sell_threshold

        # RSI regras (relaxadas com MACD)
        buy_rsi_ok = False
        sell_rsi_ok = False

        # regra strict
        if rsi_now <= RSI_STRICT_BUY:
            buy_rsi_ok = True
        elif RSI_STRICT_BUY < rsi_now <= RSI_RELAX:
            # permitir se MACD estiver fortemente positivo (se estamos usando)
            if USE_MACD_CONFIRMATION and (not pd.isna(macd_diff)) and macd_diff > MACD_DIFF_RELAX:
                buy_rsi_ok = True
        # symmetrical for sell
        if rsi_now >= RSI_STRICT_SELL:
            sell_rsi_ok = True
        elif RSI_RELAX <= rsi_now < RSI_STRICT_SELL:
            if USE_MACD_CONFIRMATION and (not pd.isna(macd_diff)) and macd_diff < -MACD_DIFF_RELAX:
                sell_rsi_ok = True

        # MACD confirmation if enabled — extra gate (but we already used it for relax)
        macd_confirms_buy = True
        macd_confirms_sell = True
        if USE_MACD_CONFIRMATION:
            if pd.isna(macd_diff):
                macd_confirms_buy = False
                macd_confirms_sell = False
            else:
                macd_confirms_buy = macd_diff > -MACD_DIFF_RELAX  # tolerate small noise
                macd_confirms_sell = macd_diff < MACD_DIFF_RELAX

        # assemble conditions
        cond_buy = cruzou_para_cima and close_below_lower and buy_rsi_ok and macd_confirms_buy
        cond_sell = cruzou_para_baixo and close_above_upper and sell_rsi_ok and macd_confirms_sell

        # cooldown check
        last_time = last_signal_time.get(symbol, 0)
        now_ts = time.time()
        if now_ts - last_time < SIGNAL_MIN_INTERVAL_SECONDS:
            # ainda em cooldown; permitimos avaliar mas não enviar novo sinal
            if cond_buy or cond_sell:
                log(f"[{symbol}] Condição satisfeita, mas em cooldown ({int(now_ts-last_time)}s desde último sinal).")
            return None

        # evitar repetir sinal no mesmo candle
        current_state = last_signal_state.get(symbol)
        if cond_buy:
            if current_state == "COMPRA" and last_signal_candle.get(symbol) == epoch:
                log(f"[{symbol}] COMPRA já enviada neste candle (skip).")
                return None
            # envia sinal
            last_signal_state[symbol] = "COMPRA"
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = now_ts
            log(f"✅ [{symbol}] SINAL COMPRA (Lógica B) gerado. close={close:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
            return "COMPRA"

        if cond_sell:
            if current_state == "VENDA" and last_signal_candle.get(symbol) == epoch:
                log(f"[{symbol}] VENDA já enviada neste candle (skip).")
                return None
            last_signal_state[symbol] = "VENDA"
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = now_ts
            log(f"✅ [{symbol}] SINAL VENDA (Lógica B) gerado. close={close:.5f} RSI={rsi_now:.2f} macd_diff={macd_diff}")
            return "VENDA"

        # se nada, limpa estado (permite novo sinal depois)
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

# ---------------- Monitor por símbolo (conexão 24/7 + reconexão infinita) ----------------
async def monitor_symbol(symbol: str):
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                send_telegram(f"🔌 [{symbol}] Conectado ao WebSocket.", symbol)
                log(f"[{symbol}] WebSocket conectado.")

                # Autorização
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

                # Histórico inicial (200 candles)
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

                # Mensagem de download separada do sinal (apenas 1x por deploy)
                if not sent_download_message[symbol]:
                    send_telegram(f"📥 [{symbol}] Download de velas completo ({len(df)} candles).", symbol)
                    sent_download_message[symbol] = True

                # Subscribes para candles ao vivo
                sub_req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))
                log(f"[{symbol}] Assinado para candles ao vivo.")

                ultimo_candle_time = time.time()

                # Loop de recebimento - cada msg geralmente contém campo "candle"
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    except asyncio.TimeoutError:
                        if time.time() - ultimo_candle_time > 300:
                            log(f"[{symbol}] Nenhum candle há 5 minutos — forçando reconexão.", "warning")
                            raise Exception("Reconexão forçada por inatividade")
                        else:
                            log(f"[{symbol}] Timeout aguardando mensagem, mantendo conexão...", "info")
                            continue

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        log(f"[{symbol}] Mensagem não-JSON recebida: {raw}", "warning")
                        continue

                    candle = msg.get("candle")
                    if not candle:
                        # ignorar outros tipos de mensagens
                        continue

                    # se for um novo candle fechado
                    if df.empty or df.iloc[-1]["epoch"] != candle["epoch"]:
                        df.loc[len(df)] = candle
                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        # gera sinal apenas após fechamento (quando candle novo é recebido)
                        sinal = gerar_sinal(df, symbol)
                        if sinal:
                            arrow = "🟢" if sinal == "COMPRA" else "🔴"
                            close_price = float(candle["close"])
                            utc_time = datetime.utcnow().strftime("%H:%M:%S")
                            mensagem_sinal = (
                                f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                f"• Par: {symbol.replace('frx','')}\n"
                                f"• Direção: {arrow} *{sinal}*\n"
                                f"• Preço: {close_price:.5f}\n"
                                f"• Horário: {utc_time} UTC"
                            )
                            # envia a mensagem de sinal separada da mensagem de download
                            send_telegram(mensagem_sinal, symbol)

                    ultimo_candle_time = time.time()

        except Exception as e:
            log(f"[{symbol}] ERRO no WebSocket / loop: {e}\n{traceback.format_exc()}", "error")
            # backoff curto com jitter para evitar bursts no Render gratuito
            await asyncio.sleep(random.uniform(2.0, 6.0))
            continue

# ---------------- Flask (diagnóstico) ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Lógica B (EMA20/50 + BB20 + RSI14) — 24/7"

# ---------------- Execução principal ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    log(f"🌐 Iniciando Flask na porta {port} (no thread).")
    # importante: use_reloader=False para evitar múltiplos processos
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

async def main():
    # roda Flask em thread separada (sem reloader)
    threading.Thread(target=run_flask, daemon=True).start()

    send_telegram("✅ Bot iniciado com LÓGICA B (EMA20/EMA50 + BB20 + RSI14).")
    log("▶ Iniciando monitoramento paralelo por par (24/7, reconexão infinita)...")

    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
