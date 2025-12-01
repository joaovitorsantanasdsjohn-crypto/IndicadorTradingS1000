# ===============================================================
# deriv_telegram_bot.py — versão corrigida e instrumentada
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
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

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
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

# Controle de estado de sinais
last_signal_state = {s: None for s in SYMBOLS}       # None / "COMPRA" / "VENDA"
last_signal_candle = {s: None for s in SYMBOLS}      # epoch do candle que gerou o último sinal
sent_download_message = {s: False for s in SYMBOLS}
last_notify_time = {}                                # throttle Telegram por par

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    """Log para Render + flush imediato."""
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

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None):
    """Envia mensagem ao Telegram (com throttle por símbolo)."""
    now = time.time()
    if symbol:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            # evita flood (3s)
            log(f"Telegram rate limit skipped for {symbol}", "warning")
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log(f"⚠️ Telegram não configurado. Mensagem não enviada: {message}", "warning")
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
    """Calcula EMA20, EMA50, RSI14 e Bollinger20(2)."""
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    # EMAs
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    # RSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df

# ---------------- Lógica de Sinal (com CRUZAMENTO) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """
    Requisitos:
    - SINAL SÓ APÓS FECHAMENTO DA VELA (executado quando o novo candle chega)
    - COMPRA:
        1) EMA20 anterior <= EMA50 anterior
        2) EMA20 atual  > EMA50 atual  (cruzou pra cima)
        3) close atual < bb_lower (fechou abaixo da banda inferior)
        4) RSI atual < 50
    - VENDA:
        1) EMA20 anterior >= EMA50 anterior
        2) EMA20 atual  < EMA50 atual  (cruzou pra baixo)
        3) close atual > bb_upper (fechou acima da banda superior)
        4) RSI atual > 50
    - evita reenvio do mesmo sinal gerado no mesmo candle
    """

    # precisa de pelo menos 2 candles para detectar cruzamento
    if len(df) < 2:
        return None

    atual = df.iloc[-1]
    anterior = df.iloc[-2]

    epoch = int(atual["epoch"])
    close = float(atual["close"])

    ema20_now = atual.get("ema20")
    ema50_now = atual.get("ema50")
    ema20_prev = anterior.get("ema20")
    ema50_prev = anterior.get("ema50")
    rsi_now = atual.get("rsi")
    bb_lower = atual.get("bb_lower")
    bb_upper = atual.get("bb_upper")

    # log detalhado para debug
    try:
        log(f"[{symbol}] close={close:.5f} EMA20_prev={ema20_prev:.5f} EMA50_prev={ema50_prev:.5f} EMA20_now={ema20_now:.5f} EMA50_now={ema50_now:.5f} RSI={rsi_now:.1f} BB_low={bb_lower:.5f} BB_up={bb_upper:.5f}")
    except Exception:
        log(f"[{symbol}] (log indicadores: possível NaN)")

    # validação de NaNs
    if any(pd.isna([ema20_prev, ema50_prev, ema20_now, ema50_now, rsi_now, bb_lower, bb_upper])):
        log(f"[{symbol}] Indicadores incompletos (NaN) — aguardando mais candles.")
        return None

    # condição de COMPRA: cruzamento de baixo->cima + fechamento abaixo da banda inferior + rsi<50
    cruzou_para_cima = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
    cond_buy = cruzou_para_cima and (close < bb_lower) and (rsi_now < 50)

    # condição de VENDA: cruzamento de cima->baixo + fechamento acima da banda superior + rsi>50
    cruzou_para_baixo = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)
    cond_sell = cruzou_para_baixo and (close > bb_upper) and (rsi_now > 50)

    estado_atual = last_signal_state.get(symbol)

    # COMPRA
    if cond_buy:
        # evita reenviar no mesmo candle
        if estado_atual == "COMPRA" and last_signal_candle.get(symbol) == epoch:
            log(f"[{symbol}] COMPRA já enviada neste candle (skip).")
            return None
        last_signal_state[symbol] = "COMPRA"
        last_signal_candle[symbol] = epoch
        log(f"✅ [{symbol}] SINAL COMPRA gerado (cruzamento + fechamento abaixo BB + RSI<50).")
        return "COMPRA"

    # VENDA
    if cond_sell:
        if estado_atual == "VENDA" and last_signal_candle.get(symbol) == epoch:
            log(f"[{symbol}] VENDA já enviada neste candle (skip).")
            return None
        last_signal_state[symbol] = "VENDA"
        last_signal_candle[symbol] = epoch
        log(f"✅ [{symbol}] SINAL VENDA gerado (cruzamento + fechamento acima BB + RSI>50).")
        return "VENDA"

    # se nenhuma condição, limpar estado (permite novo sinal quando condição reaparecer)
    if last_signal_state.get(symbol) is not None:
        last_signal_state[symbol] = None
        last_signal_candle[symbol] = None
        log(f"[{symbol}] Estado de sinal limpo (nenhuma condição ativa).")

    return None

# ---------------- Salvar candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    log(f"[{symbol}] Últimos candles salvos em {path}")

# ---------------- Monitor Símbolo (24/7 + reconexão infinita) ----------------
async def monitor_symbol(symbol: str):
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                send_telegram(f"🔌 [{symbol}] Conectado ao WebSocket.", symbol)
                log(f"[{symbol}] WebSocket conectado.")

                # Autorizar (envia token e aguarda resposta)
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth_raw = await ws.recv()
                try:
                    auth = json.loads(auth_raw)
                except Exception:
                    log(f"[{symbol}] Não foi possível parsear authorize response: {auth_raw}", "error")
                    raise Exception("Authorize parse error")

                # Tratar possíveis formatos de resposta:
                if "error" in auth:
                    log(f"[{symbol}] Resposta de authorize contém error: {auth}", "error")
                    raise Exception("Authorize error from Deriv")
                if "authorize" not in auth:
                    # log completo para debugging remoto
                    log(f"[{symbol}] Falha na autorização (payload inesperado): {auth}", "error")
                    raise Exception("Falha na autorização Deriv (campo 'authorize' ausente).")

                # Autorization ok (registro)
                log(f"[{symbol}] Autorizado na Deriv: {auth.get('authorize')}")

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
                    # tenta novamente depois de breve pausa
                    await asyncio.sleep(5)
                    continue

                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)

                # Mensagem de download (se ainda não enviada)
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
                        # se ficar sem mensagens por muito tempo, força reconexão
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
    return "Bot ativo — Nova lógica: EMA20/50 + BB20 + RSI14 (cruzamento + fechamento)"

# ---------------- Execução principal ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    # IMPORTANT: use_reloader=False é crítico para evitar múltiplos processos e logs "sumindo"
    log(f"🌐 Iniciando Flask na porta {port} (no thread).")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

async def main():
    # roda Flask em thread separada (sem reloader)
    threading.Thread(target=run_flask, daemon=True).start()

    send_telegram("✅ Bot iniciado com NOVA LÓGICA (EMA20/EMA50 + BB20 + RSI14).")
    log("▶ Iniciando monitoramento paralelo por par (24/7, reconexão infinita)...")

    # cria tasks paralelas (uma por símbolo)
    tasks = [monitor_symbol(s) for s in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
