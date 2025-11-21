# deriv_telegram_bot.py
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

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações ----------------
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

# Controle de mensagens / estado de sinais
last_notify_time = {}
sent_download_message = {s: False for s in SYMBOLS}
last_signal_state = {s: None for s in SYMBOLS}  # None / "COMPRA"/"VENDA"
last_signal_candle = {s: None for s in SYMBOLS}  # epoch do candle que gerou o último sinal

# ---------------- Logging (para aparecer no Render) ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    # imprime também para garantir flush imediato (Render)
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
    now = time.time()
    if symbol:
        last_time = last_notify_time.get(symbol, 0)
        # evita flood telegram para o mesmo par (3 segundos)
        if now - last_time < 3:
            log(f"Telegram rate limit skipped for {symbol}", "warning")
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("⚠️ Telegram não configurado. Mensagem: " + message, "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
        log(f"Telegram enviado: {message}")
    except Exception as e:
        log(f"❌ Erro ao enviar Telegram: {e}", "error")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('epoch').reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    # RSI
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    # EMAs para tendência (usadas no Modelo B)
    df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    # MACD (fast=12 slow=26 signal=9 padrão)
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # Bollinger (mantido para info, embora não seja gatilho principal)
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- Geração de sinal (Lógica B - afrouxada equilibrada) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """
    Lógica B aplicada:
    - Gate de tendência: EMA20 vs EMA50
    - RSI flexível: compra RSI<40, venda RSI>60
    - MACD simples: macd > signal => bullish, macd < signal => bearish
    - Evitar reenvio do mesmo sinal até condição ser limpa
    Objetivo: aumentar frequência (meta ~15 sinais/dia) mantendo razoável assertividade.
    """
    ultima = df.iloc[-1]
    epoch = int(ultima.get('epoch'))
    close = float(ultima.get('close'))
    rsi = ultima.get('rsi')
    ema20 = ultima.get('ema20')
    ema50 = ultima.get('ema50')
    macd = ultima.get('macd')
    macd_signal = ultima.get('macd_signal')

    # logs para debugging
    try:
        log(f"🧮 [{symbol}] RSI={rsi:.2f} | EMA20={ema20:.5f} | EMA50={ema50:.5f} | MACD={macd:.6f} | Signal={macd_signal:.6f}")
    except Exception:
        log(f"🧮 [{symbol}] Indicadores calculados (alguns NaN possivelmente)")

    # validação
    if pd.isna(rsi) or pd.isna(ema20) or pd.isna(ema50) or pd.isna(macd) or pd.isna(macd_signal):
        log(f"⚠️ [{symbol}] Indicadores incompletos — aguardando mais dados...")
        return None

    # regras (afrouxadas e combinadas)
    trend_buy = ema20 > ema50
    trend_sell = ema20 < ema50

    macd_bull = macd > macd_signal
    macd_bear = macd < macd_signal

    buy_cond = trend_buy and macd_bull and (rsi <= 40)
    sell_cond = trend_sell and macd_bear and (rsi >= 60)

    # pequenos "relaxamentos" para garantir fluxo:
    # - se MACD muito forte (macd_diff grande) podemos aceitar RSI um pouco fora dos thresholds
    macd_diff = macd - macd_signal
    if not buy_cond and trend_buy and macd_diff > 0.0001 and rsi <= 45:
        # caso MACD forte e RSI até 45, permitir compra (aumenta frequência)
        buy_cond = True
        log(f"   [{symbol}] Relaxamento: MACD forte positivo abriu buy (macd_diff={macd_diff:.6f}, rsi={rsi:.2f})")
    if not sell_cond and trend_sell and macd_diff < -0.0001 and rsi >= 55:
        sell_cond = True
        log(f"   [{symbol}] Relaxamento: MACD forte negativo abriu sell (macd_diff={macd_diff:.6f}, rsi={rsi:.2f})")

    # evitar repetir sinal até condição limpar
    current_state = last_signal_state.get(symbol)

    if buy_cond:
        # evita reenviar no mesmo candle
        if current_state == "COMPRA" and last_signal_candle.get(symbol) == epoch:
            log(f"   [{symbol}] COMPRA já enviada neste candle (skip).")
            return None
        if current_state != "COMPRA":
            last_signal_state[symbol] = "COMPRA"
            last_signal_candle[symbol] = epoch
            log(f"✅ [{symbol}] Condição COMPRA atendida (enviando sinal). Close={close:.5f} RSI={rsi:.2f}")
            return "COMPRA"
        return None

    if sell_cond:
        if current_state == "VENDA" and last_signal_candle.get(symbol) == epoch:
            log(f"   [{symbol}] VENDA já enviada neste candle (skip).")
            return None
        if current_state != "VENDA":
            last_signal_state[symbol] = "VENDA"
            last_signal_candle[symbol] = epoch
            log(f"✅ [{symbol}] Condição VENDA atendida (enviando sinal). Close={close:.5f} RSI={rsi:.2f}")
            return "VENDA"
        return None

    # se nenhuma condição ativa, limpa estado (permite novos sinais quando reaparecer)
    if not buy_cond and not sell_cond:
        if last_signal_state.get(symbol) is not None:
            log(f"🔄 [{symbol}] Condição limpa — sinal anterior ({last_signal_state[symbol]}) removido.")
        last_signal_state[symbol] = None
        last_signal_candle[symbol] = None

    return None

# ---------------- Salvar candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    log(f"💾 [{symbol}] Últimos candles salvos em {path}")

# ---------------- Monitor por símbolo (conexão 24/7 + reconexão infinita) ----------------
async def monitor_symbol(symbol: str):
    reconnect_count = 0
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                reconnect_count += 1
                if reconnect_count > 1:
                    send_telegram(f"🔄 [{symbol}] Reconectado à Deriv (tentativa {reconnect_count}).", symbol)
                    log(f"🔄 [{symbol}] Reconectado à Deriv (tentativa {reconnect_count}).")
                else:
                    log(f"🔌 [{symbol}] Nova conexão WebSocket iniciada.")
                    send_telegram(f"✅ [{symbol}] Conexão WebSocket estabelecida com sucesso.", symbol)

                # Autorização
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth = json.loads(await ws.recv())
                if not auth.get("authorize"):
                    log(f"❌ Falha na autorização para {symbol}: {auth}", "error")
                    await asyncio.sleep(5)
                    continue
                log(f"🔐 [{symbol}] Autorizado na Deriv.")

                # Histórico inicial (200 candles)
                req_hist = {
                    "ticks_history": symbol,
                    "count": 200,
                    "end": "latest",
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req_hist))
                data = json.loads(await ws.recv())

                if "candles" not in data:
                    log(f"⚠️ [{symbol}] Nenhum dado de candle recebido no histórico inicial: {data}", "warning")
                    await asyncio.sleep(5)
                    continue

                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)

                # FORÇAR cálculo inicial e enviar a confirmação de download separada do sinal
                try:
                    initial_signal = gerar_sinal(df, symbol)
                    if initial_signal:
                        # Mensagem de download permanece — sinal é enviado separadamente abaixo se necessário
                        send_telegram(f"📥 [{symbol}] Download de velas executado com sucesso ({len(df)} candles).", symbol)
                        sent_download_message[symbol] = True
                        # Também enviar sinal inicial como mensagem separada:
                        # (mantemos sinal separado — formato padronizado)
                        arrow = "🟢" if initial_signal == "COMPRA" else "🔴"
                        close_price = float(df.iloc[-1]["close"])
                        utc_time = datetime.utcnow().strftime('%H:%M:%S')
                        mensagem_sinal = (
                            f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                            f"• Par: {symbol.replace('frx','')}\n"
                            f"• Direção: {arrow} *{initial_signal}*\n"
                            f"• Preço: {close_price:.5f}\n"
                            f"• Horário: {utc_time} UTC"
                        )
                        send_telegram(mensagem_sinal, symbol)
                    else:
                        send_telegram(f"📥 [{symbol}] Download de velas executado com sucesso ({len(df)} candles).", symbol)
                        sent_download_message[symbol] = True
                except Exception as e:
                    log(f"⚠️ [{symbol}] Erro ao avaliar sinal inicial: {e}", "error")

                # Assinar candles ao vivo
                sub_req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))
                log(f"✅ [{symbol}] Assinado para candles ao vivo.")

                ultimo_candle_time = time.time()

                # Loop de recebimento (24/7, reconecta se cair ou timeout)
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=180)
                        data = json.loads(msg)

                        # mensagens podem ser variadas — buscamos candle
                        candle = data.get("candle")
                        if not candle:
                            # pode vir ping/response com outros campos; ignorar
                            continue

                        # log do candle recebido
                        candle_time = datetime.utcfromtimestamp(candle['epoch']).strftime('%Y-%m-%d %H:%M:%S')
                        log(f"📊 [{symbol}] Novo candle recebido às {candle_time} UTC | close={candle['close']}")
                        ultimo_candle_time = time.time()

                        # atualiza df e indicadores se for novo candle
                        if df.empty or df.iloc[-1]['epoch'] != candle['epoch']:
                            df.loc[len(df)] = candle
                            df = calcular_indicadores(df)
                            save_last_candles(df, symbol)

                            # gera sinal e envia conforme lógica
                            sinal = gerar_sinal(df, symbol)
                            if sinal:
                                arrow = "🟢" if sinal == "COMPRA" else "🔴"
                                close_price = float(df.iloc[-1]["close"])
                                utc_time = datetime.utcnow().strftime('%H:%M:%S')
                                mensagem_sinal = (
                                    f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                    f"• Par: {symbol.replace('frx','')}\n"
                                    f"• Direção: {arrow} *{sinal}*\n"
                                    f"• Preço: {close_price:.5f}\n"
                                    f"• Horário: {utc_time} UTC"
                                )
                                send_telegram(mensagem_sinal, symbol)
                    except asyncio.TimeoutError:
                        # se ficar sem mensagens por muito tempo, força reconexão
                        if time.time() - ultimo_candle_time > 300:
                            log(f"⚠️ [{symbol}] Nenhum candle há 5 minutos — forçando reconexão.", "warning")
                            raise Exception("Reconexão forçada por inatividade")
                        else:
                            log(f"⏱ [{symbol}] Aguardando novo candle...", "info")
                            continue

        except Exception as e:
            # Reconexão infinita com backoff curto + jitter para render gratuito
            log(f"⚠️ [{symbol}] Erro WebSocket / loop: {e}", "error")
            wait = random.uniform(2, 8)
            log(f"⏳ [{symbol}] Aguardando {wait:.1f}s antes de tentar reconectar...", "info")
            await asyncio.sleep(wait)

# ---------------- Flask (diagnóstico) ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅ (versão Lógica B — 24/7, reconexão infinita, sinais aumentados)"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    log(f"🌐 Flask rodando na porta {port}")
    app.run(host="0.0.0.0", port=port)

# ---------------- Execução principal ----------------
async def main():
    # starta Flask em thread
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise! 🔍 (Lógica B ativa)")
    log("▶ Iniciando monitoramento paralelo por par (Lógica B — 24/7 com reconexão infinita)...")

    # cria uma task por símbolo (mantemos conexões paralelas; se preferir reduzir uso de conexões,
    # podemos alterar para um loop que abre uma única conexão e troca symbol.requests)
    tasks = [monitor_symbol(symbol) for symbol in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
