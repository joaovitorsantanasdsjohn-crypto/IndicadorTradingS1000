# deriv_telegram_bot.py
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

# Controle de mensagens / estado de sinais
last_notify_time = {}
sent_download_message = {s: False for s in SYMBOLS}
last_signal_state = {s: None for s in SYMBOLS}  # None / "COMPRA"/"VENDA"
last_signal_candle = {s: None for s in SYMBOLS}  # armazena epoch do candle que gerou o último sinal

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

# ---------------- Controle de horário Forex (mantido) ----------------
def is_forex_open() -> bool:
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=segunda ... 6=domingo
    hour = now.hour
    # regras aproximadas (manter como no código anterior)
    if weekday == 6 and hour < 22:
        return False
    if weekday == 4 and hour >= 21:
        return False
    if weekday == 5:
        return False
    return True

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('epoch').reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    # calcula indicadores
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema55'] = EMAIndicator(df['close'], window=55).ema_indicator()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- Geração de sinal (afrouxada, Opção A) ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """
    Opção A (modo escolhido): envia 1 sinal quando condição satisfeita e só envia
    novamente do mesmo tipo depois que a condição for perdida e reaparecer.
    Critérios afrouxados para maior frequência (aprox. 1 sinal/hora por par).
    """
    ultima = df.iloc[-1]
    ema9 = ultima.get('ema9')
    ema21 = ultima.get('ema21')
    ema55 = ultima.get('ema55')
    rsi = ultima.get('rsi')
    close = float(ultima.get('close'))
    bb_upper = ultima.get('bb_upper')
    bb_lower = ultima.get('bb_lower')
    bb_mavg = ultima.get('bb_mavg')
    epoch = int(ultima.get('epoch'))

    log(f"🧮 [{symbol}] Indicadores calculados: RSI={rsi:.2f} | EMA9={ema9:.5f} | EMA21={ema21:.5f} | EMA55={ema55:.5f}")

    # logs detalhados para ajudar debugging
    log(f"   Bollinger: lower={bb_lower:.5f} | mavg={bb_mavg:.5f} | upper={bb_upper:.5f} | close={close:.5f}")

    # Validação de NaNs
    if pd.isna(ema9) or pd.isna(ema21) or pd.isna(ema55) or pd.isna(rsi):
        log(f"⚠️ [{symbol}] Indicadores incompletos — aguardando mais dados...")
        return None

    # Ajuste de "afrouxamento":
    buy_threshold = bb_lower + 0.4 * (bb_mavg - bb_lower)
    sell_threshold = bb_upper - 0.4 * (bb_upper - bb_mavg)

    # Decisões:
    buy_cond = (ema9 > ema21) and (35 <= rsi <= 55) and (close <= buy_threshold)
    sell_cond = (ema9 < ema21) and (45 <= rsi <= 65) and (close >= sell_threshold)

    log(f"   Avaliação cond: buy_cond={buy_cond} | sell_cond={sell_cond} | buy_thr={buy_threshold:.5f} | sell_thr={sell_threshold:.5f}")

    # Opção A: evitar repetir sinal até condição limpar
    current_state = last_signal_state.get(symbol)

    if buy_cond:
        if current_state == "COMPRA" and last_signal_candle.get(symbol) == epoch:
            log(f"   [{symbol}] Sinal COMPRA já enviado para este candle (skip).")
            return None
        if current_state != "COMPRA":
            last_signal_state[symbol] = "COMPRA"
            last_signal_candle[symbol] = epoch
            log(f"✅ [{symbol}] Condição de COMPRA atendida (enviando sinal).")
            return "COMPRA"
        else:
            log(f"   [{symbol}] COMPRA já ativa, aguardando limpeza da condição.")
            return None

    if sell_cond:
        if current_state == "VENDA" and last_signal_candle.get(symbol) == epoch:
            log(f"   [{symbol}] Sinal VENDA já enviado para este candle (skip).")
            return None
        if current_state != "VENDA":
            last_signal_state[symbol] = "VENDA"
            last_signal_candle[symbol] = epoch
            log(f"✅ [{symbol}] Condição de VENDA atendida (enviando sinal).")
            return "VENDA"
        else:
            log(f"   [{symbol}] VENDA já ativa, aguardando limpeza da condição.")
            return None

    # Se nenhuma condição ativa, limpa estado
    if not buy_cond and not sell_cond:
        if last_signal_state.get(symbol) is not None:
            log(f"🔄 [{symbol}] Condição limpa — sinal anterior ({last_signal_state[symbol]}) removido.")
        last_signal_state[symbol] = None
        last_signal_candle[symbol] = None

    log(f"🚫 [{symbol}] Nenhum sinal — condições não atendidas.")
    return None

# ---------------- Salvar candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    log(f"💾 [{symbol}] Últimos candles salvos em {path}")

# ---------------- Monitor por símbolo ----------------
async def monitor_symbol(symbol: str):
    reconnect_count = 0
    while True:
        try:
            # respeita janela de mercado
            if not is_forex_open():
                log(f"🌙 Mercado Forex fechado. Pausando monitoramento de {symbol} por 10 minutos.")
                await asyncio.sleep(600)
                continue

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
                    await asyncio.sleep(10)
                    continue
                log(f"🔐 [{symbol}] Autorizado na Deriv.")

                # Histórico inicial
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

                # FORÇAR cálculo inicial
                try:
                    initial_signal = gerar_sinal(df, symbol)
                    if initial_signal:
                        send_telegram(f"📥 [{symbol}] Download de velas executado com sucesso ({len(df)} candles). Sinal inicial: *{initial_signal}*.", symbol)
                        sent_download_message[symbol] = True
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

                # Loop vivo
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=180)
                        data = json.loads(msg)

                        candle = data.get("candle")
                        if not candle:
                            continue

                        candle_time = datetime.utcfromtimestamp(candle['epoch']).strftime('%Y-%m-%d %H:%M:%S')
                        log(f"📊 [{symbol}] Novo candle recebido às {candle_time} UTC | close={candle['close']}")

                        ultimo_candle_time = time.time()

                        # atualiza df e indicadores
                        if df.empty or df.iloc[-1]['epoch'] != candle['epoch']:
                            df.loc[len(df)] = candle
                            df = calcular_indicadores(df)
                            save_last_candles(df, symbol)

                            sinal = gerar_sinal(df, symbol)
                            if sinal:
                                # ------------------------------
                                # NOVA FORMATAÇÃO DO SINAL (Modelo 1)
                                # ------------------------------
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
                        if time.time() - ultimo_candle_time > 300:
                            log(f"⚠️ [{symbol}] Nenhum candle há 5 minutos — forçando reconexão.", "warning")
                            raise Exception("Reconexão forçada por inatividade")
                        else:
                            log(f"⏱ [{symbol}] Aguardando novo candle...", "info")
                            continue

        except Exception as e:
            log(f"⚠️ [{symbol}] Erro WebSocket / loop: {e}", "error")
            wait = random.uniform(3, 8)
            log(f"⏳ [{symbol}] Aguardando {wait:.1f}s antes de tentar reconectar...", "info")
            await asyncio.sleep(wait)

# ---------------- Flask (diagnóstico) ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅ (versão estável com reconexão e candles ao vivo)"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    log(f"🌐 Flask rodando na porta {port}")
    app.run(host="0.0.0.0", port=port)

# ---------------- Execução principal ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise! 🔍 (conta REAL)")
    log("▶ Iniciando monitoramento paralelo por par (modo estável com reconexão automática)...")

    tasks = [monitor_symbol(symbol) for symbol in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
