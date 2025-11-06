import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import threading
from flask import Flask
import pathlib
import time
import logging

# ---------------- init ----------------
load_dotenv()

# Logging (aparece no terminal / logs do Render)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Deriv App ID: usa DERIV_APP_ID se configurado, senão usa 1089 (público)
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
GRANULARITY = CANDLE_INTERVAL * 60  # segundos
REQUEST_COUNT = int(os.getenv("REQUEST_COUNT", "100"))  # quantas candles pedir (o código guarda só as últimas 50)
STORE_LAST_N = int(os.getenv("STORE_LAST_N", "50"))  # você pediu: últimas 50 velas

# Delay progressivo entre abrir conexões (segundos)
START_DELAY_BETWEEN_SYMBOLS = int(os.getenv("START_DELAY_BETWEEN_SYMBOLS", "5"))

# Backoff de reconexão (segundos): 5s,15s,30s,60s repetindo
RECONNECT_BACKOFFS = [5, 15, 30, 60]

# Lista de SYMBOLS lida de env ou default (20 pares)
SYMBOLS = os.getenv("SYMBOLS", "").split(",")
if not SYMBOLS or SYMBOLS == [""]:
    SYMBOLS = [
        "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
        "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
        "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
        "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
    ]

# Pasta para salvar candles
CANDLES_DIR = pathlib.Path("candles")
CANDLES_DIR.mkdir(exist_ok=True)

# ---------------- Helper: Telegram ----------------
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("TELEGRAM_TOKEN/CHAT_ID não configurados — pulando envio Telegram.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            logging.warning(f"Telegram returned {res.status_code}: {res.text}")
    except Exception as e:
        logging.exception(f"Erro ao enviar Telegram: {e}")

# ---------------- Indicadores (RSI + Bollinger) ----------------
def calcular_indicadores(df):
    # Garante ordenação por epoch ou time
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    # converte para float caso sejam strings
    df['close'] = df['close'].astype(float)
    # RSI
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    # Bollinger Bands
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_medio'] = bb.bollinger_mavg()
    df['bb_sup'] = bb.bollinger_hband()
    df['bb_inf'] = bb.bollinger_lband()
    return df

# ---------------- Armazenamento / leitura candles ----------------
def path_for_symbol(symbol):
    return CANDLES_DIR / f"{symbol}.csv"

def load_stored_candles(symbol):
    p = path_for_symbol(symbol)
    if p.exists():
        try:
            df = pd.read_csv(p)
            return df
        except Exception:
            logging.exception(f"Erro lendo arquivo candles para {symbol}. Removendo arquivo corrompido.")
            try:
                p.unlink()
            except Exception:
                pass
    return pd.DataFrame()

def store_last_candles(symbol, df_new):
    """
    Mantém apenas as últimas STORE_LAST_N velas em CSV (overwrite).
    df_new deve ter colunas compatíveis ('epoch','open','high','low','close')
    """
    df_new = df_new.copy().reset_index(drop=True)
    # se já existe histórico, podemos combinar (mas pedimos para manter só últimas N)
    path = path_for_symbol(symbol)
    if path.exists():
        try:
            df_old = pd.read_csv(path)
            combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['epoch']).sort_values('epoch')
        except Exception:
            logging.exception(f"Erro ao combinar candles antigos para {symbol}, usando apenas novos.")
            combined = df_new
    else:
        combined = df_new
    # manter últimas STORE_LAST_N
    combined = combined.sort_values('epoch').reset_index(drop=True)
    if len(combined) > STORE_LAST_N:
        combined = combined.iloc[-STORE_LAST_N:].reset_index(drop=True)
    try:
        combined.to_csv(path, index=False)
    except Exception:
        logging.exception(f"Erro ao salvar candles para {symbol} em {path}")

# ---------------- Gerar sinal (RSI + Bollinger — ambos devem concordar) ----------------
def gerar_sinal(df):
    """
    Critério (mais criterioso):
      - COMPRA: RSI > 55 AND close > bb_medio
      - VENDA:  RSI < 45 AND close < bb_medio
    Ambos os indicadores precisam concordar.
    """
    if df.empty or len(df) < 1:
        return None
    ultima = df.iloc[-1]
    try:
        close = float(ultima['close'])
        rsi = float(ultima['rsi'])
        bb_medio = float(ultima['bb_medio'])
    except Exception:
        return None

    if (rsi > 55) and (close > bb_medio):
        return "COMPRA"
    elif (rsi < 45) and (close < bb_medio):
        return "VENDA"
    else:
        return None

# ---------------- Monitoramento WebSocket com confirmações e reconexão ----------------
async def monitor_symbol(symbol, start_delay=0):
    await asyncio.sleep(start_delay)  # delay progressivo para não abrir tudo ao mesmo tempo
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"
    backoff_index = 0

    while True:
        try:
            logging.info(f"Tentando conectar {symbol} -> {url}")
            async with websockets.connect(url, ping_interval=20, close_timeout=5) as ws:
                # reset backoff on successful open
                backoff_index = 0
                msg = f"✅ Conexão ativa com WebSocket da Deriv para {symbol}!"
                send_telegram(msg)
                logging.info(msg)

                # pedido de candles (ticks_history)
                req = {
                    "ticks_history": symbol,
                    "count": REQUEST_COUNT,
                    "granularity": GRANULARITY,
                    "style": "candles"
                }
                await ws.send(json.dumps(req))
                logging.info(f"Solicitado histórico ({REQUEST_COUNT}) para {symbol} (granularity={GRANULARITY})")

                first_response = True

                while True:
                    try:
                        # espera por mensagem (sem timeout abusivo; websockets.recv bloqueará)
                        response = await asyncio.wait_for(ws.recv(), timeout=60)
                    except asyncio.TimeoutError:
                        # trata timeout de recv como perda de dados; tenta reabrir
                        msg = f"⚠️ Timeout sem receber dados do WebSocket para {symbol}"
                        send_telegram(msg)
                        logging.warning(msg)
                        break

                    try:
                        data = json.loads(response)
                    except Exception:
                        logging.exception("Resposta inválida JSON recebida.")
                        continue

                    # se recebemos 'history' (resposta ao ticks_history)
                    if "history" in data and "candles" in data["history"]:
                        candles = data["history"]["candles"]
                        if not candles:
                            logging.warning(f"Resposta de candles vazia para {symbol}")
                        else:
                            df = pd.DataFrame(candles)
                            # padroniza colunas esperadas (Deriv: epoch, open, high, low, close)
                            # garantir epoch
                            if 'epoch' not in df.columns and 'time' in df.columns:
                                df.rename(columns={'time': 'epoch'}, inplace=True)
                            # convert types
                            for c in ['close', 'open', 'high', 'low', 'epoch']:
                                if c in df.columns:
                                    df[c] = pd.to_numeric(df[c], errors='coerce')

                            # salvar últimas STORE_LAST_N velas no CSV
                            store_last_candles(symbol, df)

                            # carregar histórico salvo (últimas N) e calcular indicadores com base nele
                            df_hist = load_stored_candles(symbol)
                            if df_hist.empty:
                                df_hist = df.copy()
                            # Ajuste: garantir colunas corretas e tipos
                            df_hist = df_hist[['epoch','open','high','low','close']].dropna().reset_index(drop=True)
                            df_hist['close'] = df_hist['close'].astype(float)

                            # calcular indicadores (usa as últimas candles armazenadas)
                            df_ind = calcular_indicadores(df_hist)

                            # confirmação concreta de que velas foram salvas/recebidas:
                            last_close = df_ind.iloc[-1]['close'] if not df_ind.empty else None
                            msg = f"📡 Candles recebidos e salvos para *{symbol}* — fechamento: `{last_close}` (últimas {len(df_ind)} velas armazenadas)."
                            send_telegram(msg)
                            logging.info(msg)

                            # gerar sinal com base no histórico atualizado
                            sinal = gerar_sinal(df_ind)
                            if sinal:
                                send_telegram(f"💹 Sinal *{sinal}* detectado para *{symbol}* (vela {CANDLE_INTERVAL} min).")
                                logging.info(f"Sinal {sinal} para {symbol}")

                            # primeira resposta tratada
                            if first_response:
                                first_response = False

                    # continua escutando mensagens — Deriv pode enviar keepalives e outros eventos
                    # Aqui dormimos um pouco para não sobrecarregar o loop
                    await asyncio.sleep(0.1)

        except websockets.exceptions.InvalidStatusCode as e:
            logging.exception(f"InvalidStatusCode ao conectar {symbol}: {e}")
            send_telegram(f"❌ Erro ao conectar WebSocket para {symbol}: {e}")
        except Exception as e:
            logging.exception(f"Erro genérico WebSocket {symbol}: {e}")
            send_telegram(f"🔄 Tentando reconectar WebSocket para {symbol} após erro: {e}")

        # backoff antes de nova tentativa
        wait = RECONNECT_BACKOFFS[min(backoff_index, len(RECONNECT_BACKOFFS)-1)]
        logging.info(f"Esperando {wait}s antes de reconectar {symbol}")
        await asyncio.sleep(wait)
        backoff_index = min(backoff_index + 1, len(RECONNECT_BACKOFFS)-1)

# ---------------- Flask Web Service (para Render) ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    # Flask dev server (ok para Render free web service, já que usamos thread)
    app.run(host="0.0.0.0", port=port)

# ---------------- Função Principal ----------------
async def main():
    # inicia Flask em thread separada para que Render detecte porta
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    logging.info("Bot iniciado — iniciando monitoramento dos símbolos")

    # cria tasks com delay progressivo entre cada para reduzir picos
    tasks = []
    for i, symbol in enumerate(SYMBOLS):
        send_telegram(f"📊 Começando a monitorar *{symbol}*.")
        delay = i * START_DELAY_BETWEEN_SYMBOLS
        tasks.append(asyncio.create_task(monitor_symbol(symbol, start_delay=delay)))

    # aguarda todas as tasks (elas rodam indefinidamente)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Encerrando por KeyboardInterrupt")
