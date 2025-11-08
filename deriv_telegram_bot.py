# deriv_telegram_bot.py
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time

# ---------------- CONFIGURAÇÃO INICIAL ----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")  # TOKEN API DERIV
APP_ID = os.getenv("DERIV_APP_ID", "1089")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos

# Lista de pares de moedas
SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
    "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

# Diretório para salvar candles
DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# Limite de conexões simultâneas
MAX_CONCURRENT_WS = 3
ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)

# Controle de mensagens para evitar flood
last_connection_notify = {}  # armazena última notificação de conexão
last_signal_sent = {}        # armazena última notificação de sinal

# ---------------- FUNÇÃO TELEGRAM ----------------
def send_telegram(message: str):
    """Envia mensagem no Telegram."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram não configurado. Mensagem:", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
        print(f"📨 Telegram: {message[:100]}")
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")

# ---------------- INDICADORES ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula RSI e Bandas de Bollinger."""
    if len(df) < 20:
        return df
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- GERAR SINAL ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """Gera sinal apenas se RSI e Bollinger concordarem."""
    if len(df) < 20:
        return None
    ultima = df.iloc[-1]
    close = float(ultima['close'])
    rsi = ultima.get('rsi', np.nan)
    bb_low = ultima.get('bb_lower', np.nan)
    bb_up = ultima.get('bb_upper', np.nan)

    if np.isnan(rsi) or np.isnan(bb_low) or np.isnan(bb_up):
        return None

    # Sinal apenas quando RSI e Bollinger concordam
    if close <= bb_low and rsi <= 30:
        return "COMPRA"
    if close >= bb_up and rsi >= 70:
        return "VENDA"
    return None

# ---------------- GERAR CANDLE ----------------
def add_tick_to_candles(candles, tick, interval_sec):
    """Atualiza a lista de candles com o novo tick."""
    epoch = int(tick["epoch"])
    price = float(tick["quote"])
    candle_time = epoch - (epoch % interval_sec)

    if not candles or candles[-1]["epoch"] != candle_time:
        candles.append({"epoch": candle_time, "open": price, "high": price, "low": price, "close": price})
    else:
        c = candles[-1]
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price

    return candles[-200:]  # mantém as últimas 200 velas

# ---------------- SALVAR CANDLES ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    """Salva as velas localmente."""
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.to_csv(path, index=False)
    print(f"[{symbol}] {len(df)} velas salvas em {path}")

# ---------------- WEBSOCKET ----------------
async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    """Mantém conexão com WebSocket da Deriv para um par."""
    await asyncio.sleep(start_delay)
    interval_sec = CANDLE_INTERVAL * 60
    candles = []
    retry_delay = 5

    while True:
        async with ws_semaphore:
            try:
                url = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    # Autoriza token
                    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                    auth_resp = json.loads(await ws.recv())
                    if "error" in auth_resp:
                        send_telegram(f"❌ Erro de autorização: {auth_resp['error']['message']}")
                        await asyncio.sleep(30)
                        continue

                    # Subscreve ticks
                    await ws.send(json.dumps({"ticks_subscribe": symbol}))

                    now = time.time()
                    # Só envia mensagem de conexão se for primeira vez ou passou 10 minutos
                    if symbol not in last_connection_notify or now - last_connection_notify[symbol] > 600:
                        send_telegram(f"✅ Conexão WebSocket ativa para {symbol}")
                        last_connection_notify[symbol] = now

                    print(f"[{symbol}] WebSocket ativo.")
                    retry_delay = 5

                    # Loop principal de recepção
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if "tick" not in data:
                            continue

                        tick = data["tick"]
                        candles = add_tick_to_candles(candles, tick, interval_sec)
                        df = pd.DataFrame(candles)
                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        sinal = gerar_sinal(df, symbol)
                        if sinal:
                            now = time.time()
                            if symbol not in last_signal_sent or now - last_signal_sent[symbol] > 300:
                                hora = datetime.now(timezone.utc).strftime("%H:%M:%S")
                                send_telegram(f"💹 {symbol} — *{sinal}* detectado às {hora} UTC")
                                last_signal_sent[symbol] = now

            except Exception as e:
                print(f"⚠️ {symbol} desconectado: {e}")
                now = time.time()
                # só envia mensagem de reconexão se passou 10 min da última
                if symbol not in last_connection_notify or now - last_connection_notify[symbol] > 600:
                    send_telegram(f"⚠️ Reconectando {symbol} em {retry_delay}s...")
                    last_connection_notify[symbol] = now
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 120)

# ---------------- FLASK KEEP-ALIVE ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- MAIN ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado no Render e pronto para análise!")
    print("Monitorando pares:", SYMBOLS)

    tasks = []
    for i, sym in enumerate(SYMBOLS):
        delay = (i // MAX_CONCURRENT_WS) * 10  # espaça as conexões
        tasks.append(asyncio.create_task(monitor_symbol(sym, start_delay=delay)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
