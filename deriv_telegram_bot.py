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
from pathlib import Path
from flask import Flask
import threading

# ---------------- CONFIGURAÇÃO ----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID", "1089")  # app_id público
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "1"))  # minutos

# Lista completa de pares
SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
    "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

MAX_CONCURRENT_WS = 3
ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)

# ---------------- FLASK PARA UPTIME ----------------
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot Deriv ativo e rodando normalmente"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

threading.Thread(target=run_flask, daemon=True).start()

# ---------------- TELEGRAM ----------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram não configurado:", msg)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        print("📩 Enviado:", msg[:60])
    except Exception as e:
        print("Erro Telegram:", e)

# ---------------- INDICADORES ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    df = df.sort_values("epoch").reset_index(drop=True)
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    return df

def gerar_sinal(df: pd.DataFrame):
    if len(df) < 20:
        return None
    ultima = df.iloc[-1]
    close = ultima["close"]
    rsi = ultima["rsi"]
    bb_low = ultima["bb_lower"]
    bb_up = ultima["bb_upper"]

    if close <= bb_low and rsi <= 30:
        return "📈 *SINAL DE COMPRA* — RSI e Bandas concordam"
    if close >= bb_up and rsi >= 70:
        return "📉 *SINAL DE VENDA* — RSI e Bandas concordam"
    return None

# ---------------- CONSTRUÇÃO DAS VELAS ----------------
def add_tick_to_candles(candles, tick, interval_sec):
    epoch = tick["epoch"]
    price = float(tick["quote"])
    candle_time = epoch - (epoch % interval_sec)
    if not candles or candles[-1]["epoch"] != candle_time:
        candles.append({
            "epoch": candle_time,
            "open": price,
            "high": price,
            "low": price,
            "close": price
        })
    else:
        c = candles[-1]
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price
    return candles

def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(500).to_csv(path, index=False)
    print(f"[{symbol}] {len(df)} velas salvas.")

# ---------------- MONITORAMENTO PERSISTENTE ----------------
async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)
    interval_sec = CANDLE_INTERVAL * 60
    candles = []
    retry_delay = 5

    while True:
        try:
            url = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}&l=EN"
            async with ws_semaphore:
                async with websockets.connect(url) as ws:
                    # Autenticação com token
                    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                    auth_resp = json.loads(await ws.recv())
                    if "error" in auth_resp:
                        print(f"Erro ao autorizar {symbol}: {auth_resp['error']['message']}")
                        await asyncio.sleep(30)
                        continue

                    # Subscrição de ticks
                    await ws.send(json.dumps({"ticks_subscribe": symbol}))
                    send_telegram(f"✅ Conexão WebSocket ativa para {symbol}")

                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if "error" in data:
                            print(f"Erro no WebSocket ({symbol}):", data["error"]["message"])
                            break
                        if "tick" not in data:
                            continue

                        tick = data["tick"]
                        candles = add_tick_to_candles(candles, tick, interval_sec)

                        # Atualiza dataframe e calcula indicadores
                        df = pd.DataFrame(candles)
                        df = calcular_indicadores(df)

                        # Gera e envia sinal quando necessário
                        sinal = gerar_sinal(df)
                        if sinal:
                            hora = datetime.now(timezone.utc).strftime("%H:%M:%S")
                            msg = f"📊 *{symbol}* — {sinal}\nHora UTC: {hora}"
                            send_telegram(msg)

                        save_last_candles(df, symbol)

        except Exception as e:
            print(f"⚠️ Conexão perdida em {symbol}: {e}")
            send_telegram(f"⚠️ Reconectando {symbol} em {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 120)

# ---------------- INICIALIZAÇÃO ----------------
async def main():
    send_telegram("🤖 Bot Deriv iniciado com sucesso e pronto para análise dos 20 pares!")
    tasks = []
    for i, sym in enumerate(SYMBOLS):
        tasks.append(asyncio.create_task(monitor_symbol(sym, start_delay=i * 5)))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
