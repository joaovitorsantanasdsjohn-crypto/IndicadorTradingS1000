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
    "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
    "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# Controle de mensagens
last_notify_time = {}
sent_download_message = {s: False for s in SYMBOLS}
candles_data = {}

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()
    if symbol:
        last_time = last_notify_time.get(symbol, 0)
        if now - last_time < 3:
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram não configurado. Mensagem:", message)
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")

# ---------------- Controle de horário Forex ----------------
def is_forex_open() -> bool:
    now = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour = now.hour
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
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema55'] = EMAIndicator(df['close'], window=55).ema_indicator()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

def gerar_sinal(df: pd.DataFrame):
    ultima = df.iloc[-1]
    ema9, ema21, ema55 = ultima['ema9'], ultima['ema21'], ultima['ema55']
    rsi, close = ultima['rsi'], ultima['close']
    bb_upper, bb_lower = ultima['bb_upper'], ultima['bb_lower']

    if pd.isna(ema9) or pd.isna(ema21) or pd.isna(ema55) or pd.isna(rsi):
        return None
    if ema9 > ema21 > ema55 and 30 <= rsi <= 45 and close <= bb_lower:
        return "COMPRA"
    elif ema9 < ema21 < ema55 and 55 <= rsi <= 70 and close >= bb_upper:
        return "VENDA"
    return None

def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)

# ---------------- WebSocket único ----------------
async def monitor_all_symbols():
    async with websockets.connect(WS_URL) as ws:
        # Autorizar
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        auth_resp = json.loads(await ws.recv())
        if not auth_resp.get("authorize"):
            print("❌ Falha na autorização:", auth_resp)
            return

        print("🔐 Autorizado na Deriv.")
        send_telegram("🔌 Conexão WebSocket estabelecida com sucesso e autorizada na Deriv! ✅")

        # Histórico inicial
        for symbol in SYMBOLS:
            if not is_forex_open():
                print("🌙 Forex fechado, aguardando...")
                await asyncio.sleep(600)
                continue

            req = {"ticks_history": symbol, "count": 200, "end": "latest",
                   "granularity": CANDLE_INTERVAL * 60, "style": "candles"}
            await ws.send(json.dumps(req))
            data = json.loads(await ws.recv())

            if "candles" in data:
                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)
                candles_data[symbol] = df
                save_last_candles(df, symbol)

                if not sent_download_message[symbol]:
                    send_telegram(f"📥 [{symbol}] Download de velas executado com sucesso ({len(df)} candles).", symbol)
                    sent_download_message[symbol] = True
            await asyncio.sleep(0.5)

        # Assinar todos os símbolos
        for symbol in SYMBOLS:
            req = {"ticks_history": symbol, "style": "candles",
                   "granularity": CANDLE_INTERVAL * 60, "end": "latest", "subscribe": 1}
            await ws.send(json.dumps(req))
            await asyncio.sleep(0.3)
        print("✅ Todos os pares assinados para candles ao vivo.")

        # Loop de recebimento
        while True:
            if not is_forex_open():
                print("🌙 Mercado fechado, pausando...")
                await asyncio.sleep(600)
                continue

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=180)
                data = json.loads(msg)
                candle = data.get("candle")
                symbol = data.get("echo_req", {}).get("ticks_history")

                if not candle or not symbol:
                    continue

                if symbol not in candles_data:
                    continue

                df = candles_data[symbol]
                last_epoch = df.iloc[-1]['epoch']
                if candle['epoch'] != last_epoch:
                    df.loc[len(df)] = candle
                    df = calcular_indicadores(df)
                    candles_data[symbol] = df

                    sinal = gerar_sinal(df)
                    ultima = df.iloc[-1]
                    print(f"[{symbol}] 🧩 Close={ultima['close']:.5f}, RSI={ultima['rsi']:.2f}, "
                          f"EMA9={ultima['ema9']:.5f}, EMA21={ultima['ema21']:.5f}, EMA55={ultima['ema55']:.5f}, "
                          f"Sinal={sinal or 'Nenhum'}")

                    if sinal:
                        send_telegram(f"💹 [{symbol}] *Sinal {sinal}* detectado!", symbol)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"⚠️ Erro WebSocket: {e}")
                await asyncio.sleep(5)

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅ (Diagnóstico)"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- Execução principal ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise! 🔍 (conta REAL)")
    print("▶ Iniciando monitoramento único para todos os pares...")
    await monitor_all_symbols()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Encerrando...")
