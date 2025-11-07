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
import time
from pathlib import Path

load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
APP_ID = os.getenv("DERIV_APP_ID", "1089")

# Lista de pares
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]
if not SYMBOLS:
    SYMBOLS = [
        "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
        "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
        "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
        "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
    ]

# Diretório para armazenar candles
DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Telegram ----------------
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram não configurado. Mensagem:", message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print(f"📨 Mensagem enviada ao Telegram: {message[:60]}...")
        else:
            print(f"❌ Falha ao enviar para Telegram (HTTP {response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- Gerar Sinal ----------------
def gerar_sinal(df: pd.DataFrame):
    ultima = df.iloc[-1]
    close = float(ultima['close'])
    rsi = float(ultima['rsi']) if not pd.isna(ultima['rsi']) else None
    bb_low = float(ultima['bb_lower']) if not pd.isna(ultima['bb_lower']) else None
    bb_up = float(ultima['bb_upper']) if not pd.isna(ultima['bb_upper']) else None

    if rsi is None or bb_low is None or bb_up is None:
        return None

    if close <= bb_low and rsi <= 30:
        return "COMPRA"
    if close >= bb_up and rsi >= 70:
        return "VENDA"

    return None

# ---------------- Salvamento de candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str, max_rows: int = 200):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df_to_save = df.tail(max_rows).copy()
    if 'epoch' in df_to_save.columns:
        try:
            df_to_save['timestamp_utc'] = pd.to_datetime(df_to_save['epoch'], unit='s', utc=True)
        except Exception:
            pass
    df_to_save.to_csv(path, index=False)
    print(f"[{symbol}] Salvas últimas {len(df_to_save)} velas em {path}")

# ---------------- REST para candles ----------------
def get_candles_rest(symbol: str, granularity: int):
    url = f"https://api.deriv.com/api/v3/ticks_history"
    params = {
        "ticks_history": symbol,
        "count": 500,
        "end": "latest",
        "granularity": granularity,
        "style": "candles",
        "app_id": APP_ID
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if "history" in data and "candles" in data["history"]:
            candles = data["history"]["candles"]
            return pd.DataFrame(candles)
    except Exception as e:
        print(f"[{symbol}] Erro ao obter candles REST: {e}")
    return None

# ---------------- Monitoramento WebSocket ----------------
async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=15) as ws:
                send_telegram(f"✅ Conexão ativa com WebSocket da Deriv para {symbol}")
                print(f"[{symbol}] WebSocket conectado.")

                while True:
                    # Pega candles via REST (seguro)
                    df = get_candles_rest(symbol, CANDLE_INTERVAL * 60)
                    if df is not None and not df.empty:
                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)
                        sinal = gerar_sinal(df)
                        if sinal:
                            send_telegram(f"💹 Sinal {sinal} detectado para {symbol} ({CANDLE_INTERVAL} min)")
                    else:
                        send_telegram(f"⚠️ Falha ao obter candles REST para {symbol}")

                    await asyncio.sleep(CANDLE_INTERVAL * 60)

        except Exception as e:
            send_telegram(f"⚠️ Erro ou desconexão no WebSocket para {symbol}: {e}")
            print(f"[{symbol}] Exceção no WS {symbol}: {e}")
            await asyncio.sleep(10)

# ---------------- Flask Web Service ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- Execução em blocos ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    print("Iniciando monitoramento dos símbolos:", SYMBOLS)

    grupo_tamanho = 3
    while True:
        for i in range(0, len(SYMBOLS), grupo_tamanho):
            subset = SYMBOLS[i:i + grupo_tamanho]
            print(f"\n🚀 Iniciando grupo: {subset}")
            tasks = [asyncio.create_task(monitor_symbol(sym, start_delay=idx * 2)) for idx, sym in enumerate(subset)]
            await asyncio.gather(*tasks)
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
