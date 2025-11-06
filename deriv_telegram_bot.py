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

load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CANDLE_INTERVAL = 5  # minutos

# Lê os pares da variável de ambiente SYMBOLS
SYMBOLS = os.getenv("SYMBOLS", "").split(",")

if not SYMBOLS or SYMBOLS == [""]:
    SYMBOLS = [
        "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
        "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
        "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
        "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
    ]

# ---------------- Função Telegram ----------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}")

# ---------------- Indicadores ----------------
def calcular_indicadores(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df

# ---------------- Gerar Sinal ----------------
def gerar_sinal(df):
    ultima = df.iloc[-1]

    condicao_compra = ultima['close'] <= ultima['bb_low'] and ultima['rsi'] < 30
    condicao_venda = ultima['close'] >= ultima['bb_high'] and ultima['rsi'] > 70

    if condicao_compra:
        return "COMPRA"
    elif condicao_venda:
        return "VENDA"
    else:
        return None

# ---------------- Monitoramento WebSocket ----------------
async def monitor_symbol(symbol):
    url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                send_telegram(f"✅ Conexão ativa com WebSocket da Deriv para {symbol}!")
                print(f"🚀 Conexão ativa com WebSocket da Deriv para {symbol}")

                req = {
                    "ticks_history": symbol,
                    "count": 100,
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req))

                first_response = True
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)

                        # --- 🔧 Correção da Opção 1: detecta candles em ambos formatos ---
                        if "candles" in data or "candles" in data.get("history", {}):
                            candles = (
                                data["candles"]
                                if "candles" in data
                                else data["history"]["candles"]
                            )

                            df = pd.DataFrame(candles)
                            df['close'] = df['close'].astype(float)
                            df['open'] = df['open'].astype(float)
                            df = calcular_indicadores(df)

                            if first_response:
                                send_telegram(f"📡 Primeira resposta de candles recebida do WebSocket ({symbol})!")
                                first_response = False

                            sinal = gerar_sinal(df)
                            if sinal:
                                send_telegram(f"💹 Sinal {sinal} detectado para {symbol} (vela {CANDLE_INTERVAL} min)")

                    except Exception as e:
                        send_telegram(f"❌ Erro no WebSocket para {symbol}: {e}")
                        break

                    await asyncio.sleep(CANDLE_INTERVAL * 60)

        except Exception as e:
            send_telegram(f"🔄 Tentando reconectar WebSocket para {symbol} após erro: {e}")
            print(f"🔄 Reconectando {symbol} depois de erro: {e}")
            await asyncio.sleep(5)

# ---------------- Flask Web Service ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- Função Principal ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()

    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    for symbol in SYMBOLS:
        send_telegram(f"📊 Começando a monitorar **{symbol}**.")

    tasks = [monitor_symbol(symbol) for symbol in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
