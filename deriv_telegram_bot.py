import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Configurações
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CANDLE_INTERVAL = 5  # minutos
SYMBOLS = ["frxEURUSD", "frxEURJPY", "frxUSDCHF"]
APP_ID = 1089  # App ID da Deriv

# Função para enviar mensagens para Telegram
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}")

# Indicadores
def calcular_indicadores(df):
    df['ema_curta'] = EMAIndicator(df['close'], window=5).ema_indicator()
    df['ema_media'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema_longa'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_medio'] = bb.bollinger_mavg()
    df['bb_sup'] = bb.bollinger_hband()
    df['bb_inf'] = bb.bollinger_lband()
    return df

# Gerar sinal
def gerar_sinal(df):
    ultima = df.iloc[-1]
    if (
        ultima['close'] > ultima['ema_curta'] > ultima['ema_media'] > ultima['ema_longa']
        and ultima['rsi'] > 50
        and ultima['close'] > ultima['bb_medio']
    ):
        return "COMPRA"
    elif (
        ultima['close'] < ultima['ema_curta'] < ultima['ema_media'] < ultima['ema_longa']
        and ultima['rsi'] < 50
        and ultima['close'] < ultima['bb_medio']
    ):
        return "VENDA"
    else:
        return None

# Monitoramento WebSocket
async def monitor_symbol(symbol):
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

    while True:  # Loop para reconexão automática
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                send_telegram(f"✅ Conexão ativa com WebSocket da Deriv para {symbol}!")
                print(f"🚀 Conexão ativa com WebSocket da Deriv para {symbol}")

                # Solicitar candles de 5 min
                req = {
                    "ticks_history": symbol,
                    "count": 100,
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req))

                primeira_vez = True

                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(response)

                        if "history" in data:
                            candles = data["history"]["candles"]
                            df = pd.DataFrame(candles)
                            df['close'] = df['close'].astype(float)
                            df['open'] = df['open'].astype(float)
                            df = calcular_indicadores(df)

                            if primeira_vez:
                                send_telegram(f"📡 Primeira resposta de candles recebida do WebSocket ({symbol})!")
                                primeira_vez = False
                                print(f"📡 Primeira resposta de candles recebida ({symbol})")

                            sinal = gerar_sinal(df)
                            if sinal:
                                send_telegram(f"💹 Sinal {sinal} detectado para {symbol} (vela {CANDLE_INTERVAL} min)")

                        await asyncio.sleep(CANDLE_INTERVAL * 60)

                    except asyncio.TimeoutError:
                        send_telegram(f"⚠️ Timeout: não foi possível receber dados do WebSocket para {symbol}")
                        print(f"⚠️ Timeout para {symbol}")
                        break  # Sai para reconectar

        except Exception as e:
            send_telegram(f"❌ Erro no WebSocket para {symbol}: {e}")
            print(f"❌ Erro no WebSocket {symbol}: {e}")

        send_telegram(f"🔄 Tentando reconectar WebSocket para {symbol} em 5 segundos...")
        await asyncio.sleep(5)

# Função principal
async def main():
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    for symbol in SYMBOLS:
        send_telegram(f"📊 Começando a monitorar **{symbol}**.")

    tasks = [monitor_symbol(symbol) for symbol in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
