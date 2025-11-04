import asyncio
import json
import pandas as pd
import requests
import os
import threading
from datetime import datetime
import websockets
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from flask import Flask
from dotenv import load_dotenv

# ====== CONFIGURAÇÕES ======
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_APP_ID = os.getenv("DERIV_APP_ID")
SYMBOL = "frxEURUSD"  # Par que estamos analisando
TIMEFRAME = 5  # candles de 5 minutos

# ====== FUNÇÕES AUXILIARES ======
def enviar_telegram(msg):
    """Envia mensagens formatadas para o Telegram"""
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
        print(f"📤 Enviado para Telegram: {msg}")
    else:
        print("⚠️ Token ou Chat ID não configurados.")

async def conectar_deriv():
    """Conecta ao WebSocket da Deriv e analisa candles"""
    url = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
    print("🚀 Conectando ao WebSocket da Deriv...")

    async with websockets.connect(url) as ws:
        # Solicita o stream de candles
        await ws.send(json.dumps({
            "ticks_history": SYMBOL,
            "adjust_start_time": 1,
            "count": 200,
            "end": "latest",
            "granularity": TIMEFRAME * 60,
            "style": "candles"
        }))

        while True:
            response = await ws.recv()
            data = json.loads(response)

            if "candles" in data:
                candles = pd.DataFrame(data["candles"])
                candles["open"] = candles["open"].astype(float)
                candles["high"] = candles["high"].astype(float)
                candles["low"] = candles["low"].astype(float)
                candles["close"] = candles["close"].astype(float)

                # ====== INDICADORES ======
                candles["EMA_curta"] = EMAIndicator(candles["close"], window=9).ema_indicator()
                candles["EMA_media"] = EMAIndicator(candles["close"], window=21).ema_indicator()
                candles["EMA_longa"] = EMAIndicator(candles["close"], window=50).ema_indicator()
                candles["RSI"] = RSIIndicator(candles["close"], window=14).rsi()

                bb = BollingerBands(candles["close"], window=20, window_dev=2)
                candles["BB_upper"] = bb.bollinger_hband()
                candles["BB_lower"] = bb.bollinger_lband()

                ultimo = candles.iloc[-1]

                # ====== LÓGICA DE SINAL ======
                ema_alinhadas_compra = ultimo["EMA_curta"] > ultimo["EMA_media"] > ultimo["EMA_longa"]
                ema_alinhadas_venda = ultimo["EMA_curta"] < ultimo["EMA_media"] < ultimo["EMA_longa"]
                rsi_compra = ultimo["RSI"] < 30
                rsi_venda = ultimo["RSI"] > 70
                perto_banda_inferior = ultimo["close"] <= ultimo["BB_lower"]
                perto_banda_superior = ultimo["close"] >= ultimo["BB_upper"]

                msg_time = datetime.utcnow().strftime("%H:%M:%S")

                # COMPRA
                if ema_alinhadas_compra and rsi_compra and perto_banda_inferior:
                    msg = f"🟢 <b>SINAL DE COMPRA</b>\n⏰ {msg_time} UTC\n💱 Par: {SYMBOL}\n⏳ Candle: {TIMEFRAME}m"
                    enviar_telegram(msg)

                # VENDA
                elif ema_alinhadas_venda and rsi_venda and perto_banda_superior:
                    msg = f"🔴 <b>SINAL DE VENDA</b>\n⏰ {msg_time} UTC\n💱 Par: {SYMBOL}\n⏳ Candle: {TIMEFRAME}m"
                    enviar_telegram(msg)

            await asyncio.sleep(5)

# ====== FLASK (Render mantém o serviço ativo) ======
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot de Análise Deriv ativo no Render!"

def iniciar_bot():
    """Inicia o bot em thread separada"""
    asyncio.run(conectar_deriv())

# ====== MAIN ======
if __name__ == "__main__":
    # Envia mensagem de confirmação
    enviar_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")

    # Inicia o bot em paralelo
    threading.Thread(target=iniciar_bot, daemon=True).start()

    # Mantém o serviço online
    app.run(host="0.0.0.0", port=10000)
