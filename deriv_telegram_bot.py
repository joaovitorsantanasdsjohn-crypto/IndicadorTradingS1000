import os
import asyncio
import json
import requests
import pandas as pd
import ta
import websockets
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask
import threading

# =======================
# 🔧 Configurações
# =======================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_SYMBOL = os.getenv("DERIV_SYMBOL", "frxEURUSD")

# =======================
# 📤 Envio para Telegram
# =======================
def send_telegram_message(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Token ou Chat ID do Telegram não configurados!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# =======================
# 🤖 Lógica principal do bot
# =======================
async def deriv_loop():
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    print("🚀 Conectando ao WebSocket da Deriv...")

    async for websocket in websockets.connect(url):
        try:
            print(f"✅ Conectado ao WebSocket — símbolo: {DERIV_SYMBOL}")
            await websocket.send(json.dumps({
                "ticks_history": DERIV_SYMBOL,
                "adjust_start_time": 1,
                "count": 100,
                "end": "latest",
                "style": "candles",
                "granularity": 300  # 5 minutos
            }))

            last_signal = None

            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if "candles" in data:
                    df = pd.DataFrame(data["candles"])
                    df["close"] = pd.to_numeric(df["close"])
                    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

                    latest_rsi = df["rsi"].iloc[-1]
                    msg_time = datetime.utcnow().strftime("%H:%M:%S")

                    if latest_rsi > 70 and last_signal != "PUT":
                        send_telegram_message(f"📉 <b>SINAL PUT</b> — {DERIV_SYMBOL}\n🕐 {msg_time}\nRSI: {latest_rsi:.2f}")
                        last_signal = "PUT"

                    elif latest_rsi < 30 and last_signal != "CALL":
                        send_telegram_message(f"📈 <b>SINAL CALL</b> — {DERIV_SYMBOL}\n🕐 {msg_time}\nRSI: {latest_rsi:.2f}")
                        last_signal = "CALL"

                await asyncio.sleep(10)

        except Exception as e:
            print(f"⚠️ Erro no loop Deriv: {e}")
            await asyncio.sleep(5)
            continue

# =======================
# 🌐 Servidor Flask (Render precisa disso)
# =======================
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 Bot Deriv está ativo e conectado!"

def start_asyncio_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(deriv_loop())

if __name__ == "__main__":
    # Inicia o bot em segundo plano (não bloqueante)
    threading.Thread(target=start_asyncio_loop, daemon=True).start()

    # Inicia o Flask (Render detecta essa porta)
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Servidor Flask iniciado na porta {port}")
    app.run(host="0.0.0.0", port=port)
