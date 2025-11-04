import os
import asyncio
import websockets
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import ta
import threading
from flask import Flask

# =======================
# 🔧 Configuração
# =======================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_SYMBOL = os.getenv("DERIV_SYMBOL", "frxEURUSD")

# =======================
# 🧠 Funções auxiliares
# =======================
def send_telegram_message(message: str):
    """Envia mensagem formatada para o Telegram"""
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
# 📊 Lógica do Bot
# =======================
async def deriv_loop():
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    print("🚀 Conectando ao WebSocket da Deriv...")

    async for websocket in websockets.connect(url):
        try:
            print("✅ Conectado ao WebSocket da Deriv!")
            await websocket.send(json.dumps({"ticks_history": DERIV_SYMBOL, "adjust_start_time": 1, "count": 100, "end": "latest", "style": "candles", "granularity": 60}))
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

        except Exception as e:
            print(f"⚠️ Erro no loop Deriv: {e}")
            await asyncio.sleep(5)
            continue

# =======================
# 🌐 Servidor Flask
# =======================
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 Bot de Sinais Deriv Online ✅"

def start_bot_thread():
    asyncio.run(deriv_loop())

if __name__ == "__main__":
    # Inicia o bot em segundo plano
    threading.Thread(target=start_bot_thread, daemon=True).start()
    # Inicia o Flask (Render detecta essa porta)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
