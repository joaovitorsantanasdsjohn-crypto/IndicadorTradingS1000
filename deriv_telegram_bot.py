# deriv_telegram_bot.py
import asyncio
import websockets
import json
import pandas as pd
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

load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # ✅ Chat ID do Telegram
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # ✅ 5 minutos
APP_ID = os.getenv("DERIV_APP_ID", "111022")  # ✅ App ID da Deriv
DERIV_TOKEN = os.getenv("DERIV_TOKEN")  # ✅ Token da Deriv

# Lista de 20 pares
SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
    "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK", "cryBTCUSD"
]

# Diretório para salvar candles
DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# Limite de conexões simultâneas (3)
MAX_CONCURRENT_WS = 3
ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)

# Dicionário para controle de mensagens (limite 1 por 10min por par)
last_notify_time = {}

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()
    if symbol:
        last_time = last_notify_time.get(symbol, 0)
        if now - last_time < 600:  # 10 minutos
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram não configurado. Mensagem:", message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print("📨 Telegram:", message[:100])
        else:
            print(f"❌ Erro Telegram {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('epoch').reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- Sinais ----------------
def gerar_sinal(df: pd.DataFrame):
    ultima = df.iloc[-1]
    close = float(ultima['close'])
    rsi = ultima['rsi']
    bb_low = ultima['bb_lower']
    bb_up = ultima['bb_upper']

    if pd.isna(rsi) or pd.isna(bb_low) or pd.isna(bb_up):
        return None

    if close <= bb_low and rsi <= 30:
        return "COMPRA"
    elif close >= bb_up and rsi >= 70:
        return "VENDA"
    return None

# ---------------- Candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    print(f"[{symbol}] Salvou {len(df)} candles.")

# ---------------- Utilitários ----------------
def seconds_to_next_candle(interval_minutes: int):
    now = datetime.now(timezone.utc)
    total_seconds = int(now.timestamp())
    period = interval_minutes * 60
    return (period - (total_seconds % period)) or period

# ---------------- WebSocket ----------------
async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)
    if not DERIV_TOKEN:
        send_telegram(f"❌ DERIV_TOKEN não configurado. Abortando monitoramento de {symbol}.", symbol)
        return

    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}&l=EN&brand=deriv"
    backoff_seconds = 5
    connected_once = False

    while True:
        await ws_semaphore.acquire()
        try:
            async with websockets.connect(url) as ws:
                # Autenticação com token
                auth_req = {"authorize": DERIV_TOKEN}
                await ws.send(json.dumps(auth_req))
                auth_resp = json.loads(await ws.recv())
                if "error" in auth_resp:
                    send_telegram(f"❌ Erro de autenticação no WebSocket para {symbol}: {auth_resp['error']}", symbol)
                    break

                if not connected_once:
                    send_telegram(f"✅ Conexão ativa com WebSocket da Deriv para {symbol}", symbol)
                    connected_once = True

                print(f"[{symbol}] Conectado e autorizado.")
                first_received = False

                while True:
                    wait = seconds_to_next_candle(CANDLE_INTERVAL)
                    await asyncio.sleep(wait + 1)

                    req = {
                        "ticks_history": symbol,
                        "count": 500,
                        "end": "latest",
                        "granularity": CANDLE_INTERVAL * 60,
                        "style": "candles"
                    }
                    await ws.send(json.dumps(req))
                    try:
                        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    except asyncio.TimeoutError:
                        send_telegram(f"⚠️ Timeout para {symbol}", symbol)
                        break

                    if "history" in data and "candles" in data["history"]:
                        df = pd.DataFrame(data["history"]["candles"])
                        df['close'] = df['close'].astype(float)
                        df_ind = calcular_indicadores(df)
                        save_last_candles(df_ind, symbol)

                        if not first_received:
                            send_telegram(f"📡 [{symbol}] Último fechamento: {df_ind.iloc[-1]['close']:.5f}", symbol)
                            first_received = True

                        sinal = gerar_sinal(df_ind)
                        if sinal:
                            send_telegram(f"💹 Sinal {sinal} detectado para {symbol} ({CANDLE_INTERVAL} min)", symbol)
                    else:
                        send_telegram(f"⚠️ Erro ao obter dados de {symbol}", symbol)
                        break
        except Exception as e:
            send_telegram(f"⚠️ Erro ou desconexão no WebSocket de {symbol}: {e}", symbol)
        finally:
            ws_semaphore.release()
            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 120)

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- Principal ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    send_telegram("🔍 Teste de conexão Telegram: se você recebeu esta mensagem, o bot está OK ✅")

    tasks = []
    for i, sym in enumerate(SYMBOLS):
        tasks.append(asyncio.create_task(monitor_symbol(sym, start_delay=i * 5)))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Encerrando.")
