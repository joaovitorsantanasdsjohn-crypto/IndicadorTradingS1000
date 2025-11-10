# deriv_telegram_bot_diag.py
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
    "frxEURUSD",
    "frxGBPUSD",
    "frxUSDJPY",
    "frxUSDCHF",
    "frxAUDUSD",
    "frxUSDCAD",
    "CRYBTCUSD"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

MAX_CONCURRENT_WS = 3
ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)
last_notify_time = {}

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()
    if symbol:
        last_time = last_notify_time.get(symbol, 0)
        if now - last_time < 300:
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

def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    print(f"[{symbol}] ✅ {len(df)} candles salvos.")

def seconds_to_next_candle(interval_minutes: int):
    now = datetime.now(timezone.utc)
    total_seconds = int(now.timestamp())
    period = interval_minutes * 60
    return (period - (total_seconds % period)) or period

# ---------------- WebSocket ----------------
async def authorize_deriv(ws):
    if not DERIV_TOKEN:
        print("⚠️ DERIV_TOKEN não configurado no ambiente!")
        return False
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        response = json.loads(await ws.recv())
        if response.get("authorize"):
            print(f"🔐 Autorizado como {response['authorize'].get('loginid', 'desconhecido')}")
            return True
        else:
            print(f"❌ Falha na autorização: {response}")
            return False
    except Exception as e:
        print(f"❌ Erro ao autorizar: {e}")
        return False

async def fetch_candles(ws, symbol: str, granularity: int):
    """Obtém candles e mostra erros detalhados."""
    req = {
        "ticks_history": symbol,
        "count": 500,
        "end": int(datetime.now(timezone.utc).timestamp()),  # ⏰ Força timestamp UTC
        "granularity": granularity,
        "style": "candles",
        "subscribe": 0,
        "adjust_start_time": 1
    }
    await ws.send(json.dumps(req))
    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))

    if "error" in data:
        print(f"[{symbol}] ❌ Erro Deriv: {data['error'].get('message')} ({data['error'].get('code')})")
        return None

    candles = data.get("candles")
    if not candles:
        print(f"[{symbol}] ⚠️ Nenhum candle retornado. Resposta bruta:")
        print(json.dumps(data, indent=2))
    return candles

async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)
    connected_once = False

    while True:
        await ws_semaphore.acquire()
        try:
            async with websockets.connect(WS_URL) as ws:
                if not await authorize_deriv(ws):
                    send_telegram(f"❌ Falha na autorização Deriv para {symbol}", symbol)
                    break

                if not connected_once:
                    send_telegram(f"✅ Conexão WebSocket aberta para {symbol} (REAL).", symbol)
                    connected_once = True

                print(f"[{symbol}] 🔌 Conectado à Deriv (real).")

                wait = seconds_to_next_candle(CANDLE_INTERVAL)
                await asyncio.sleep(wait + 1)

                try:
                    candles = await fetch_candles(ws, symbol, CANDLE_INTERVAL * 60)
                except asyncio.TimeoutError:
                    print(f"[{symbol}] ⏱ Timeout ao receber dados.")
                    break
                except Exception as e:
                    print(f"[{symbol}] ❌ Erro inesperado: {e}")
                    break

                if candles:
                    df = pd.DataFrame(candles)
                    df['close'] = df['close'].astype(float)
                    df_ind = calcular_indicadores(df)
                    save_last_candles(df_ind, symbol)

                    close_price = df_ind.iloc[-1]['close']
                    send_telegram(f"📊 [{symbol}] Último fechamento: {close_price:.5f}", symbol)

                    sinal = gerar_sinal(df_ind)
                    if sinal:
                        send_telegram(f"💹 *Sinal {sinal}* detectado em {symbol} ({CANDLE_INTERVAL} min)", symbol)
                else:
                    send_telegram(f"⚠️ Não foi possível obter candles para {symbol}", symbol)
                    break

        except Exception as e:
            print(f"[{symbol}] ⚠️ Erro WebSocket: {e}")
        finally:
            ws_semaphore.release()
            await asyncio.sleep(random.randint(15, 45))

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
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise! 🔍 (LER: usa conta REAL)")

    group_size = 2
    delay_between_groups = 30
    groups = [SYMBOLS[i:i + group_size] for i in range(0, len(SYMBOLS), group_size)]

    for index, group in enumerate(groups):
        send_telegram(f"⏳ Iniciando grupo {index + 1}/{len(groups)}: {', '.join(group)}")
        tasks = [asyncio.create_task(monitor_symbol(sym, start_delay=i * 5)) for i, sym in enumerate(group)]
        await asyncio.gather(*tasks)
        if index < len(groups) - 1:
            await asyncio.sleep(delay_between_groups)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Encerrando...")
