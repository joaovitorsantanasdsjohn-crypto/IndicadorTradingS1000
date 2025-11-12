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

# ---------------- Controle de horário Forex ----------------
def is_forex_open() -> bool:
    """
    Retorna True se o mercado Forex estiver aberto.
    Horário global Forex:
    - Abre: Domingo 22:00 UTC
    - Fecha: Sexta 21:00 UTC
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=segunda ... 6=domingo
    hour = now.hour
    minute = now.minute

    # Domingo antes das 22:00 UTC → fechado
    if weekday == 6 and (hour < 22):
        return False
    # Sexta após 21:00 UTC → fechado
    if weekday == 4 and (hour >= 21):
        return False
    # Sábado inteiro → fechado
    if weekday == 5:
        return False
    # Domingo antes das 22h → fechado
    if weekday == 6 and hour < 22:
        return False

    # Caso contrário → aberto
    return True

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('epoch').reset_index(drop=True)
    df['close'] = df['close'].astype(float)

    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema55'] = EMAIndicator(df['close'], window=55).ema_indicator()
    df['ema200'] = EMAIndicator(df['close'], window=200).ema_indicator()

    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

def gerar_sinal(df: pd.DataFrame):
    ultima = df.iloc[-1]
    ema9 = ultima['ema9']
    ema55 = ultima['ema55']
    ema200 = ultima['ema200']
    rsi = ultima['rsi']

    if pd.isna(ema9) or pd.isna(ema55) or pd.isna(ema200) or pd.isna(rsi):
        return None

    if ema9 > ema55 > ema200 and rsi > 52:
        return "COMPRA"
    elif ema9 < ema55 < ema200 and rsi < 48:
        return "VENDA"
    return None

def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    print(f"[{symbol}] ✅ {len(df)} candles salvos.")

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

async def fetch_history(ws, symbol: str, granularity: int):
    req = {
        "ticks_history": symbol,
        "count": 200,
        "end": "latest",
        "granularity": granularity,
        "style": "candles"
    }
    await ws.send(json.dumps(req))
    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
    if "error" in data:
        print(f"[{symbol}] ❌ Erro Deriv: {data['error'].get('message')} ({data['error'].get('code')})")
        return None
    return data.get("candles")

# ✅ Correção aplicada aqui
async def subscribe_candles(ws, symbol: str, granularity: int):
    """
    Corrigido: uso de 'ticks_history' + style=candles + subscribe=1,
    conforme a API atual da Deriv.
    """
    req = {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": granularity,
        "end": "latest",
        "subscribe": 1
    }
    await ws.send(json.dumps(req))

async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)
    connected_once = False

    while True:
        if not is_forex_open():
            print(f"[{symbol}] 🌙 Mercado Forex fechado — aguardando abertura...")
            await asyncio.sleep(600)
            continue

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

                candles = await fetch_history(ws, symbol, CANDLE_INTERVAL * 60)
                if not candles:
                    send_telegram(f"⚠️ Nenhum candle inicial retornado para {symbol}", symbol)
                    break

                df = pd.DataFrame(candles)
                df['close'] = df['close'].astype(float)
                df_ind = calcular_indicadores(df)
                save_last_candles(df_ind, symbol)

                last_epoch = df_ind.iloc[-1]['epoch']
                send_telegram(f"📊 [{symbol}] Histórico inicial carregado ({len(df)} candles). Último close: {df_ind.iloc[-1]['close']:.5f}", symbol)

                await subscribe_candles(ws, symbol, CANDLE_INTERVAL * 60)
                print(f"[{symbol}] 🔔 Assinatura de candles iniciada ({CANDLE_INTERVAL}m).")

                while True:
                    if not is_forex_open():
                        print(f"[{symbol}] ⚠️ Mercado fechou durante execução — pausa.")
                        send_telegram(f"🌙 Mercado Forex fechado — pausando análise de {symbol}.", symbol)
                        break

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=180)
                        data = json.loads(msg)
                        if "error" in data:
                            print(f"[{symbol}] ⚠️ Erro Deriv: {data['error']}")
                            break

                        candle = data.get("candles") or data.get("candle")
                        if candle:
                            epoch = candle.get("epoch")
                            if epoch and epoch != last_epoch:
                                last_epoch = epoch
                                close = float(candle["close"])
                                df.loc[len(df)] = candle
                                df_ind = calcular_indicadores(df)
                                sinal = gerar_sinal(df_ind)
                                msg = f"🕐 [{symbol}] Novo candle {datetime.utcfromtimestamp(epoch).strftime('%H:%M:%S')} — Close: {close:.5f}"
                                if sinal:
                                    msg += f"\n💹 *Sinal {sinal}* detectado!"
                                send_telegram(msg, symbol)
                    except asyncio.TimeoutError:
                        print(f"[{symbol}] ⏳ Timeout aguardando novo candle.")
                        break
        except Exception as e:
            print(f"[{symbol}] ⚠️ Erro WebSocket: {e}")
        finally:
            ws_semaphore.release()
            await asyncio.sleep(random.randint(20, 45))

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
