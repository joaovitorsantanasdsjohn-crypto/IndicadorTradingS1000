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

# Inicializa controle de mensagens de download para todos os pares
sent_download_message = {s: False for s in SYMBOLS}

# ---------------- Telegram ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()
    if symbol:
        last_time = last_notify_time.get(symbol, 0)
        if now - last_time < 3:
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        # apenas log no console se não estiver configurado
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
    req = {"ticks_history": symbol, "count": 200, "end": "latest", "granularity": granularity, "style": "candles"}
    await ws.send(json.dumps(req))
    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
    if "error" in data:
        print(f"[{symbol}] ❌ Erro Deriv (history): {data['error'].get('message')} ({data['error'].get('code')})")
        return None
    # data may include 'candles' key
    return data.get("candles")

async def subscribe_candles(ws, symbol: str, granularity: int):
    req = {"ticks_history": symbol, "style": "candles", "granularity": granularity, "end": "latest", "subscribe": 1}
    await ws.send(json.dumps(req))

async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)
    print(f"[{symbol}] ▶ Monitor starting (start_delay={start_delay}s)")

    while True:
        if not is_forex_open():
            print(f"[{symbol}] 🌙 Mercado Forex fechado — aguardando abertura...")
            await asyncio.sleep(600)
            continue

        await ws_semaphore.acquire()
        try:
            async with websockets.connect(WS_URL) as ws:
                if not await authorize_deriv(ws):
                    print(f"[{symbol}] ❌ autorização falhou, tentando novamente em 30s")
                    await asyncio.sleep(30)
                    continue

                print(f"[{symbol}] 🔌 Conectado à Deriv (real).")

                candles = None
                # tenta obter histórico algumas vezes antes de desistir temporariamente
                for attempt in range(3):
                    candles = await fetch_history(ws, symbol, CANDLE_INTERVAL * 60)
                    if candles:
                        break
                    print(f"[{symbol}] ⚠️ fetch_history retornou vazio (attempt {attempt+1}/3).")
                    await asyncio.sleep(3)

                if not candles:
                    # se não obteve o histórico, libera o semáforo e re-tenta depois
                    print(f"[{symbol}] ❌ Não foi possível baixar histórico depois de tentativas — re-tentando em 30s")
                    await ws.close()
                    continue

                df = pd.DataFrame(candles)
                df['close'] = df['close'].astype(float)
                df_ind = calcular_indicadores(df)
                save_last_candles(df_ind, symbol)

                # Envia apenas uma vez a confirmação de download por símbolo
                if not sent_download_message.get(symbol, False):
                    send_telegram(f"📥 [{symbol}] Download de velas executado com sucesso ({len(df)} candles).", symbol)
                    sent_download_message[symbol] = True
                    print(f"[{symbol}] 📥 Mensagem de download enviada.")

                last_epoch = df_ind.iloc[-1]['epoch']
                await subscribe_candles(ws, symbol, CANDLE_INTERVAL * 60)
                print(f"[{symbol}] 🔔 Assinatura de candles iniciada ({CANDLE_INTERVAL}m).")

                while True:
                    if not is_forex_open():
                        print(f"[{symbol}] 🌙 Mercado fechou durante execução — saindo para aguardar abertura.")
                        break
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=180)
                        data = json.loads(msg)
                        if "error" in data:
                            print(f"[{symbol}] ⚠️ Erro recebido do WS: {data['error']}")
                            break

                        # trata tanto o caso 'candles' (lista) quanto 'candle' (dict)
                        candle_data = data.get("candles") or data.get("candle")
                        if not candle_data:
                            continue

                        if isinstance(candle_data, list):
                            candle = candle_data[-1]
                        else:
                            candle = candle_data

                        epoch = candle.get("epoch")
                        if not epoch:
                            continue

                        if epoch != last_epoch:
                            last_epoch = epoch
                            # adiciona novo candle
                            df.loc[len(df)] = candle
                            df_ind = calcular_indicadores(df)
                            sinal = gerar_sinal(df_ind)

                            ultima = df_ind.iloc[-1]
                            # LOG detalhado só no console
                            print(
                                f"[{symbol}] 🧩 Avaliação: Close={ultima['close']:.5f}, "
                                f"RSI={ultima['rsi']:.2f}, EMA9={ultima['ema9']:.5f}, "
                                f"EMA21={ultima['ema21']:.5f}, EMA55={ultima['ema55']:.5f}, "
                                f"BB_up={ultima['bb_upper']:.5f}, BB_low={ultima['bb_lower']:.5f}, "
                                f"Sinal={sinal or 'Nenhum'}"
                            )

                            if sinal:
                                send_telegram(f"💹 [{symbol}] *Sinal {sinal}* detectado!", symbol)
                    except asyncio.TimeoutError:
                        # sem novas mensagens dentro do timeout: apenas reinicia a espera do recv
                        # isso permite sair do inner loop e re-subscrever se necessário
                        # (mas aqui apenas repetimos a espera)
                        # print(f"[{symbol}] ⏳ Timeout aguardando novo candle.")
                        continue
        except Exception as e:
            print(f"[{symbol}] ⚠️ Erro WebSocket geral: {e}")
        finally:
            try:
                ws_semaphore.release()
            except Exception:
                pass
            # pequeno delay aleatório antes de tentar reconectar ao próximo ciclo
            await asyncio.sleep(random.randint(5, 15))

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
    # start flask in background
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise! 🔍 (conta REAL)")
    print("▶ Main started - iniciando monitores para todos os símbolos.")

    # Executa todos os símbolos em paralelo (respeitando o limite do semáforo)
    tasks = [asyncio.create_task(monitor_symbol(sym, start_delay=i * 2)) for i, sym in enumerate(SYMBOLS)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Encerrando...")
