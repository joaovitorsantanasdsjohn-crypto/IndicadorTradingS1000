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
    "frxEURAUD", "frxAUDJPY", "frxGBPAUD","frxGBPCAD", "frxAUDNZD",
    "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

last_notify_time = {}
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
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema55'] = EMAIndicator(df['close'], window=55).ema_indicator()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- Geração de sinal ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    ultima = df.iloc[-1]
    ema9, ema21, ema55 = ultima['ema9'], ultima['ema21'], ultima['ema55']
    rsi, close = ultima['rsi'], ultima['close']
    bb_upper, bb_lower = ultima['bb_upper'], ultima['bb_lower']

    # DEBUG: Log de cálculo
    print(f"\n🧮 [{symbol}] Indicadores calculados:")
    print(f"   RSI={rsi:.2f} | EMA9={ema9:.5f} | EMA21={ema21:.5f} | EMA55={ema55:.5f}")
    print(f"   Bollinger: lower={bb_lower:.5f} | upper={bb_upper:.5f} | close={close:.5f}")

    if pd.isna(ema9) or pd.isna(ema21) or pd.isna(ema55) or pd.isna(rsi):
        print(f"⚠️ [{symbol}] Indicadores incompletos — aguardando mais dados...")
        return None

    # Condições com logs de decisão
    if ema9 > ema21 > ema55 and 30 <= rsi <= 45 and close <= bb_lower:
        print(f"✅ [{symbol}] Condição de *COMPRA* atendida!")
        return "COMPRA"
    elif ema9 < ema21 < ema55 and 55 <= rsi <= 70 and close >= bb_upper:
        print(f"✅ [{symbol}] Condição de *VENDA* atendida!")
        return "VENDA"
    else:
        print(f"🚫 [{symbol}] Nenhum sinal — condições não atendidas.")
        print(f"   EMA alinhadas: {ema9>ema21>ema55 or ema9<ema21<ema55}")
        print(f"   RSI={rsi:.2f} (compra→30-45 | venda→55-70)")
        print(f"   Fechamento vs Bollinger: close={close:.5f} | bb_lower={bb_lower:.5f} | bb_upper={bb_upper:.5f}")
    return None

# ---------------- Salvar candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)

# ---------------- Monitor por símbolo ----------------
async def monitor_symbol(symbol: str):
    reconnect_count = 0
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                reconnect_count += 1
                if reconnect_count > 1:
                    send_telegram(f"🔄 [{symbol}] Reconectado à Deriv (tentativa {reconnect_count}).", symbol)
                else:
                    print(f"🔌 [{symbol}] Nova conexão iniciada.")
                    send_telegram(f"✅ [{symbol}] Conexão WebSocket estabelecida com sucesso.", symbol)

                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                auth = json.loads(await ws.recv())
                if not auth.get("authorize"):
                    print(f"❌ Falha na autorização para {symbol}")
                    await asyncio.sleep(10)
                    continue

                req_hist = {"ticks_history": symbol, "count": 200, "end": "latest",
                            "granularity": CANDLE_INTERVAL * 60, "style": "candles"}
                await ws.send(json.dumps(req_hist))
                data = json.loads(await ws.recv())
                if "candles" not in data:
                    print(f"⚠️ Nenhum dado de candle recebido para {symbol}")
                    await asyncio.sleep(5)
                    continue

                df = pd.DataFrame(data["candles"])
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)

                # 🔹 ADIÇÃO: Forçar log e cálculo imediato na inicialização
                print(f"\n🔍 [{symbol}] Exibindo cálculo inicial dos indicadores após o primeiro download...")
                gerar_sinal(df, symbol)

                if not sent_download_message[symbol]:
                    send_telegram(f"📥 [{symbol}] Download de velas executado com sucesso ({len(df)} candles).", symbol)
                    sent_download_message[symbol] = True

                sub_req = {"ticks_history": symbol, "style": "candles",
                           "granularity": CANDLE_INTERVAL * 60, "end": "latest", "subscribe": 1}
                await ws.send(json.dumps(sub_req))

                print(f"✅ [{symbol}] Conexão ativa e assinada (recebendo candles ao vivo).")

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=180)
                    data = json.loads(msg)
                    candle = data.get("candle")
                    if not candle:
                        continue

                    # DEBUG: log do candle recebido
                    candle_time = datetime.utcfromtimestamp(candle['epoch']).strftime('%H:%M:%S')
                    print(f"📊 [{symbol}] Novo candle recebido às {candle_time} UTC | close={candle['close']}")

                    if df.iloc[-1]['epoch'] != candle['epoch']:
                        df.loc[len(df)] = candle
                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)
                        sinal = gerar_sinal(df, symbol)
                        if sinal:
                            send_telegram(f"💹 [{symbol}] *Sinal {sinal}* detectado!", symbol)
        except asyncio.TimeoutError:
            print(f"⏳ Timeout [{symbol}], tentando novamente...")
            continue
        except Exception as e:
            print(f"⚠️ [{symbol}] erro WebSocket: {e}")
            await asyncio.sleep(random.uniform(3, 7))

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅ (versão multi-conexão com debug detalhado)"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- Execução principal ----------------
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise! 🔍 (conta REAL)")
    print("▶ Iniciando monitoramento paralelo por par (modo debug detalhado)...")

    tasks = [monitor_symbol(symbol) for symbol in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Encerrando...")
