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
import threading
from flask import Flask

load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", 5))  # minutos
CANDLES_KEEP = int(os.getenv("CANDLES_KEEP", 50))  # quantas velas salvar por par
SAVE_FILE = os.getenv("CANDLES_FILE", "candles_data.json")

# Lê os pares da variável de ambiente SYMBOLS (separados por vírgula) ou usa lista padrão
SYMBOLS = [s for s in os.getenv("SYMBOLS", "").split(",") if s]
if not SYMBOLS:
    SYMBOLS = [
        "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
        "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
        "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
        "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
    ]

# Em memória: dicionário {symbol: [candles]}
candles_store = {}

# ---------------- Função Telegram ----------------
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram não configurado (TELEGRAM_TOKEN/CHAT_ID faltando). Mensagem:", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}")

# ---------------- Indicadores ----------------
def calcular_indicadores(df):
    # mantém sua implementação de indicadores (EMA, RSI, Bollinger)
    df['ema_curta'] = EMAIndicator(df['close'], window=5).ema_indicator()
    df['ema_media'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema_longa'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_medio'] = bb.bollinger_mavg()
    df['bb_sup'] = bb.bollinger_hband()
    df['bb_inf'] = bb.bollinger_lband()
    return df

# ---------------- Gerar Sinal ----------------
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

# ---------------- Funções de armazenamento ----------------
def load_saved_candles():
    global candles_store
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                candles_store = json.load(f)
                # garantir listas e corte de tamanho
                for s in list(candles_store.keys()):
                    candles_store[s] = candles_store[s][-CANDLES_KEEP:]
        except Exception as e:
            print("Falha ao carregar arquivo de candles:", e)
            candles_store = {}
    else:
        candles_store = {}

def save_candles_file():
    # grava de forma segura (temp -> rename)
    try:
        tmp = SAVE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(candles_store, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SAVE_FILE)
    except Exception as e:
        print("Erro ao salvar candles:", e)

def update_candles_store(symbol, candles):
    """
    candles: lista de candles (cada candle é dict com keys timestamp/open/close/high/low)
    Mantém últimas CANDLES_KEEP entradas por símbolo.
    """
    existing = candles_store.get(symbol, [])
    # converter incoming candles to minimal serializable format (já vem como dicts)
    # manter ordem cronológica: presumimos que lista vem do mais antigo para o mais novo
    # juntar e cortar
    # Evitar duplicatas baseadas em epoch (epoch pode estar em 'epoch' ou 'dt' — usamos 'epoch' se existir)
    def key(c):
        return c.get("epoch") or c.get("time") or c.get("tick") or c.get("dt") or c.get("open_time") or c.get("timestamp")
    # combine, dedupe by epoch-like key
    combined = existing + candles
    seen = {}
    out = []
    for c in combined:
        k = key(c)
        if k is None:
            # fallback: use string of candle
            k = json.dumps(c, sort_keys=True)
        if k in seen:
            continue
        seen[k] = True
        out.append(c)
    # keep last N candles
    out = out[-CANDLES_KEEP:]
    candles_store[symbol] = out
    save_candles_file()

# Inicializa armazenamento ao startup
load_saved_candles()

# ---------------- Monitoramento WebSocket ----------------
async def monitor_symbol(symbol, start_delay=0):
    await asyncio.sleep(start_delay)  # delay para evitar bursts de conexão
    url = f"wss://ws.binaryws.com/websockets/v3?app_id=1089"

    while True:
        try:
            # open_timeout evita ficar preso no handshake
            async with websockets.connect(url, ping_interval=20, open_timeout=10) as ws:
                send_telegram(f"✅ Conexão ativa com WebSocket da Deriv para {symbol}!")
                print(f"{datetime.utcnow().isoformat()} - 🚀 Conexão ativa com WebSocket da Deriv para {symbol}")

                # Solicita histórico de candles (últimas)
                req = {
                    "ticks_history": symbol,
                    "count": 500,  # baixa as últimas 500 (você já usava 100; mantive 500 para mais contexto)
                    "granularity": CANDLE_INTERVAL * 60,
                    "style": "candles"
                }
                await ws.send(json.dumps(req))

                got_first_history = False
                last_saved_epoch = None

                # força um recv com timeout para garantir que realmente recebemos candles
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=15)
                except asyncio.TimeoutError:
                    # se não vier resposta inicial, alerta e reconecta
                    send_telegram(f"⚠️ Timeout inicial ao solicitar candles para {symbol} (nenhuma resposta em 15s). Vou reconectar.")
                    print(f"{datetime.utcnow().isoformat()} - ⚠️ Timeout inicial para {symbol}")
                    raise Exception("timeout_initial_history")

                # processa a primeira resposta (pode ou não conter 'history')
                data = json.loads(response)
                if "history" in data and "candles" in data["history"]:
                    candles = data["history"]["candles"]
                    # salva últimas CANDLES_KEEP
                    # converte campos numéricos para tipos simples
                    processed = []
                    for c in candles[-CANDLES_KEEP:]:
                        processed.append({
                            "epoch": c.get("epoch"),
                            "open": float(c.get("open")),
                            "high": float(c.get("high")),
                            "low": float(c.get("low")),
                            "close": float(c.get("close"))
                        })
                    update_candles_store(symbol, processed)
                    last_saved_epoch = processed[-1].get("epoch")
                    send_telegram(f"📡 Primeiro download de candles concluído para *{symbol}* — salvei {len(processed)} velas. Fechamento mais recente: {processed[-1]['close']}")
                    print(f"{datetime.utcnow().isoformat()} - 📡 Primeiro download candles {symbol}: {len(processed)} velas, close={processed[-1]['close']}")
                    got_first_history = True
                else:
                    # Se a primeira resposta não é history, apenas logamos e continuamos
                    print(f"{datetime.utcnow().isoformat()} - resposta inicial sem 'history' para {symbol}: {data.get('msg_type','no_type')}")
                    # Continue to listen

                # Loop para receber mensagens contínuas
                while True:
                    try:
                        # aguarda próxima mensagem (sem bloquear indefinidamente)
                        response = await asyncio.wait_for(ws.recv(), timeout=60)
                        data = json.loads(response)

                        # se for history, atualizamos o store e rodamos análise de sinal
                        if "history" in data and "candles" in data["history"]:
                            candles = data["history"]["candles"]
                            processed = []
                            for c in candles[-CANDLES_KEEP:]:
                                processed.append({
                                    "epoch": c.get("epoch"),
                                    "open": float(c.get("open")),
                                    "high": float(c.get("high")),
                                    "low": float(c.get("low")),
                                    "close": float(c.get("close"))
                                })
                            update_candles_store(symbol, processed)

                            # confirma concretamente que salvou nova vela (se mudou)
                            new_epoch = processed[-1].get("epoch")
                            if new_epoch != last_saved_epoch:
                                last_saved_epoch = new_epoch
                                send_telegram(f"💾 Velas atualizadas para *{symbol}* — última vela fechada: {processed[-1]['close']} (epoch={new_epoch})")
                                print(f"{datetime.utcnow().isoformat()} - 💾 Atualizado {symbol}, last_close={processed[-1]['close']}, epoch={new_epoch}")

                            # transforma em DataFrame e calcula indicadores/sinais
                            df = pd.DataFrame(processed)
                            df['close'] = df['close'].astype(float)
                            df['open'] = df['open'].astype(float)
                            df = calcular_indicadores(df)

                            # gerar sinal conforme sua função
                            sinal = gerar_sinal(df)
                            if sinal:
                                send_telegram(f"💹 Sinal *{sinal}* detectado para *{symbol}* (vela {CANDLE_INTERVAL} min) — close: {df.iloc[-1]['close']}")
                                print(f"{datetime.utcnow().isoformat()} - 💹 Sinal {sinal} para {symbol}, close={df.iloc[-1]['close']}")

                        # outros tipos de mensagens: keepalive, ticks, etc — apenas log mínimo
                    except asyncio.TimeoutError:
                        # se nenhum dado chegou por 60s, checar/alertar (mas não necessariamente desconectar)
                        send_telegram(f"⚠️ Timeout: sem novas mensagens do WebSocket para *{symbol}* nos últimos 60s — verificando conexão.")
                        print(f"{datetime.utcnow().isoformat()} - ⚠️ Timeout 60s sem mensagens para {symbol}")
                        # faz um breve sleep e depois continua esperando (ou podemos decidir reconectar)
                    except websockets.ConnectionClosedOK:
                        send_telegram(f"⚠️ Conexão fechada normalmente para {symbol}. Vou reconectar.")
                        print(f"{datetime.utcnow().isoformat()} - conexão fechada normalmente {symbol}")
                        break
                    except websockets.ConnectionClosedError as e:
                        send_telegram(f"❌ Conexão do WebSocket fechada com erro para {symbol}: {e}. Reconectando...")
                        print(f"{datetime.utcnow().isoformat()} - conexão fechada com erro {symbol}: {e}")
                        break
                    except Exception as e:
                        send_telegram(f"❌ Erro ao processar mensagem do WebSocket para {symbol}: {e}")
                        print(f"{datetime.utcnow().isoformat()} - erro processamento {symbol}: {e}")
                        break

                    # espera até a próxima vela completa antes de processar novamente (reduz custo)
                    # observe: isso apenas faz sleep entre checagens; recebimento de WS continua assincrono
                    await asyncio.sleep(0.1)

        except Exception as e:
            # Mensagem clara no Telegram e no log; faz backoff pequeno antes de reconectar
            send_telegram(f"🔄 Tentando reconectar WebSocket para *{symbol}* após erro: {e}")
            print(f"{datetime.utcnow().isoformat()} - 🔄 Reconectando {symbol} depois de erro: {e}")
            await asyncio.sleep(5)

# ---------------- Flask Web Service ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    # O Flask será apenas um health-check para Render
    app.run(host="0.0.0.0", port=port)

# ---------------- Função Principal ----------------
async def main():
    # inicia flask em thread separada
    threading.Thread(target=run_flask, daemon=True).start()

    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    for symbol in SYMBOLS:
        send_telegram(f"📊 Começando a monitorar **{symbol}**.")

    # cria tarefas com pequeno delay entre cada abertura para reduzir handshakes simultâneos
    tasks = []
    delay_between = float(os.getenv("DELAY_BETWEEN_STARTS", 1.5))  # segundos
    for i, symbol in enumerate(SYMBOLS):
        start_delay = i * delay_between
        tasks.append(asyncio.create_task(monitor_symbol(symbol, start_delay=start_delay)))

    # aguarda todas (rodarão indefinidamente)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
