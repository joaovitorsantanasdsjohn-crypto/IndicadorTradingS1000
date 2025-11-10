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
import math

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")  # ✅ token da conta REAL
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos ( padrão 5 )
APP_ID = os.getenv("DERIV_APP_ID", "111022")

# Utiliza apenas ambiente REAL conforme solicitado
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# 7 pares mais líquidos (mantive CRYBTCUSD com prefixo MAIÚSCULO)
SYMBOLS = [
    "frxEURUSD",
    "frxGBPUSD",
    "frxUSDJPY",
    "frxUSDCHF",
    "frxAUDUSD",
    "frxUSDCAD",
    "CRYBTCUSD"
]

# Diretório para salvar candles
DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# Limites / controles
MAX_CONCURRENT_WS = 3
ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)

# Controle de notificações Telegram:
# - notify_state[symbol] holds booleans for sent flags: connection_sent, first_candles_sent
# - last_reconnect_msg tracks last reconnect message timestamp per symbol (min 600s)
notify_state = {s: {"connection_sent": False, "first_candles_sent": False} for s in SYMBOLS}
last_reconnect_msg = {}  # symbol -> timestamp (to limit reconnection notices to 10 min)
RECONNECT_NOTIFY_MIN_SECONDS = 600  # 10 minutos

# ---------------- Telegram (apenas mensagens permitidas) ----------------
def send_telegram(message: str, force: bool = False):
    """Envia mensagem ao Telegram se token/config ok.
    'force' permite envio independente de controles locais (uso restrito)."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram não configurado. Mensagem:", message)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"❌ Erro Telegram {r.status_code}: {r.text}")
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
    # critérios: ambos (RSI + Bollinger) devem concordar
    if close <= bb_low and rsi <= 30:
        return "COMPRA"
    if close >= bb_up and rsi >= 70:
        return "VENDA"
    return None

# ---------------- Salvamento candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df.tail(200).to_csv(path, index=False)
    print(f"[{symbol}] ✅ {len(df.tail(200))} candles salvos em {path}")

# ---------------- Utilitários ----------------
def seconds_to_next_candle(interval_minutes: int):
    now = datetime.now(timezone.utc)
    total_seconds = int(now.timestamp())
    period = interval_minutes * 60
    seconds_passed = total_seconds % period
    return (period - seconds_passed) if seconds_passed != 0 else 0

# ---------------- WebSocket & Authorization ----------------
async def authorize_deriv(ws):
    """Autoriza usando DERIV_TOKEN. Retorna True se autorizado."""
    if not DERIV_TOKEN:
        print("❌ DERIV_TOKEN não configurado.")
        return False
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        # espera por resposta autorizacao (com timeout curto)
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        resp = json.loads(raw)
        if "authorize" in resp and resp["authorize"].get("is_virtual") is not None:
            print("🔐 Autorizado. loginid:", resp["authorize"].get("loginid"))
            return True
        # se veio erro
        if "error" in resp:
            print("❌ Resposta de autorização:", resp["error"])
            return False
        # fallback
        return False
    except Exception as e:
        print("❌ Exceção na autorização:", e)
        return False

async def fetch_history_candles(ws, symbol: str, granularity: int, retries: int = 2):
    """Pede ticks_history no websocket. Tenta 'retries' vezes antes de falhar."""
    req = {
        "ticks_history": symbol,
        "count": 500,
        "end": "latest",
        "granularity": granularity,
        "style": "candles",
        "adjust_start_time": 1
    }
    for attempt in range(1, retries + 2):
        try:
            await ws.send(json.dumps(req))
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
            data = json.loads(raw)
            candles = data.get("history", {}).get("candles")
            if candles:
                return candles
            # se veio erro com mensagem útil, tenta novamente
            err = data.get("error") or data.get("msg")
            print(f"[{symbol}] tentativa {attempt}: sem candles (mensagem: {err})")
        except asyncio.TimeoutError:
            print(f"[{symbol}] tentativa {attempt}: timeout aguardando resposta.")
        except Exception as e:
            print(f"[{symbol}] tentativa {attempt}: exceção fetch_history: {e}")
        await asyncio.sleep(1 + attempt)  # pequeno intervalo entre tentativas
    return None

# ---------------- Monitor por símbolo ----------------
async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)

    # backoff para reconexões
    backoff = 5

    while True:
        await ws_semaphore.acquire()
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10, open_timeout=15) as ws:
                # autoriza com token
                authorized = await authorize_deriv(ws)
                if not authorized:
                    now = time.time()
                    last = last_reconnect_msg.get(symbol, 0)
                    if now - last >= RECONNECT_NOTIFY_MIN_SECONDS:
                        send_telegram(f"❌ Falha de autorização para {symbol} — verifique DERIV_TOKEN.", force=True)
                        last_reconnect_msg[symbol] = now
                    print(f"[{symbol}] autorização falhou; aguardando backoff {backoff}s.")
                    await ws.close()
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    continue

                # envia mensagem de conexão somente uma vez por símbolo
                if not notify_state[symbol]["connection_sent"]:
                    send_telegram(f"✅ Conexão WebSocket aberta para {symbol} (REAL).", force=True)
                    notify_state[symbol]["connection_sent"] = True

                # Espera até próximo fechamento de candle para pedir histórico 5m
                while True:
                    wait = seconds_to_next_candle(CANDLE_INTERVAL)
                    # aguarda até fecho da vela (um segundo a mais para segurança)
                    if wait > 0:
                        await asyncio.sleep(wait + 1)
                    else:
                        await asyncio.sleep(1)

                    # pede candles de CANDLE_INTERVAL minutos (em segundos)
                    gran = CANDLE_INTERVAL * 60
                    candles = await fetch_history_candles(ws, symbol, gran)
                    if not candles:
                        # tenta fallback para 60s (1m) grains ou notifica conforme política
                        candles = await fetch_history_candles(ws, symbol, 60)
                        if not candles:
                            # notifica de reconexão/erro no máximo uma vez a cada 10 minutos
                            now = time.time()
                            last = last_reconnect_msg.get(symbol, 0)
                            if now - last >= RECONNECT_NOTIFY_MIN_SECONDS:
                                send_telegram(f"⚠️ Não foi possível obter candles para {symbol} (tentativas).", force=True)
                                last_reconnect_msg[symbol] = now
                            print(f"[{symbol}] nenhum candle retornado após tentativas.")
                            # enfraquece a conexão atual para forçar reconexão
                            break

                    # processa candles
                    df = pd.DataFrame(candles)
                    if 'close' not in df.columns:
                        print(f"[{symbol}] dados retornados inválidos: sem 'close' column.")
                        break
                    df['close'] = df['close'].astype(float)
                    df_ind = calcular_indicadores(df)
                    save_last_candles(df_ind, symbol)

                    # envia apenas o primeiro download de candles para confirmar
                    if not notify_state[symbol]["first_candles_sent"]:
                        last_close = df_ind.iloc[-1]['close']
                        send_telegram(f"📡 [{symbol}] Candles recebidos. Último fechamento: {last_close:.5f}", force=True)
                        notify_state[symbol]["first_candles_sent"] = True

                    # gera sinal e envia se houver
                    sinal = gerar_sinal(df_ind)
                    if sinal:
                        send_telegram(f"💹 Sinal *{sinal}* detectado para {symbol} ({CANDLE_INTERVAL}m)", force=True)

                    # mantém loop para próxima vela (não envia logs adicionais)
                    # reset backoff to default on success
                    backoff = 5

        except Exception as e:
            # reconexão: enviar mensagem limitada por tempo
            now = time.time()
            last = last_reconnect_msg.get(symbol, 0)
            if now - last >= RECONNECT_NOTIFY_MIN_SECONDS:
                send_telegram(f"⚠️ Reconexão/erro WebSocket para {symbol}: {e}", force=True)
                last_reconnect_msg[symbol] = now
            print(f"[{symbol}] exceção geral no monitor: {e}")
        finally:
            ws_semaphore.release()
            # backoff exponencial antes de tentar reconectar
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 120)

# ---------------- Flask (manter processo ativo no Render) ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------- Execução principal ----------------
async def main():
    # Start flask thread
    threading.Thread(target=run_flask, daemon=True).start()

    # Mensagem de inicialização única
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise! 🔍 (LER: usa conta REAL)")

    # cria tasks em grupos de 2 pares por vez (escalonamento)
    group_size = 2
    delay_between_groups = 30
    groups = [SYMBOLS[i:i + group_size] for i in range(0, len(SYMBOLS), group_size)]

    for idx, group in enumerate(groups):
        send_telegram(f"⏳ Iniciando grupo {idx+1}/{len(groups)}: {', '.join(group)}")
        tasks = []
        for i, sym in enumerate(group):
            tasks.append(asyncio.create_task(monitor_symbol(sym, start_delay=i * 5)))
        # aguarda os tasks do grupo. Cada task roda loop infinito; se um task falhar, o gather aguardará até a reconexão interna.
        # usamos gather para manter o grupo vivo. Se preferir paralelizar todos os grupos em background, é só criar tasks sem await.
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Encerrando...")
