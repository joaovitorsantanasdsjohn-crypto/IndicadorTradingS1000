# main.py (versão completa com limpeza e impressão contínua)
import websocket
import json
import pandas as pd
import numpy as np
import threading
import time
import traceback
from flask import Flask
from telegram import Bot
from ml_model import SignalFilter
from queue import PriorityQueue
import os
import ssl

# ========================
# CONFIGURAÇÕES
# ========================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"
DERIV_WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

ativos = [
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD",
    "frxUSDCAD", "frxEURJPY", "frxGBPJPY", "frxUSDCHF"
]

# ========================
# INICIAIS
# ========================
bot = Bot(token=TELEGRAM_TOKEN)
ml_filter = SignalFilter()

candles_por_ativo = {ativo: [] for ativo in ativos}
ticks_por_ativo = {ativo: [] for ativo in ativos}
current_candle_time = {ativo: None for ativo in ativos}
last_tick_time_by_asset = {ativo: None for ativo in ativos}

_lock = threading.Lock()

PRINT_EVERY_TICK = True
PRINT_TICK_SUMMARY_EVERY = 10
TICK_TIMEOUT = 60  # segundos para avisar se ticks pararem
INACTIVITY_LIMIT = 120  # segundos sem ticks para considerar inativo

# ========================
# FILA PRIORITÁRIA DE MENSAGENS TELEGRAM
# ========================
telegram_queue = PriorityQueue()
MAX_CONCURRENT_TELEGRAM = 2
_telegram_semaphore = threading.Semaphore(MAX_CONCURRENT_TELEGRAM)

def telegram_worker():
    while True:
        priority, message = telegram_queue.get()
        try:
            _telegram_semaphore.acquire()
            bot.send_message(chat_id=CHAT_ID, text=message)
        except Exception as e:
            print(f"[Telegram Worker] ❌ Erro ao enviar Telegram: {e}")
            traceback.print_exc()
        finally:
            _telegram_semaphore.release()
            telegram_queue.task_done()
        time.sleep(0.1)

for _ in range(MAX_CONCURRENT_TELEGRAM):
    t = threading.Thread(target=telegram_worker, daemon=True)
    t.start()

def send_telegram(message, priority=2):
    telegram_queue.put((priority, message))
    print(f"[send_telegram] Mensagem adicionada à fila (priority={priority})")

# ========================
# INDICADORES E ANÁLISE
# ========================
def calculate_indicators(df):
    try:
        import ta
    except Exception as e:
        raise RuntimeError("Pacote 'ta' não encontrado. Adicione 'ta' no requirements.txt.") from e

    if df.empty:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    df['EMA_short'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
    df['EMA_medium'] = ta.trend.EMAIndicator(df['close'], window=13).ema_indicator()
    df['EMA_long'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df

def create_analysis_text(last, prob, ativo):
    ema_s = last.get('EMA_short', np.nan)
    ema_m = last.get('EMA_medium', np.nan)
    ema_l = last.get('EMA_long', np.nan)
    rsi = last.get('RSI', np.nan)
    bb_u = last.get('BB_upper', np.nan)
    bb_l = last.get('BB_lower', np.nan)
    close = last.get('close', np.nan)
    time_ts = last.get('time', None)

    decision = "NEUTRO"
    if not np.isnan(ema_s) and not np.isnan(ema_m) and not np.isnan(ema_l) and not np.isnan(rsi):
        if ema_s > ema_m > ema_l and rsi < 70 and close > bb_l and prob > 0.6:
            decision = "COMPRA"
        elif ema_s < ema_m < ema_l and rsi > 30 and close < bb_u and prob > 0.6:
            decision = "VENDA"

    txt = (
        f"{ativo} | time={time_ts}\n"
        f"close={close:.6f} EMA5={ema_s:.6f} EMA13={ema_m:.6f} EMA21={ema_l:.6f}\n"
        f"RSI={rsi:.2f} BB_upper={bb_u:.6f} BB_lower={bb_l:.6f}\n"
        f"ML_prob={prob:.4f} => DECISÃO: {decision}"
    )
    return txt, decision

def generate_and_notify(df, ativo):
    if df.empty:
        return
    last = df.iloc[-1].to_dict()
    features = [
        float(last.get('EMA_short', np.nan)),
        float(last.get('EMA_medium', np.nan)),
        float(last.get('EMA_long', np.nan)),
        float(last.get('RSI', np.nan)),
        float(last.get('BB_upper', np.nan)),
        float(last.get('BB_lower', np.nan))
    ]
    prob = ml_filter.predict(features)
    analysis_text, decision = create_analysis_text(last, prob, ativo)

    print("[generate_and_notify] ANÁLISE =>\n", analysis_text)
    priority = 1 if decision in ["COMPRA", "VENDA"] else 2
    send_telegram(analysis_text, priority=priority)

# ========================
# DIAGNÓSTICO / MONITOR
# ========================
connected_event = threading.Event()

def status_monitor():
    while True:
        try:
            qsize = telegram_queue.qsize()
            connected = connected_event.is_set()
            buf_summary = {a: len(ticks_por_ativo.get(a, [])) for a in ativos}
            last_ticks = {a: last_tick_time_by_asset.get(a) for a in ativos}
            print("==== STATUS MONITOR ====")
            print("[STATUS] WebSocket conectado:", connected)
            print("[STATUS] Telegram queue size:", qsize)
            print("[STATUS] Tick buffers:", buf_summary)
            print("[STATUS] Últimos tick times:", last_ticks)

            now = time.time()
            for a, t in last_tick_time_by_asset.items():
                if t is not None and now - t > TICK_TIMEOUT:
                    print(f"[status_monitor] ⚠️ Sem ticks de {a} nos últimos {TICK_TIMEOUT}s!")

            print("========================")
        except Exception:
            print("[status_monitor] Erro:\n", traceback.format_exc())
        time.sleep(30)

threading.Thread(target=status_monitor, daemon=True).start()

# ========================
# LIMPEZA DE SÍMBOLOS INATIVOS
# ========================
def limpar_simbolos_inativos():
    while True:
        time.sleep(10)
        now = time.time()
        with _lock:
            inativos = [a for a, t in last_tick_time_by_asset.items() if t is not None and now - t > INACTIVITY_LIMIT]
            for a in inativos:
                ticks_por_ativo.pop(a, None)
                candles_por_ativo.pop(a, None)
                current_candle_time.pop(a, None)
                last_tick_time_by_asset.pop(a, None)
                if a in ativos:
                    ativos.remove(a)
                print(f"[limpar_simbolos_inativos] Símbolo removido por inatividade: {a}")

threading.Thread(target=limpar_simbolos_inativos, daemon=True).start()

# ========================
# PROCESSAMENTO DE CANDLES
# ========================
def process_candle_if_needed(ativo):
    try:
        with _lock:
            tick_time = last_tick_time_by_asset.get(ativo)
            if tick_time is None:
                return
            while tick_time >= current_candle_time.get(ativo, 0) + 300:
                candle_ticks = ticks_por_ativo.get(ativo, [])
                if candle_ticks:
                    candle = {
                        'time': current_candle_time[ativo],
                        'open': candle_ticks[0],
                        'high': max(candle_ticks),
                        'low': min(candle_ticks),
                        'close': candle_ticks[-1]
                    }
                    if ativo not in candles_por_ativo:
                        candles_por_ativo[ativo] = []
                    candles_por_ativo[ativo].append(candle)
                    if len(candles_por_ativo[ativo]) > 500:
                        candles_por_ativo[ativo].pop(0)

                    df = pd.DataFrame(candles_por_ativo[ativo])
                    df = calculate_indicators(df)

                    last = df.iloc[-1]
                    print(f"[Indicadores] {ativo} | O={last['open']} H={last['high']} L={last['low']} C={last['close']}"
                          f" EMA5={last['EMA_short']:.5f} EMA13={last['EMA_medium']:.5f} EMA21={last['EMA_long']:.5f}"
                          f" RSI={last['RSI']:.2f} BBU={last['BB_upper']:.5f} BBL={last['BB_lower']:.5f}")

                    generate_and_notify(df, ativo)

                ticks_por_ativo[ativo] = []
                current_candle_time[ativo] += 300
    except Exception:
        print("[process_candle_if_needed] Erro:\n", traceback.format_exc())

# ========================
# WEBSOCKET DERIV
# ========================
def on_message(ws, message):
    try:
        data = json.loads(message)
        if 'tick' not in data:
            return

        tick = data['tick']
        ativo = tick.get('symbol')
        if ativo not in ativos:
            return

        tick_price = float(tick.get('quote'))
        tick_time = int(tick.get('epoch'))
        last_tick_time_by_asset[ativo] = tick_time

        if PRINT_EVERY_TICK:
            print(f"[on_message] Tick: {ativo} price={tick_price} epoch={tick_time}")

        with _lock:
            ticks_por_ativo.setdefault(ativo, []).append(tick_price)
            if current_candle_time.get(ativo) is None:
                current_candle_time[ativo] = tick_time - (tick_time % 300)

        threading.Thread(target=process_candle_if_needed, args=(ativo,), daemon=True).start()
    except Exception:
        print("[on_message] Erro:\n", traceback.format_exc())

def on_open(ws):
    connected_event.set()
    for ativo in ativos:
        sub = {"ticks": ativo, "subscribe": 1}
        ws.send(json.dumps(sub))

def on_error(ws, error):
    print("[on_error] Erro WebSocket:", error)

def on_close(ws, close_status_code, close_msg):
    connected_event.clear()
    print(f"[on_close] WebSocket fechado: code={close_status_code} msg={close_msg}")

def on_pong(ws, message):
    print("[on_pong] Pong recebido:", message)

# ========================
# EXECUÇÃO / RECONNECT
# ========================
def run_ws_forever():
    websocket.enableTrace(True)
    backoff = 1
    while True:
        try:
            headers = ["Origin: https://binary.com"]
            ws = websocket.WebSocketApp(
                DERIV_WEBSOCKET_URL,
                header=headers,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_pong=on_pong
            )
            sslopt = {"cert_reqs": ssl.CERT_REQUIRED}
            ws.run_forever(ping_interval=30, ping_timeout=10, sslopt=sslopt)
        except Exception:
            print("[run_ws_forever] Exceção:\n", traceback.format_exc())
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)

# ========================
# FLASK PARA UPTIME
# ========================
app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 IndicadorTradingS1000 ativo e rodando!"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    run_ws_forever()
