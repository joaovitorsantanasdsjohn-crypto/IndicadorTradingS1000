# main.py
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

_lock = threading.Lock()

PRINT_EVERY_TICK = True
PRINT_TICK_SUMMARY_EVERY = 10

# ========================
# FILA PRIORITÁRIA DE MENSAGENS TELEGRAM
# ========================
# Prioridade: COMPRA/VENDA = 1 (alta), NEUTRO = 2 (baixa)
telegram_queue = PriorityQueue()
MAX_CONCURRENT_TELEGRAM = 2
_telegram_semaphore = threading.Semaphore(MAX_CONCURRENT_TELEGRAM)

def telegram_worker():
    while True:
        priority, message = telegram_queue.get()
        try:
            _telegram_semaphore.acquire()
            bot.send_message(chat_id=CHAT_ID, text=message)
            print("✅ Telegram enviado com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao enviar Telegram: {e}")
            traceback.print_exc()
        finally:
            _telegram_semaphore.release()
            telegram_queue.task_done()
        time.sleep(0.1)  # evita sobrecarga

# inicia threads do worker
for _ in range(MAX_CONCURRENT_TELEGRAM):
    t = threading.Thread(target=telegram_worker, daemon=True)
    t.start()

def send_telegram(message, priority=2):
    """
    Adiciona mensagem à fila de envio.
    priority=1 => COMPRA/VENDA
    priority=2 => NEUTRO
    """
    telegram_queue.put((priority, message))
    print(f"Mensagem adicionada à fila (priority={priority})")

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
    print("ANÁLISE =>\n", analysis_text)

    # prioridade: COMPRA/VENDA = 1, NEUTRO = 2
    priority = 1 if decision in ["COMPRA", "VENDA"] else 2
    send_telegram(analysis_text, priority=priority)

# ========================
# WEBSOCKET DERIV
# ========================
def on_message(ws, message):
    try:
        with _lock:
            data = json.loads(message)
            if 'tick' not in data:
                return
            tick = data['tick']
            ativo = tick.get('symbol')
            if ativo not in ativos:
                return

            tick_price = float(tick.get('quote'))
            tick_time = int(tick.get('epoch'))

            if PRINT_EVERY_TICK:
                print(f"[TICK] {ativo} price={tick_price} epoch={tick_time}")

            if current_candle_time[ativo] is None:
                current_candle_time[ativo] = tick_time - (tick_time % 300)
                while tick_time >= current_candle_time[ativo] + 300:
                    current_candle_time[ativo] += 300

            if tick_time >= current_candle_time[ativo] + 300:
                while tick_time >= current_candle_time[ativo] + 300:
                    candle_ticks = ticks_por_ativo[ativo]
                    if candle_ticks:
                        candle = {
                            'time': current_candle_time[ativo],
                            'open': candle_ticks[0],
                            'high': max(candle_ticks),
                            'low': min(candle_ticks),
                            'close': candle_ticks[-1]
                        }
                        candles_por_ativo[ativo].append(candle)
                        if len(candles_por_ativo[ativo]) > 500:
                            candles_por_ativo[ativo].pop(0)
                        print(f"[{ativo}] Candle fechado: O={candle['open']} H={candle['high']} L={candle['low']} C={candle['close']}")
                        df = pd.DataFrame(candles_por_ativo[ativo])
                        try:
                            df = calculate_indicators(df)
                        except Exception as e:
                            print("Erro ao calcular indicadores:", e)
                        generate_and_notify(df, ativo)
                    else:
                        print(f"[{ativo}] Sem ticks no período {current_candle_time[ativo]} ( candle vazio )")
                    ticks_por_ativo[ativo] = []
                    current_candle_time[ativo] += 300

            ticks_por_ativo[ativo].append(tick_price)
            if len(ticks_por_ativo[ativo]) % PRINT_TICK_SUMMARY_EVERY == 0:
                print(f"[{ativo}] ticks buffer: {len(ticks_por_ativo[ativo])} last={tick_price}")

    except Exception:
        print("Erro em on_message:\n", traceback.format_exc())

def on_error(ws, error):
    print("Erro WebSocket:", error)

def on_close(ws, close_status_code, close_msg):
    print("Conexão fechada:", close_status_code, close_msg)

def on_open(ws):
    print("Conexão WebSocket aberta - assinando ativos...")
    for ativo in ativos:
        ws.send(json.dumps({"ticks": ativo, "subscribe": 1}))
        print("Inscrito em:", ativo)

def run_ws_forever():
    backoff = 1
    while True:
        try:
            print("Conectando ao WebSocket da Deriv...")
            ws = websocket.WebSocketApp(
                DERIV_WEBSOCKET_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception:
            print("Exceção em run_ws_forever:\n", traceback.format_exc())
        print(f"Reconectando em {backoff}s...")
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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    print("Iniciando servidor Flask + WebSocket Deriv (principal)")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    run_ws_forever()



