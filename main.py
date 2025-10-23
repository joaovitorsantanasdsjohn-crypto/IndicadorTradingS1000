import websocket
import json
import pandas as pd
import numpy as np
from telegram import Bot
from ml_model import SignalFilter
from flask import Flask
import threading
import time
import traceback

# ========================
# CONFIGURAÇÕES
# ========================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

DERIV_WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

# 15 pares de moedas
ativos = [
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD",
    "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP", "frxAUDJPY",
    "frxEURCHF", "frxUSDCHF", "frxGBPCHF", "frxAUDNZD", "frxNZDJPY"
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

# ========================
# FUNÇÕES AUXILIARES
# ========================
def send_telegram(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
        print("Telegram enviado:", message)
    except Exception as e:
        print("Erro ao enviar Telegram:", e)

def calculate_indicators(df):
    import ta
    if df.empty or len(df) < 5:
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

# ========================
# FUNÇÃO DE ANÁLISE E SINAL
# ========================
def generate_signal(df, ativo):
    try:
        if df.empty or len(df) < 21:
            return

        last = df.iloc[-1]
        features = [
            float(last.get('EMA_short', np.nan)),
            float(last.get('EMA_medium', np.nan)),
            float(last.get('EMA_long', np.nan)),
            float(last.get('RSI', np.nan)),
            float(last.get('BB_upper', np.nan)),
            float(last.get('BB_lower', np.nan))
        ]

        prob = ml_filter.predict(features)
        direction = "NEUTRO"
        emoji = "⚪"

        # Lógica principal de sinal
        if last['EMA_short'] > last['EMA_medium'] > last['EMA_long'] and last['RSI'] < 70 and last['close'] > last['BB_lower'] and prob > 0.6:
            direction = "COMPRA"
            emoji = "📈"
        elif last['EMA_short'] < last['EMA_medium'] < last['EMA_long'] and last['RSI'] > 30 and last['close'] < last['BB_upper'] and prob > 0.6:
            direction = "VENDA"
            emoji = "📉"

        # Cria o texto da análise completa
        msg = (
            f"{emoji} *Análise {ativo}*\n\n"
            f"EMA5: `{last['EMA_short']:.5f}`\n"
            f"EMA13: `{last['EMA_medium']:.5f}`\n"
            f"EMA21: `{last['EMA_long']:.5f}`\n"
            f"RSI: `{last['RSI']:.2f}`\n"
            f"Bollinger Superior: `{last['BB_upper']:.5f}`\n"
            f"Bollinger Inferior: `{last['BB_lower']:.5f}`\n"
            f"Probabilidade ML: `{prob:.2f}`\n\n"
            f"*Sinal:* {direction}"
        )

        send_telegram(msg)

    except Exception:
        print("Erro em generate_signal:\n", traceback.format_exc())

# ========================
# WEBSOCKET HANDLERS
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
                        df = calculate_indicators(df)
                        generate_signal(df, ativo)
                    else:
                        print(f"[{ativo}] Sem ticks no período {current_candle_time[ativo]}")
                    ticks_por_ativo[ativo] = []
                    current_candle_time[ativo] += 300

            ticks_por_ativo[ativo].append(tick_price)
            if len(ticks_por_ativo[ativo]) % 10 == 0:
                print(f"[{ativo}] ticks buffer: {len(ticks_por_ativo[ativo])} última cotação: {tick_price}")
    except Exception:
        print("Erro em on_message:\n", traceback.format_exc())

def on_error(ws, error):
    print("Erro WebSocket:", error)

def on_close(ws, close_status_code, close_msg):
    print("Conexão fechada:", close_status_code, close_msg)

def on_open(ws):
    print("Conexão WebSocket aberta - assinando ativos...")
    try:
        for ativo in ativos:
            ws.send(json.dumps({"ticks": ativo, "subscribe": 1}))
            print("Inscrito em:", ativo)
    except Exception:
        print("Erro em on_open:\n", traceback.format_exc())

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
# FLASK (para uptime)
# ========================
app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 IndicadorTradingS1000 ativo e rodando!"

def run_flask():
    app.run(host="0.0.0.0", port=5000)

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    print("Iniciando servidor Flask + WebSocket Deriv")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    run_ws_forever()
