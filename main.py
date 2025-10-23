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

# ========================
# CONFIGURAÇÕES (já com seu token/chat)
# ========================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

DERIV_WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

# 8 pares mais lucrativos que você pediu
ativos = [
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD",
    "frxUSDCAD", "frxEURJPY", "frxGBPJPY", "frxUSDCHF"
]

# ========================
# INICIAIS
# ========================
bot = Bot(token=TELEGRAM_TOKEN)
ml_filter = SignalFilter()  # arquivo ml_model.py no repositório

candles_por_ativo = {ativo: [] for ativo in ativos}
ticks_por_ativo = {ativo: [] for ativo in ativos}
current_candle_time = {ativo: None for ativo in ativos}

_lock = threading.Lock()

# control prints to avoid absolute spam but still show activity
PRINT_EVERY_TICK = True        # se True mostra cada tick (você pediu visibilidade)
PRINT_TICK_SUMMARY_EVERY = 10  # mostra contagem a cada N ticks no buffer

# ========================
# AUXILIARES: indicadores, envio telegram, formatação
# ========================
def send_telegram(message):
    """
    Envia mensagem para o Telegram.
    Mantém logs detalhados para garantir que as mensagens sejam enviadas.
    """
    try:
        print(f"Tentando enviar Telegram: {message}")
        result = bot.send_message(chat_id=CHAT_ID, text=message)
        print(f"✅ Telegram enviado com sucesso. Message ID: {result.message_id}")
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")
        traceback.print_exc()

def calculate_indicators(df):
    """
    Recebe DataFrame com colunas: time, open, high, low, close
    Retorna df com colunas adicionadas: EMA_short, EMA_medium, EMA_long, RSI, BB_upper, BB_lower
    """
    try:
        import ta
    except Exception as e:
        # se pacote não estiver instalado, levanta exceção clara no log
        raise RuntimeError("Pacote 'ta' não encontrado. Adicione 'ta' no requirements.txt.") from e

    if df.empty:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    # queremos pelo menos 21 candles para cálculos; funções do ta lidam com NaN automaticamente
    df['EMA_short'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
    df['EMA_medium'] = ta.trend.EMAIndicator(df['close'], window=13).ema_indicator()
    df['EMA_long'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df

def create_analysis_text(last, prob, ativo):
    """
    Gera o texto da análise (para terminal e telegram) com indicadores e decisão.
    Sempre retorna algo — inclusive se neutro.
    """
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
        else:
            decision = "NEUTRO"

    # Formata mensagem detalhada
    txt = (
        f"{ativo} | time={time_ts}\n"
        f"close={close:.6f} EMA5={ema_s:.6f} EMA13={ema_m:.6f} EMA21={ema_l:.6f}\n"
        f"RSI={rsi:.2f} BB_upper={bb_u:.6f} BB_lower={bb_l:.6f}\n"
        f"ML_prob={prob:.4f} => DECISÃO: {decision}"
    )
    return txt, decision

# ========================
# GERA SINAL / ENVIO (também envia neutros)
# ========================
def generate_and_notify(df, ativo):
    try:
        if df.empty:
            return
        # we require at least 5 candles to compute some indicators; ML uses features whatever available
        # ensure integer/float conversion
        last = df.iloc[-1].to_dict()
        # Prepare features for ML consistent with earlier design:
        features = [
            float(last.get('EMA_short', np.nan)),
            float(last.get('EMA_medium', np.nan)),
            float(last.get('EMA_long', np.nan)),
            float(last.get('RSI', np.nan)),
            float(last.get('BB_upper', np.nan)),
            float(last.get('BB_lower', np.nan))
        ]
        prob = ml_filter.predict(features)
        # Format analysis and send it ALWAYS (even neutro)
        analysis_text, decision = create_analysis_text(last, prob, ativo)
        # Print to terminal
        print("ANÁLISE =>\n", analysis_text)
        # Send telegram with full analysis (every closed candle)
        send_telegram(analysis_text)
    except Exception:
        print("Erro em generate_and_notify:\n", traceback.format_exc())

# ========================
# WEBSOCKET HANDLERS (DERIV)
# ========================
def on_message(ws, message):
    try:
        with _lock:
            data = json.loads(message)
            # Deriv ticks structure: {'tick': {'epoch':..., 'quote':..., 'symbol':...}}
            if 'tick' not in data:
                # Ex.: subscription responses, etc.
                return
            tick = data['tick']
            ativo = tick.get('symbol')
            if ativo not in ativos:
                return

            tick_price = float(tick.get('quote'))
            tick_time = int(tick.get('epoch'))

            # show tick arrival so you see activity
            if PRINT_EVERY_TICK:
                print(f"[TICK] {ativo} price={tick_price} epoch={tick_time}")

            # init candle start aligned to 5-min (300s)
            if current_candle_time[ativo] is None:
                current_candle_time[ativo] = tick_time - (tick_time % 300)
                # if tick_time already beyond multiple candles (catch-up), fast-forward
                while tick_time >= current_candle_time[ativo] + 300:
                    current_candle_time[ativo] += 300

            # If tick belongs to next candle(s), close previous candle(s)
            if tick_time >= current_candle_time[ativo] + 300:
                # can close multiple empty candles if long gap
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
                        # Calcular indicadores e gerar/anunciar análise para candle fechado
                        df = pd.DataFrame(candles_por_ativo[ativo])
                        try:
                            df = calculate_indicators(df)
                        except Exception as e:
                            print("Erro ao calcular indicadores:", e)
                        generate_and_notify(df, ativo)
                    else:
                        print(f"[{ativo}] Sem ticks no período {current_candle_time[ativo]} ( candle vazio )")
                    # reset tick buffer and advance candle time
                    ticks_por_ativo[ativo] = []
                    current_candle_time[ativo] += 300

            # append current tick to buffer for current candle
            ticks_por_ativo[ativo].append(tick_price)

            # print summary every N ticks to show buffer growth
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
    try:
        for ativo in ativos:
            sub = {"ticks": ativo, "subscribe": 1}
            ws.send(json.dumps(sub))
            print("Inscrito em:", ativo)
    except Exception:
        print("Erro em on_open:\n", traceback.format_exc())

# ========================
# run_ws_forever: loop de conexão com backoff
# ========================
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
            # ping_interval/timeout ajudam a detectar desconexões rapidamente
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception:
            print("Exceção em run_ws_forever:\n", traceback.format_exc())
        # reconexão com backoff exponencial (1s,2s,4s,..até 60s)
        print(f"Reconectando em {backoff}s...")
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)

# ========================
# FLASK para uptime (thread)
# ========================
app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 IndicadorTradingS1000 ativo e rodando!"

def run_flask():
    # debug off; porta 5000
    app.run(host="0.0.0.0", port=5000)

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    print("Iniciando servidor Flask + WebSocket Deriv (principal)")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    # run ws in main thread (principal)
    run_ws_forever()

