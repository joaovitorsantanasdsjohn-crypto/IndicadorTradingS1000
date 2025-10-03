import yfinance as yf
import pandas as pd
import time
import os
from flask import Flask
from threading import Thread
import requests

# ============================
# CONFIGURAÇÕES BOT TRADING
# ============================

ativos = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "CHFJPY=X", "EURGBP=X",
    "EURCAD=X", "AUDJPY=X", "USDCHF=X", "EURNZD=X", "CADJPY=X"
]

timeframes = ["5m", "15m"]

# Telegram
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
TELEGRAM_CHAT_ID = "6370166264"

# Histórico de sinais para evitar duplicados e contraditórios
historico_sinais = {}

# ============================
# FUNÇÕES DE INDICADORES
# ============================

def calcular_rsi(series, period=14):
    delta = series.diff()
    ganho = delta.clip(lower=0)
    perda = -1*delta.clip(upper=0)
    media_ganho = ganho.rolling(window=period).mean()
    media_perda = perda.rolling(window=period).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))

def calcular_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def indicadores(df):
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["RSI"] = calcular_rsi(df["Close"], 14)
    df["Upper"] = df["Close"].rolling(window=20).mean() + 2.5*df["Close"].rolling(window=20).std()  # Bollinger mais criteriosa
    df["Lower"] = df["Close"].rolling(window=20).mean() - 2.5*df["Close"].rolling(window=20).std()
    df["ATR"] = df["High"].rolling(5).max() - df["Low"].rolling(5).min()
    df["MACD"], df["MACD_SIGNAL"] = calcular_macd(df["Close"])
    return df

# ============================
# FUNÇÕES DE SINAIS
# ============================

def obter_sinal(df, timeframe, ativo):
    ultimo = df.iloc[-1]
    close = float(ultimo["Close"])
    ema9 = float(ultimo["EMA9"])
    ema21 = float(ultimo["EMA21"])
    rsi = float(ultimo["RSI"])
    upper = float(ultimo["Upper"])
    lower = float(ultimo["Lower"])
    atr = float(ultimo["ATR"].tail(5).mean())
    macd = float(ultimo["MACD"])
    macd_signal = float(ultimo["MACD_SIGNAL"])

    sinal = None

    # Critério de volatilidade mínima (ATR)
    if atr < (0.0007 if "USD" in ativo else 0.05):
        return None

    # Critério de confiança máxima: todos os indicadores devem concordar
    if ema9 > ema21 and close <= lower*1.005 and rsi < 50 and macd > macd_signal:
        sinal = "CALL"
    elif ema9 < ema21 and close >= upper*0.995 and rsi > 50 and macd < macd_signal:
        sinal = "PUT"

    # Verificar duplicados e contraditórios
    chave = f"{ativo}_{timeframe}"
    ultimo_sinal = historico_sinais.get(chave)

    if ultimo_sinal == sinal or (ultimo_sinal and sinal and ultimo_sinal != sinal):
        return None

    if sinal:
        historico_sinais[chave] = sinal

    return sinal

# ============================
# ENVIO DE SINAIS PARA TELEGRAM
# ============================

def enviar_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem}
    try:
        requests.post(url, data=payload)
    except:
        pass

# ============================
# LOOP PRINCIPAL DO BOT
# ============================

def executar_bot():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("="*60)
        print("       INDICADOR TRADING DO JOÃO (MÁXIMA ASSERTIVIDADE)")
        print("="*60)

        for ativo in ativos:
            sinais_timeframes = []
            for timeframe in timeframes:
                try:
                    df = yf.download(ativo, period="2d", interval=timeframe, progress=False)
                    if df.empty:
                        continue
                    df = indicadores(df)
                    sinal = obter_sinal(df, timeframe, ativo)
                    sinais_timeframes.append(sinal)
                except Exception as e:
                    print(f"Erro ao processar {ativo} {timeframe}: {e}")
                    continue

            # Envia sinal apenas se os timeframes concordarem
            if sinais_timeframes.count(sinais_timeframes[0]) == len(timeframes) and sinais_timeframes[0]:
                mensagem = f"{ativo} | {timeframes[0].upper()}+{timeframes[1].upper()} | {sinais_timeframes[0]}"
                print(mensagem)
                enviar_telegram(mensagem)

        print("Próxima análise em 60 segundos...")
        time.sleep(60)

# ============================
# FLASK (KEEP-ALIVE)
# ============================

app = Flask("bot")

@app.route("/")
def home():
    return "Bot do João rodando!"

@app.route("/ping")
def ping():
    return "pong"

def run_server():
    app.run(host="0.0.0.0", port=8080)

# ============================
# RODAR BOT + FLASK
# ============================

if __name__ == "__main__":
    t = Thread(target=run_server)
    t.start()
    executar_bot()
