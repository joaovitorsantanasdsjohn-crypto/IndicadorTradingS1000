import yfinance as yf
import pandas as pd
import time
import os
from flask import Flask
from threading import Thread
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

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

# Histórico de sinais
historico_sinais = {}

# ============================
# MODELO DE MACHINE LEARNING
# ============================

modelo = Sequential([
    Dense(32, activation="relu", input_shape=(7,)),
    Dense(16, activation="relu"),
    Dense(2, activation="softmax")
])
modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

X_treino = []
y_treino = []

# ============================
# FUNÇÕES DE INDICADORES
# ============================

def calcular_rsi(series, period=14):
    delta = series.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
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
    df["Upper"] = df["Close"].rolling(window=20).mean() + 2*df["Close"].rolling(window=20).std()
    df["Lower"] = df["Close"].rolling(window=20).mean() - 2*df["Close"].rolling(window=20).std()
    df["ATR"] = df["High"] - df["Low"]
    df["MACD"], df["MACD_SIGNAL"] = calcular_macd(df["Close"])
    return df

# ============================
# SINAIS
# ============================

def obter_sinal(df, timeframe, ativo):
    ultimo = df.iloc[-1]
    features = [
        float(ultimo["Close"]),
        float(ultimo["EMA9"]),
        float(ultimo["EMA21"]),
        float(ultimo["RSI"]),
        float(ultimo["Upper"]),
        float(ultimo["Lower"]),
        float(ultimo["MACD"] - ultimo["MACD_SIGNAL"])
    ]

    atr = float(df["ATR"].tail(5).mean())
    if atr < (0.0005 if "USD" in ativo else 0.05):
        return None

    sinal_ml = None
    if X_treino:
        pred = modelo.predict(np.array([features]), verbose=0)
        sinal_ml = "CALL" if np.argmax(pred[0]) == 1 else "PUT"

    ema9, ema21, rsi, close = features[1], features[2], features[3], features[0]
    upper, lower, macd_diff = features[4], features[5], features[6]

    sinal_indicadores = None
    if ema9 > ema21 and close <= lower*1.01 and rsi < 55 and macd_diff > 0:
        sinal_indicadores = "CALL"
    elif ema9 < ema21 and close >= upper*0.99 and rsi > 45 and macd_diff < 0:
        sinal_indicadores = "PUT"

    sinal = None
    if sinal_indicadores and (sinal_ml is None or sinal_ml == sinal_indicadores):
        sinal = sinal_indicadores

    chave = f"{ativo}_{timeframe}"
    ultimo_sinal = historico_sinais.get(chave)
    if ultimo_sinal == sinal or (ultimo_sinal and sinal and ultimo_sinal != sinal):
        return None
    if sinal:
        historico_sinais[chave] = sinal
        X_treino.append(features)
        y_treino.append(1 if sinal == "CALL" else 0)
        if len(X_treino) >= 10:
            modelo.fit(np.array(X_treino), np.array(y_treino), epochs=5, verbose=0)

    return sinal

# ============================
# TELEGRAM
# ============================

def enviar_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem}
    try:
        requests.post(url, data=payload)
    except:
        pass

# ============================
# LOOP PRINCIPAL
# ============================

def executar_bot():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("="*60)
        print("       BOT TRADING JOÃO (ML + Máxima Criteriosidade)")
        print("="*60)

        for ativo in ativos:
            for timeframe in timeframes:
                try:
                    df = yf.download(ativo, period="2d", interval=timeframe, progress=False)
                    if df.empty:
                        continue
                    df = indicadores(df)
                    sinal = obter_sinal(df, timeframe, ativo)
                    if sinal:
                        mensagem = f"{ativo} | {timeframe.upper()} | {sinal}"
                        print(mensagem)
                        enviar_telegram(mensagem)
                except Exception as e:
                    print(f"Erro {ativo} {timeframe}: {e}")
                    continue

        print("Próxima análise em 60 segundos...")
        time.sleep(60)

# ============================
# FLASK (KEEP-ALIVE)
# ============================

app = Flask("bot")

@app.route("/")
def home():
    return "Bot do João rodando com ML!"

@app.route("/ping")
def ping():
    return "pong"

def run_server():
    app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    t = Thread(target=run_server)
    t.start()
    executar_bot()
