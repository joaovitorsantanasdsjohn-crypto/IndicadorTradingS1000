# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
from threading import Thread
from flask import Flask
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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
    df["Upper"] = df["Close"].rolling(window=20).mean() + 2*df["Close"].rolling(window=20).std()
    df["Lower"] = df["Close"].rolling(window=20).mean() - 2*df["Close"].rolling(window=20).std()
    df["ATR"] = df["High"] - df["Low"]
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
    if atr < (0.0005 if "USD" in ativo else 0.05):
        return None

    # Critério máximo: todos os indicadores devem concordar
    if ema9 > ema21 and close <= lower*1.01 and rsi < 55 and macd > macd_signal:
        sinal = "CALL"
    elif ema9 < ema21 and close >= upper*0.99 and rsi > 45 and macd < macd_signal:
        sinal = "PUT"

    chave = f"{ativo}_{timeframe}"
    ultimo_sinal = historico_sinais.get(chave)

    if ultimo_sinal == sinal or (ultimo_sinal and sinal and ultimo_sinal != sinal):
        return None

    if sinal:
        historico_sinais[chave] = sinal

    return sinal

# ============================
# FUNÇÃO TELEGRAM
# ============================

def enviar_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem}
    try:
        requests.post(url, data=payload)
    except:
        pass

# ============================
# MODELO ML (TENSORFLOW)
# ============================

# Modelo simples feed-forward para aprendizado de sinais
ml_model = Sequential([
    Dense(64, input_shape=(6,), activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
ml_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

def treinar_modelo(features, alvo):
    X = np.array(features)
    y = np.array(alvo)
    ml_model.fit(X, y, epochs=5, verbose=0)

def prever_sinal(features):
    X = np.array([features])
    pred = ml_model.predict(X, verbose=0)[0][0]
    return "CALL" if pred > 0.5 else "PUT"

# ============================
# LOOP PRINCIPAL
# ============================

def executar_bot():
    while True:
        print("="*60)
        print("       INDICADOR TRADING COM ML DO JOÃO")
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
                        features = [
                            float(df["EMA9"].iloc[-1]),
                            float(df["EMA21"].iloc[-1]),
                            float(df["RSI"].iloc[-1]),
                            float(df["Upper"].iloc[-1]),
                            float(df["Lower"].iloc[-1]),
                            float(df["MACD"].iloc[-1])
                        ]
                        # Treina o modelo com o sinal atual
                        treinar_modelo(features, [1 if sinal=="CALL" else 0])
                        # Previsão via ML
                        sinal_ml = prever_sinal(features)
                        mensagem = f"{ativo} | {timeframe.upper()} | {sinal_ml}"
                        print(mensagem)
                        enviar_telegram(mensagem)
                except Exception as e:
                    print(f"Erro em {ativo} {timeframe}: {e}")
                    continue

        print("Próxima análise em 60 segundos...")
        time.sleep(60)

# ============================
# FLASK PARA MANUTENÇÃO / KEEP-ALIVE
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
    Thread(target=run_server).start()
    executar_bot()
