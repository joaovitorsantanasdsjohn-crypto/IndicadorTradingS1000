# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import time
import os
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask
from threading import Thread

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
TELEGRAM_TOKEN = "SEU_TOKEN_AQUI"
TELEGRAM_CHAT_ID = "SEU_CHAT_ID_AQUI"

# Histórico de sinais
historico_sinais = {}

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
# CONFIGURAÇÃO ML / TensorFlow
# ============================

def criar_modelo():
    modelo = Sequential()
    modelo.add(Dense(64, input_dim=5, activation="relu"))  # EMA9, EMA21, RSI, MACD, ATR
    modelo.add(Dense(32, activation="relu"))
    modelo.add(Dense(2, activation="softmax"))  # CALL ou PUT
    modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return modelo

ml_model = criar_modelo()

# ============================
# FUNÇÃO DE SINAIS
# ============================

def obter_sinal(df, timeframe, ativo):
    ultimo = df.iloc[-1]
    features = [
        float(ultimo["EMA9"]),
        float(ultimo["EMA21"]),
        float(ultimo["RSI"]),
        float(ultimo["MACD"]),
        float(ultimo["ATR"].tail(5).mean())
    ]
    
    # Previsão ML
    import numpy as np
    pred = ml_model.predict(np.array([features]), verbose=0)[0]
    sinal_ml = "CALL" if np.argmax(pred) == 0 else "PUT"

    # Critério de máxima assertividade
    close = float(ultimo["Close"])
    upper = float(ultimo["Upper"])
    lower = float(ultimo["Lower"])
    ema9 = float(ultimo["EMA9"])
    ema21 = float(ultimo["EMA21"])
    rsi = float(ultimo["RSI"])
    macd = float(ultimo["MACD"])
    macd_signal = float(ultimo["MACD_SIGNAL"])
    atr = float(ultimo["ATR"].tail(5).mean())

    # Todos indicadores precisam concordar
    sinal = None
    if ema9 > ema21 and close <= lower*1.01 and rsi < 55 and macd > macd_signal and sinal_ml == "CALL":
        sinal = "CALL"
    elif ema9 < ema21 and close >= upper*0.99 and rsi > 45 and macd < macd_signal and sinal_ml == "PUT":
        sinal = "PUT"

    # Evitar duplicados
    chave = f"{ativo}_{timeframe}"
    ultimo_sinal = historico_sinais.get(chave)
    if ultimo_sinal == sinal or (ultimo_sinal and sinal and ultimo_sinal != sinal):
        return None

    if sinal:
        historico_sinais[chave] = sinal
        # Treinar modelo com este sinal
        # Label: [CALL, PUT]
        y_label = [1,0] if sinal=="CALL" else [0,1]
        X = np.array([features])
        Y = np.array([y_label])
        ml_model.fit(X, Y, epochs=1, verbose=0)

    return sinal

# ============================
# ENVIO DE SINAIS TELEGRAM
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
        os.system("cls" if os.name=="nt" else "clear")
        print("="*60)
        print("       INDICADOR TRADING (MAX ASSERTIVIDADE + ML)")
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
                    print(f"Erro ao processar {ativo} {timeframe}: {e}")
                    continue

        print("Próxima análise em 60 segundos...")
        time.sleep(60)

# ============================
# FLASK PARA KEEP-ALIVE
# ============================

app = Flask("bot")

@app.route("/")
def home():
    return "Bot rodando!"

def run_server():
    # Porta padrão do Render: 10000-19999 ou PORT variável de ambiente
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

# ============================
# RODAR BOT + FLASK
# ============================

if __name__=="__main__":
    t = Thread(target=run_server)
    t.start()
    executar_bot()
