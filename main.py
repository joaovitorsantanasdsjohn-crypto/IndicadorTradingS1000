# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ============================
# CONFIGURAÇÕES TELEGRAM
# ============================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
TELEGRAM_CHAT_ID = "6370166264"


# Ativos a monitorar
ATIVOS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]

# Função enviar mensagem
def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

# Função indicadores técnicos
def calcular_indicadores(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["RSI"] = 100 - (100 / (1 + (df["Close"].diff().clip(lower=0).rolling(14).mean() /
                                   df["Close"].diff().clip(upper=0).abs().rolling(14).mean())))

    df["UpperBB"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["LowerBB"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["ATR"] = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
    return df.dropna()

# Modelo simples ML
def criar_modelo(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Inicializa o modelo ML
modelo = criar_modelo(6)

# Loop principal
def main():
    while True:
        for ativo in ATIVOS:
            try:
                df = yf.download(ativo, period="2d", interval="5m")
                df = calcular_indicadores(df)

                ult = df.iloc[-1]
                features = np.array([[ult["EMA20"], ult["EMA50"], ult["RSI"], ult["MACD"],
                                      ult["Signal"], ult["ATR"]]])

                # Predição com ML
                pred = modelo.predict(features, verbose=0)[0][0]

                # Regras adicionais (criteriosas)
                if (ult["EMA20"] > ult["EMA50"]
                    and ult["Close"] > ult["EMA20"]
                    and ult["RSI"] > 55
                    and ult["MACD"] > ult["Signal"]
                    and pred > 0.7):
                    send_telegram_message(f"📈 SINAL COMPRA confirmado: {ativo}")

                elif (ult["EMA20"] < ult["EMA50"]
                      and ult["Close"] < ult["EMA20"]
                      and ult["RSI"] < 45
                      and ult["MACD"] < ult["Signal"]
                      and pred < 0.3):
                    send_telegram_message(f"📉 SINAL VENDA confirmado: {ativo}")

                # Treinamento incremental simples
                resultado = 1 if ult["Close"] > df["Close"].iloc[-2] else 0
                modelo.fit(features, np.array([resultado]), epochs=1, verbose=0)

            except Exception as e:
                print(f"Erro em {ativo}: {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()

