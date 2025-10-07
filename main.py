import os
import threading
import time
import ssl
import yfinance as yf
import pandas as pd
import requests
from flask import Flask
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

# ===================== CONFIGURAÇÕES DO TELEGRAM =====================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# ===================== CONFIGURAÇÃO DO SSL (RENDER) =====================
ssl._create_default_https_context = ssl._create_unverified_context

# ===================== LISTA DE ATIVOS =====================
ativos = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "AUDCAD=X",
    "GBPCHF=X", "EURCHF=X", "USDMXN=X"
]

# ===================== MODELO LSTM =====================
def criar_modelo():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(10, 1)))
    model.add(LSTM(50))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    return model

modelo = criar_modelo()

# ===================== INDICADORES =====================
def calcular_indicadores(df):
    df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    delta = df["Close"].diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    media_ganho = ganho.rolling(14).mean()
    media_perda = perda.rolling(14).mean()
    rs = media_ganho / media_perda
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD"] = df["Close"].rolling(20).std()
    df["Upper"] = df["MA20"] + (df["STD"] * 2)
    df["Lower"] = df["MA20"] - (df["STD"] * 2)
    return df

# ===================== PREVISÃO =====================
def prever_proximo_candle(df):
    if len(df) < 11:
        return None
    data = df["Close"].values[-11:-1]
    data = data.reshape((1, 10, 1))
    pred = modelo.predict(data, verbose=0)
    return pred[0][0]

# ===================== ANÁLISE E SINAIS =====================
def analisar_e_enviar_sinais():
    while True:
        for ativo in ativos:
            try:
                print(f"📥 Baixando dados de {ativo}...")
                df = yf.download(
                    ativo,
                    period="1d",
                    interval="15m",
                    progress=False,
                    threads=False
                )

                if df.empty:
                    print(f"⚠️ Dados vazios para {ativo}, pulando...")
                    continue

                df = calcular_indicadores(df)
                close = df["Close"].iloc[-1]
                ema = df["EMA"].iloc[-1]
                upper = df["Upper"].iloc[-1]
                lower = df["Lower"].iloc[-1]
                rsi = df["RSI"].iloc[-1]

                pred_close = prever_proximo_candle(df)
                if pred_close is None:
                    continue

                sinal = None
                if rsi < 40 and pred_close < lower
