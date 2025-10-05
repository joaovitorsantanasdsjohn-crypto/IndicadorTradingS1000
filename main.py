import os
import threading
import time
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

# ===================== BOT DE TRADING =====================
ativos = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "AUDCAD=X",
    "GBPCHF=X", "EURCHF=X", "USDMXN=X"
]

# ===================== MODELO ML =====================
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

# ===================== FUNÇÃO DE PREVISÃO =====================
def prever_proximo_candle(df):
    # usar últimos 10 fechamentos para prever próximo fechamento
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
                df = yf.download(ativo, period="1d", interval="15m")
                if df.empty:
                    continue

                df = calcular_indicadores(df)
                close = df["Close"].iloc[-1]
                ema = df["EMA"].iloc[-1]
                upper = df["Upper"].iloc[-1]
                lower = df["Lower"].iloc[-1]
                rsi = df["RSI"].iloc[-1]

                # previsão do próximo candle
                pred_close = prever_proximo_candle(df)
                if pred_close is None:
                    continue

                sinal = None
                # sinal baseado no candle previsto e indicadores
                if rsi < 40 and pred_close < lower and pred_close > ema:
                    sinal = f"🔵 COMPRA prevista em {ativo} | RSI: {rsi:.2f}"
                elif rsi > 60 and pred_close > upper and pred_close < ema:
                    sinal = f"🔴 VENDA prevista em {ativo} | RSI: {rsi:.2f}"

                if sinal:
                    send_telegram_message(sinal)

            except Exception as e:
                print(f"Erro em {ativo}: {e}")

        time.sleep(900)  # aguarda 15 minutos para o próximo candle

# ===================== FLASK APP PARA MANTER PORTA ABERTA =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Bot de Trading com ML e previsão de candle 15m rodando!"

# ===================== THREAD BOT + FLASK =====================
if __name__ == "__main__":
    t = threading.Thread(target=analisar_e_enviar_sinais, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

