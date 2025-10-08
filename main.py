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
    """Envia mensagem no Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# ===================== CONFIGURAÇÃO SSL =====================
ssl._create_default_https_context = ssl._create_unverified_context

# ===================== AJUSTE DO BACKEND YFINANCE =====================
# Isso mantém o download direto do Yahoo Finance mas com compatibilidade no Render
yf.set_tz_cache_location(None)
yf.set_backend("requests")

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
    """Calcula RSI, Bandas de Bollinger e EMAs."""
    df["EMA8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

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
    """Usa o modelo LSTM para prever o próximo fechamento."""
    if len(df) < 11:
        return None
    data = df["Close"].values[-11:-1]
    data = data.reshape((1, 10, 1))
    pred = modelo.predict(data, verbose=0)
    return pred[0][0]

# ===================== DOWNLOAD ROBUSTO =====================
def baixar_dados(ativo, tentativas=3):
    """Baixa os dados de um ativo com até 3 tentativas."""
    for i in range(tentativas):
        try:
            df = yf.download(
                ativo,
                period="1d",
                interval="15m",
                progress=False,
                threads=False,
            )
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ Erro ao baixar {ativo}: {e}")
        print(f"🔁 Tentando novamente ({i+1}/{tentativas}) para {ativo}...")
        time.sleep(2)
    print(f"🚫 Falha total ao baixar {ativo}. Pulando...")
    return pd.DataFrame()

# ===================== ANÁLISE E SINAIS =====================
def analisar_e_enviar_sinais():
    """Analisa ativos, gera previsões e envia sinais."""
    while True:
        for ativo in ativos:
            print(f"📥 Baixando dados de {ativo}...")
            df = baixar_dados(ativo)
            if df.empty:
                continue

            df = calcular_indicadores(df)
            close = df["Close"].iloc[-1]
            ema8 = df["EMA8"].iloc[-1]
            ema20 = df["EMA20"].iloc[-1]
            ema50 = df["EMA50"].iloc[-1]
            upper = df["Upper"].iloc[-1]
            lower = df["Lower"].iloc[-1]
            rsi = df["RSI"].iloc[-1]

            pred_close = prever_proximo_candle(df)
            if pred_close is None:
                continue

            sinal = None
            # Estratégia com EMAs + RSI + Bollinger
            if rsi < 40 and pred_close < lower and ema8 > ema20 > ema50:
                sinal = f"🔵 COMPRA | {ativo} | RSI: {rsi:.2f} | EMA8>EMA20>EMA50"
            elif rsi > 60 and pred_close > upper and ema8 < ema20 < ema50:
                sinal = f"🔴 VENDA | {ativo} | RSI: {rsi:.2f} | EMA8<EMA20<EMA50"

            if sinal:
                print(f"📡 Enviando sinal: {sinal}")
                send_telegram_message(sinal)

        print("⏳ Aguardando 15 minutos para nova análise...")
        time.sleep(900)

# ===================== FLASK APP =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Bot de Trading com ML, EMAs e Bollinger Bands rodando!"

# ===================== THREAD PRINCIPAL =====================
if __name__ == "__main__":
    t = threading.Thread(target=analisar_e_enviar_sinais, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
