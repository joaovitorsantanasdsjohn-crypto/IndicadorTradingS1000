import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask
import tensorflow as tf
import requests

# ===============================================================
# 🔧 CONFIGURAÇÕES INICIAIS
# ===============================================================
# Desativa cache local do yfinance (necessário para Render)
os.environ["YFINANCE_CACHE_DIR"] = "/tmp"
yf.set_tz_cache_location("/tmp")

# Configuração do Flask (mantém o Render ativo)
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot de Análise Ativo no Render!"

# ===============================================================
# ⚙️ PARÂMETROS GERAIS
# ===============================================================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"
PARES = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"]
INTERVALO = "15m"
PERIODO = "1d"

# ===============================================================
# 📊 FUNÇÃO PARA BAIXAR DADOS (SEM CACHE)
# ===============================================================
def baixar_dados(ticker):
    for tentativa in range(3):
        try:
            df = yf.download(
                ticker,
                period=PERIODO,
                interval=INTERVALO,
                progress=False,
                threads=False
            )
            if not df.empty:
                return df
        except Exception as e:
            print(f"❌ Erro ao baixar {ticker} (tentativa {tentativa+1}/3): {e}")
        time.sleep(3)
    print(f"🚫 Falha total ao baixar {ticker}.")
    return None

# ===============================================================
# 📈 INDICADORES TÉCNICOS (RSI, EMAs, BANDAS)
# ===============================================================
def calcular_indicadores(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = calcular_rsi(df["Close"], 14)
    df["Bollinger_MA"] = df["Close"].rolling(window=20).mean()
    df["Bollinger_Upper"] = df["Bollinger_MA"] + (df["Close"].rolling(window=20).std() * 2)
    df["Bollinger_Lower"] = df["Bollinger_MA"] - (df["Close"].rolling(window=20).std() * 2)
    return df

def calcular_rsi(precos, periodo=14):
    delta = precos.diff()
    ganho = np.where(delta > 0, delta, 0)
    perda = np.where(delta < 0, -delta, 0)
    media_ganho = pd.Series(ganho).rolling(periodo).mean()
    media_perda = pd.Series(perda).rolling(periodo).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))

# ===============================================================
# 🤖 PREVISÃO SIMPLES COM TENSORFLOW (ML)
# ===============================================================
def prever_tendencia(df):
    # Apenas os últimos 50 candles
    dados = df[["Close"]].values[-50:]
    if len(dados) < 50:
        return None

    modelo = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(50, 1)),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    modelo.compile(optimizer='adam', loss='mse')

    # Normalização
    dados_norm = (dados - np.min(dados)) / (np.max(dados) - np.min(dados))
    X = np.expand_dims(dados_norm, axis=0)
    y_pred = modelo.predict(X, verbose=0)
    return float(y_pred[0][0])

# ===============================================================
# 💬 ENVIO DE SINAIS PARA TELEGRAM
# ===============================================================
def enviar_telegram(mensagem):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem}
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# ===============================================================
# 🔍 ANÁLISE FINAL E GERAÇÃO DE SINAL
# ===============================================================
def analisar_par(ticker):
    df = baixar_dados(ticker)
    if df is None or df.empty:
        return

    df = calcular_indicadores(df)
    pred = prever_tendencia(df)
    if pred is None:
        return

    ultimo = df.iloc[-1]
    preco = ultimo["Close"]
    ema20 = ultimo["EMA20"]
    ema50 = ultimo["EMA50"]
    rsi = ultimo["RSI"]
    sup = ultimo["Bollinger_Lower"]
    res = ultimo["Bollinger_Upper"]

    direcao = None

    # 📊 Estratégia de decisão
    if preco < ema20 < ema50 and rsi < 35 and preco <= sup:
        direcao = "🔵 COMPRA"
    elif preco > ema20 > ema50 and rsi > 65 and preco >= res:
        direcao = "🔴 VENDA"

    if direcao:
        msg = (
            f"📈 Par: {ticker.replace('=X', '')}\n"
            f"💰 Sinal: {direcao}\n"
            f"RSI: {rsi:.2f}\n"
            f"EMA20: {ema20:.5f}\n"
            f"EMA50: {ema50:.5f}\n"
            f"Fechamento: {preco:.5f}"
        )
        print(msg)
        enviar_telegram(msg)
    else:
        print(f"Sem sinal para {ticker}")

# ===============================================================
# 🔁 LOOP PRINCIPAL
# ===============================================================
def main():
    while True:
        for par in PARES:
            print(f"\n📥 Analisando {par} ...")
            analisar_par(par)
        print("⏳ Aguardando 15 minutos para próxima varredura...\n")
        time.sleep(900)

# ===============================================================
# 🚀 EXECUÇÃO
# ===============================================================
if __name__ == '__main__':
    import threading
    threading.Thread(target=main).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
