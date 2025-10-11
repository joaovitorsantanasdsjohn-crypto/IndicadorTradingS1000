import os
import time
import json
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask
from datetime import datetime, timedelta
import tensorflow as tf

# ========== CONFIGURAÇÕES ==========
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"
LOG_FILE = "indicador_log.txt"
INTERVALO_MINUTOS = 15

# ========== FLASK APP ==========
app = Flask(__name__)

@app.route('/')
def home():
    return "Indicador Trading S1000 ativo ✅"

# ========== FUNÇÕES DE SUPORTE ==========

def enviar_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}")

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

def resetar_log_diario():
    while True:
        time.sleep(86400)
        if os.path.exists(LOG_FILE):
            open(LOG_FILE, "w").close()
            log("🔁 Log diário resetado.")

def resumo_telegram_periodico():
    while True:
        time.sleep(1800)
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    linhas = f.readlines()[-20:]
                resumo = "".join(linhas)
                enviar_telegram(f"📊 Resumo dos últimos sinais:\n\n{resumo}")
        except Exception as e:
            log(f"Erro no resumo periódico: {e}")

# ========== CÁLCULOS DE INDICADORES ==========

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def calcular_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + 2*std, sma - 2*std

# ========== CACHE E ML ==========
cache_dados = {}

def baixar_dados(ticker):
    if ticker in cache_dados and datetime.now() - cache_dados[ticker]["hora"] < timedelta(minutes=15):
        return cache_dados[ticker]["dados"]

    data = yf.download(ticker, period="1d", interval="15m", progress=False)
    cache_dados[ticker] = {"dados": data, "hora": datetime.now()}
    return data

def modelo_ml_previsao(dados):
    if len(dados) < 50:
        return None
    close = dados["Close"].values[-50:].reshape(-1, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(None, 1)),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(close[:-1].reshape(1, -1, 1), close[1:].reshape(1, -1, 1), epochs=2, verbose=0)
    pred = model.predict(close[-10:].reshape(1, -1, 1), verbose=0)
    return float(pred[-1])

# ========== ANÁLISE PRINCIPAL ==========

def analisar_ativos():
    ativos = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X"]
    while True:
        for ativo in ativos:
            try:
                df = baixar_dados(ativo)
                if df.empty:
                    continue

                close = df["Close"]
                rsi = calcular_rsi(close)
                ema = calcular_ema(close)
                bb_sup, bb_inf = calcular_bollinger_bands(close)
                pred_ml = modelo_ml_previsao(df)

                rsi_atual = rsi.iloc[-1]
                preco = close.iloc[-1]
                sinal = None

                if rsi_atual < 30 and preco <= bb_inf.iloc[-1]:
                    sinal = "📈 Possível COMPRA"
                elif rsi_atual > 70 and preco >= bb_sup.iloc[-1]:
                    sinal = "📉 Possível VENDA"

                if sinal:
                    msg = f"{ativo} | RSI: {rsi_atual:.2f} | EMA: {ema.iloc[-1]:.4f}\n{str(sinal)}\nML Previsão: {pred_ml:.4f}"
                    log(msg)
                    enviar_telegram(msg)

            except Exception as e:
                log(f"Erro em {ativo}: {e}")

        log("⏳ Aguardando próxima varredura (15min)...")
        time.sleep(900)

# ========== THREADS ==========
if __name__ == "__main__":
    threading.Thread(target=resetar_log_diario, daemon=True).start()
    threading.Thread(target=resumo_telegram_periodico, daemon=True).start()
    threading.Thread(target=analisar_ativos, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)
