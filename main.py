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
    """Envia mensagens para o Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            print("✅ Sinal enviado com sucesso para o Telegram.")
        else:
            print(f"⚠️ Erro ao enviar sinal. Código: {r.status_code}")
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# ===================== CONFIGURAÇÃO SSL =====================
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
    """Calcula EMA, RSI e Bandas de Bollinger."""
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
    """Prevê o próximo fechamento com base nas últimas 10 velas."""
    if len(df) < 11:
        return None
    data = df["Close"].values[-11:-1]
    data = data.reshape((1, 10, 1))
    pred = modelo.predict(data, verbose=0)
    return pred[0][0]

# ===================== DOWNLOAD ROBUSTO =====================
def baixar_dados(ativo, tentativas=3):
    """Baixa dados do Yahoo Finance de forma resiliente."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for i in range(tentativas):
        try:
            df = yf.download(
                ativo,
                period="1d",
                interval="15m",
                progress=False,
                threads=False,
                headers=headers,
            )
            if df is not None and not df.empty and "Close" in df.columns:
                return df
            else:
                print(f"⚠️ {ativo}: resposta vazia na tentativa {i+1}")
        except Exception as e:
            print(f"⚠️ Erro ao baixar {ativo} (tentativa {i+1}): {e}")
        time.sleep(2)
    print(f"🚫 Falha total ao baixar {ativo}. Pulando ativo.")
    return pd.DataFrame()

# ===================== ANÁLISE E SINAIS =====================
def analisar_e_enviar_sinais():
    """Analisa os ativos e envia sinais de compra/venda."""
    while True:
        for ativo in ativos:
            print(f"\n📥 Baixando dados de {ativo}...")
            df = baixar_dados(ativo)
            if df.empty:
                continue

            df = calcular_indicadores(df)
            close = df["Close"].iloc[-1]
            ema20 = df["EMA20"].iloc[-1]
            ema50 = df["EMA50"].iloc[-1]
            upper = df["Upper"].iloc[-1]
            lower = df["Lower"].iloc[-1]
            rsi = df["RSI"].iloc[-1]
            tendencia_alta = ema20 > ema50
            tendencia_baixa = ema20 < ema50

            pred_close = prever_proximo_candle(df)
            if pred_close is None:
                continue

            sinal = None

            # ===================== CONDIÇÕES DE COMPRA =====================
            if (
                rsi < 40 and 
                pred_close < lower and 
                tendencia_alta and 
                close > ema20
            ):
                sinal = (
                    f"🔵 COMPRA em {ativo}\n"
                    f"Tendência: Alta (EMA20>{ema50})\n"
                    f"RSI: {rsi:.2f}\n"
                    f"Preço: {close:.5f}\n"
                    f"Previsão próxima vela: {pred_close:.5f}"
                )

            # ===================== CONDIÇÕES DE VENDA =====================
            elif (
                rsi > 60 and 
                pred_close > upper and 
                tendencia_baixa and 
                close < ema20
            ):
                sinal = (
                    f"🔴 VENDA em {ativo}\n"
                    f"Tendência: Baixa (EMA20<{ema50})\n"
                    f"RSI: {rsi:.2f}\n"
                    f"Preço: {close:.5f}\n"
                    f"Previsão próxima vela: {pred_close:.5f}"
                )

            if sinal:
                print(f"📡 Enviando sinal: {sinal}")
                send_telegram_message(sinal)

        print("\n⏳ Aguardando 15 minutos para nova análise...")
        time.sleep(900)

# ===================== FLASK APP =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Bot de Trading com RSI + Bollinger + EMA rodando (15m)!"

# ===================== THREAD PRINCIPAL =====================
if __name__ == "__main__":
    t = threading.Thread(target=analisar_e_enviar_sinais, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
