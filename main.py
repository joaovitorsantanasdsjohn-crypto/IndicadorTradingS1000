import os
import threading
import time
import ssl
import io
import requests
import pandas as pd
import numpy as np
from flask import Flask
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ===================== CONFIGURAÇÕES TELEGRAM =====================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"❌ Erro ao enviar mensagem: {e}")

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
    if len(df) < 11:
        return None
    data = df["Close"].values[-11:-1]
    data = data.reshape((1, 10, 1))
    pred = modelo.predict(data, verbose=0)
    return pred[0][0]

# ===================== DOWNLOAD ROBUSTO =====================
def baixar_dados(ativo, tentativas=3):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ativo}?interval=15m&range=1d"
    headers = {"User-Agent": "Mozilla/5.0"}
    for i in range(tentativas):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and len(r.text) > 100:
                df = pd.read_csv(io.StringIO(r.text))
                if "Close" not in df.columns:
                    raise ValueError("Sem coluna Close")
                df["Datetime"] = pd.to_datetime(df["Date"])
                df.set_index("Datetime", inplace=True)
                df = df.dropna()
                print(f"✅ {ativo}: dados recebidos ({len(df)} candles)")
                return df
            else:
                print(f"⚠️ {ativo}: resposta vazia ({r.status_code})")
        except Exception as e:
            print(f"❌ Erro ao baixar {ativo} (tentativa {i+1}): {e}")
        time.sleep(2)
    print(f"🚫 Falha total ao baixar {ativo}")
    return pd.DataFrame()

# ===================== ANÁLISE E SINAIS =====================
def analisar_e_enviar_sinais():
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
            pred_close = prever_proximo_candle(df)
            if pred_close is None:
                continue

            # ===================== LÓGICA DE SINAIS =====================
            sinal = None

            # Tendência de alta (EMA20 > EMA50)
            if ema20 > ema50 and rsi < 40 and pred_close > ema20 and close < lower:
                sinal = f"🔵 COMPRA prevista em {ativo} | RSI: {rsi:.2f} | EMA20>EMA50"

            # Tendência de baixa (EMA20 < EMA50)
            elif ema20 < ema50 and rsi > 60 and pred_close < ema20 and close > upper:
                sinal = f"🔴 VENDA prevista em {ativo} | RSI: {rsi:.2f} | EMA20<EMA50"

            if sinal:
                print(f"📡 Enviando sinal: {sinal}")
                send_telegram_message(sinal)
            else:
                print(f"⏸ Nenhum sinal válido em {ativo}.")

        print("\n⏳ Aguardando 15 minutos para nova análise...\n")
        time.sleep(900)  # 15 minutos

# ===================== FLASK APP =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Bot de Trading com ML, EMA, RSI e Bollinger ativo (15m) rodando!"

# ===================== THREAD PRINCIPAL =====================
if __name__ == "__main__":
    print("🚀 Iniciando robô de trading com EMA e download direto do Yahoo...")
    t = threading.Thread(target=analisar_e_enviar_sinais, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
