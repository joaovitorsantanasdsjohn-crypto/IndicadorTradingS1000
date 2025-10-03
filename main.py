import yfinance as yf
import pandas as pd
import time
import os
from flask import Flask
from threading import Thread
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

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
historico_ml_file = "signals_history.csv"

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
# FUNÇÃO PARA OBTER SINAL
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

    # Critério de confiança máxima
    if ema9 > ema21 and close <= lower*1.01 and rsi < 65 and macd > macd_signal:
        sinal = "CALL"
    elif ema9 < ema21 and close >= upper*0.99 and rsi > 35 and macd < macd_signal:
        sinal = "PUT"

    # Evitar duplicados e contraditórios
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
# FUNÇÕES DE ML
# ============================

# Carregar histórico ou criar novo
if os.path.exists(historico_ml_file):
    df_ml = pd.read_csv(historico_ml_file)
else:
    df_ml = pd.DataFrame(columns=["ativo","timeframe","EMA9","EMA21","RSI","Upper","Lower","ATR","MACD","MACD_SIGNAL","sinal","resultado"])

def treinar_modelo():
    if len(df_ml) < 10:
        return None  # poucos dados para treinar
    X = df_ml[["EMA9","EMA21","RSI","Upper","Lower","ATR","MACD","MACD_SIGNAL"]].values
    y = df_ml["resultado"].apply(lambda x: 1 if x=="CORRETO" else 0).values
    model = Sequential([
        Dense(32, input_dim=X.shape[1], activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, verbose=0)
    return model

def filtrar_sinal_ml(model, df, sinal):
    ultimo = df.iloc[-1]
    X_pred = np.array([[ultimo["EMA9"],ultimo["EMA21"],ultimo["RSI"],ultimo["Upper"],ultimo["Lower"],ultimo["ATR"],ultimo["MACD"],ultimo["MACD_SIGNAL"]]])
    prob = model.predict(X_pred, verbose=0)[0][0]
    if prob > 0.6:  # só envia se a probabilidade de acerto > 60%
        return True
    return False

# ============================
# LOOP PRINCIPAL DO BOT
# ============================

def executar_bot():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("="*60)
        print("       INDICADOR TRADING DO JOÃO (ML ATIVO)")
        print("="*60)

        model = treinar_modelo()

        for ativo in ativos:
            for timeframe in timeframes:
                try:
                    df = yf.download(ativo, period="2d", interval=timeframe, progress=False)
                    if df.empty:
                        continue
                    df = indicadores(df)
                    sinal = obter_sinal(df, timeframe, ativo)
                    if sinal:
                        enviar = True
                        if model:
                            enviar = filtrar_sinal_ml(model, df, sinal)
                        if enviar:
                            mensagem = f"{ativo} | {timeframe.upper()} | {sinal}"
                            print(mensagem)
                            enviar_telegram(mensagem)
                            # Atualizar histórico ML
                            novo = {
                                "ativo": ativo, "timeframe": timeframe,
                                "EMA9": df["EMA9"].iloc[-1],
                                "EMA21": df["EMA21"].iloc[-1],
                                "RSI": df["RSI"].iloc[-1],
                                "Upper": df["Upper"].iloc[-1],
                                "Lower": df["Lower"].iloc[-1],
                                "ATR": df["ATR"].iloc[-1],
                                "MACD": df["MACD"].iloc[-1],
                                "MACD_SIGNAL": df["MACD_SIGNAL"].iloc[-1],
                                "sinal": sinal,
                                "resultado": "CORRETO"
                            }
                            df_ml.loc[len(df_ml)] = novo
                            df_ml.to_csv(historico_ml_file, index=False)
                except Exception as e:
                    print(f"Erro ao processar {ativo} {timeframe}: {e}")
                    continue

        print("Próxima análise em 60 segundos...")
        time.sleep(60)

# ============================
# FLASK (KEEP-ALIVE)
# ============================

app = Flask("bot")

@app.route("/")
def home():
    return "Bot do João rodando com ML!"

@app.route("/ping")
def ping():
    return "pong"

def run_server():
    app.run(host="0.0.0.0", port=8080)

# ============================
# RODAR BOT + FLASK
# ============================

if __name__ == "__main__":
    t = Thread(target=run_server)
    t.start()
    executar_bot()
