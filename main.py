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

# Histórico de sinais para evitar duplicados e contraditórios
historico_sinais = {}

# Histórico para ML
ml_historico = []

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
# CONFIGURAÇÃO DO MODELO ML
# ============================

ml_model = Sequential([
    Dense(32, input_shape=(6,), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # saída: probabilidade de sinal correto
])
ml_model.compile(optimizer='adam', loss='binary_crossentropy')

def treinar_ml():
    if len(ml_historico) < 10:
        return  # mínimo de dados para treinar
    dados = pd.DataFrame(ml_historico)
    X = dados[["EMA9", "EMA21", "RSI", "MACD", "MACD_SIGNAL", "ATR"]].values
    y = dados["Resultado"].values
    ml_model.fit(X, y, epochs=5, verbose=0)

def avaliar_ml(ind):
    if not ml_historico:
        return True  # sem histórico, aceita sinal
    X = [[ind["EMA9"], ind["EMA21"], ind["RSI"], ind["MACD"], ind["MACD_SIGNAL"], ind["ATR"]]]
    pred = ml_model.predict(X, verbose=0)[0][0]
    return pred > 0.6  # só envia se probabilidade > 60%

# ============================
# FUNÇÕES DE SINAIS
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
    if atr < (0.001 if "USD" in ativo else 0.1):
        return None

    # Critério de confiança máxima
    if (ema9 > ema21 + 0.0003 and close <= lower * 1.005 and rsi < 55 and macd - macd_signal > 0.0001):
        sinal = "CALL"
    elif (ema9 < ema21 - 0.0003 and close >= upper * 0.995 and rsi > 45 and macd_signal - macd > 0.0001):
        sinal = "PUT"

    if not sinal:
        return None

    # Verificar duplicados e contraditórios
    chave = f"{ativo}_{timeframe}"
    ultimo_sinal = historico_sinais.get(chave)
    if ultimo_sinal == sinal or (ultimo_sinal and sinal and ultimo_sinal != sinal):
        return None

    # Avaliar com ML
    indicador_ml = {
        "EMA9": ema9, "EMA21": ema21, "RSI": rsi,
        "MACD": macd, "MACD_SIGNAL": macd_signal, "ATR": atr
    }
    if not avaliar_ml(indicador_ml):
        return None  # filtro ML

    historico_sinais[chave] = sinal
    return sinal, indicador_ml

# ============================
# ENVIO DE SINAIS PARA TELEGRAM
# ============================

def enviar_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem}
    try:
        requests.post(url, data=payload)
    except:
        pass

# ============================
# LOOP PRINCIPAL DO BOT
# ============================

def executar_bot():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("="*60)
        print("       INDICADOR TRADING DO JOÃO (ML ATIVO + MÁXIMA ASSERTIVIDADE)")
        print("="*60)

        for ativo in ativos:
            for timeframe in timeframes:
                try:
                    df = yf.download(ativo, period="2d", interval=timeframe, progress=False)
                    if df.empty:
                        continue
                    df = indicadores(df)
                    resultado = obter_sinal(df, timeframe, ativo)
                    if resultado:
                        sinal, ind_ml = resultado
                        mensagem = f"{ativo} | {timeframe.upper()} | {sinal}"
                        print(mensagem)
                        enviar_telegram(mensagem)
                        # Adiciona ao histórico ML
                        ml_historico.append({**ind_ml, "Resultado": 1})  # por enquanto, assume sinal correto
                        treinar_ml()
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
    return "Bot do João rodando com ML ativo!"

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
