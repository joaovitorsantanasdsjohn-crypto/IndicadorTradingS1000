import yfinance as yf
import pandas as pd
import time
import os
from flask import Flask
from threading import Thread
import requests

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
    df["ATR"] = (df["High"] - df["Low"]).rolling(window=14).mean()
    df["MACD"], df["MACD_SIGNAL"] = calcular_macd(df["Close"])
    return df

# ============================
# FUNÇÕES DE SINAIS
# ============================

def obter_sinal(df, timeframe, ativo, confirmacao=None):
    # Última vela fechada (penúltima linha do dataframe)
    ultimo = df.iloc[-2]  

    close = float(ultimo["Close"])
    ema9 = float(ultimo["EMA9"])
    ema21 = float(ultimo["EMA21"])
    rsi = float(ultimo["RSI"])
    upper = float(ultimo["Upper"])
    lower = float(ultimo["Lower"])
    atr = float(ultimo["ATR"])
    macd = float(ultimo["MACD"])
    macd_signal = float(ultimo["MACD_SIGNAL"])

    sinal = None

    # Filtro 1: ATR acima da média -> volatilidade suficiente
    atr_medio = df["ATR"].mean()
    if atr < atr_medio:
        return None

    # Filtro 2: RSI rigoroso
    if ema9 > ema21 and close <= lower*1.01 and rsi < 30 and macd > macd_signal:
        sinal = "CALL"
    elif ema9 < ema21 and close >= upper*0.99 and rsi > 70 and macd < macd_signal:
        sinal = "PUT"

    # Filtro 3: confirmação multi-timeframe
    if sinal and confirmacao and confirmacao != sinal:
        return None

    # Filtro 4: evitar duplicados/contraditórios
    chave = f"{ativo}_{timeframe}"
    ultimo_sinal = historico_sinais.get(chave)
    if ultimo_sinal == sinal or (ultimo_sinal and sinal and ultimo_sinal != sinal):
        return None

    if sinal:
        historico_sinais[chave] = sinal

    return sinal

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
        print(" INDICADOR TRADING S1000 - MÁXIMA ASSERTIVIDADE ")
        print("="*60)

        for ativo in ativos:
            try:
                # Baixar dados dos 2 timeframes
                df5 = yf.download(ativo, period="2d", interval="5m", progress=False)
                df15 = yf.download(ativo, period="5d", interval="15m", progress=False)

                if df5.empty or df15.empty:
                    continue

                df5 = indicadores(df5)
                df15 = indicadores(df15)

                # Sinal principal no 5m
                sinal_15m = obter_sinal(df15, "15m", ativo)
                sinal_5m = obter_sinal(df5, "5m", ativo, confirmacao=sinal_15m)

                if sinal_5m:
                    mensagem = f"{ativo} | 5M confirmado pelo 15M | {sinal_5m}"
                    print(mensagem)
                    enviar_telegram(mensagem)

            except Exception as e:
                print(f"Erro ao processar {ativo}: {e}")
                continue

        print("Próxima análise em 60 segundos...")
        time.sleep(60)

# ============================
# FLASK (KEEP-ALIVE)
# ============================

app = Flask("bot")

@app.route("/")
def home():
    return "Bot do João rodando com máxima assertividade!"

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
