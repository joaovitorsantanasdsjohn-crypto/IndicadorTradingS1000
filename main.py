# main.py
import os
import threading
import time
import ssl
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask

# ===================== CONFIG TELEGRAM =====================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}", flush=True)

# ===================== SSL FIX =====================
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except Exception:
    pass
ssl._create_default_https_context = ssl._create_unverified_context

# ===================== YFINANCE FIX =====================
try:
    yf.set_tz_cache_location(None)
    yf.set_backend("requests")
except Exception as e:
    print(f"Aviso yfinance backend: {e}", flush=True)

# ===================== ATIVOS =====================
ativos = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"]
INTERVAL = "15m"
PERIOD = "1d"

# ===================== MODELO TENSORFLOW =====================
TF_OK = True
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
except Exception as e:
    print(f"⚠️ TensorFlow não disponível: {e}", flush=True)
    TF_OK = False

modelo = None
if TF_OK:
    try:
        def criar_modelo_LSTM():
            model = Sequential()
            model.add(LSTM(32, input_shape=(10, 1), activation="tanh"))
            model.add(Dense(16, activation="relu"))
            model.add(Dense(1, activation="linear"))
            model.compile(optimizer="adam", loss="mse")
            return model

        modelo = criar_modelo_LSTM()
        print("✅ Modelo TensorFlow (LSTM) inicializado.", flush=True)
    except Exception as e:
        print(f"⚠️ Falha ao criar modelo LSTM: {e}", flush=True)
        modelo = None

# ===================== INDICADORES =====================
def calcular_indicadores(df):
    df = df.copy()
    df["EMA8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(14).mean()
    media_perda = perda.rolling(14).mean()
    rs = media_ganho / media_perda.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD"] = df["Close"].rolling(20).std()
    df["Upper"] = df["MA20"] + 2 * df["STD"]
    df["Lower"] = df["MA20"] - 2 * df["STD"]

    return df.dropna()

# ===================== TREINO RÁPIDO DO MODELO =====================
def treinar_modelo_rapido(df):
    """Treina o modelo LSTM leve com os últimos dados."""
    global modelo
    if modelo is None or len(df) < 20:
        return
    try:
        X, y = [], []
        closes = df["Close"].values
        for i in range(len(closes) - 11):
            X.append(closes[i:i+10])
            y.append(closes[i+10])
        X = np.array(X).reshape(-1, 10, 1)
        y = np.array(y)
        modelo.fit(X, y, epochs=2, batch_size=4, verbose=0)
    except Exception as e:
        print(f"⚠️ Erro no treino do modelo: {e}", flush=True)

def prever_proximo_candle(df):
    if modelo is None or len(df) < 11:
        return None
    try:
        ultimos = df["Close"].values[-11:-1].reshape(1, 10, 1)
        pred = modelo.predict(ultimos, verbose=0)
        return float(pred[0][0])
    except Exception as e:
        print(f"Erro na previsão: {e}", flush=True)
        return None

# ===================== DOWNLOAD ROBUSTO =====================
def baixar_com_yf(ativo):
    try:
        df = yf.download(ativo, period=PERIOD, interval=INTERVAL, progress=False, threads=False)
        if not df.empty:
            return df
    except Exception as e:
        print(f"yfinance erro para {ativo}: {e}", flush=True)
    return pd.DataFrame()

def baixar_csv_fallback(ativo):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ativo}?range={PERIOD}&interval={INTERVAL}&events=history"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200 and len(r.text) > 100:
            df = pd.read_csv(io.StringIO(r.text))
            if "Date" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Date"])
                df.set_index("Datetime", inplace=True)
                return df
    except Exception as e:
        print(f"CSV fallback erro para {ativo}: {e}", flush=True)
    return pd.DataFrame()

def baixar_dados_robusto(ativo, tentativas=3):
    for i in range(1, tentativas + 1):
        print(f"🔁 Tentativa {i}/{tentativas} para {ativo}", flush=True)
        df = baixar_com_yf(ativo)
        if not df.empty:
            print(f"✅ {ativo} via yfinance ({len(df)} candles)", flush=True)
            return df
        print(f"⚠️ yfinance falhou para {ativo}, tentando fallback...", flush=True)
        df = baixar_csv_fallback(ativo)
        if not df.empty:
            print(f"✅ {ativo} via CSV fallback ({len(df)} candles)", flush=True)
            return df
        time.sleep(2 + i)
    print(f"🚫 Falha total ao baixar {ativo}", flush=True)
    return pd.DataFrame()

# ===================== GERAR SINAIS =====================
historico_sinais = {}

def gerar_sinal(df, ativo):
    df = calcular_indicadores(df)
    treinar_modelo_rapido(df)

    close = df["Close"].iloc[-1]
    ema8 = df["EMA8"].iloc[-1]
    ema20 = df["EMA20"].iloc[-1]
    ema50 = df["EMA50"].iloc[-1]
    upper = df["Upper"].iloc[-1]
    lower = df["Lower"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    pred = prever_proximo_candle(df)

    sinal = None
    if ema8 > ema20 > ema50 and rsi < 40 and close < lower and pred and pred > ema8:
        sinal = f"🔵 COMPRA prevista | {ativo} | RSI {rsi:.1f}"
    elif ema8 < ema20 < ema50 and rsi > 60 and close > upper and pred and pred < ema8:
        sinal = f"🔴 VENDA prevista | {ativo} | RSI {rsi:.1f}"

    chave = f"{ativo}_{INTERVAL}"
    if sinal and historico_sinais.get(chave) != sinal:
        historico_sinais[chave] = sinal
        return sinal
    return None

# ===================== LOOP PRINCIPAL =====================
def loop_bot():
    print("🚀 Bot iniciado (com ML ativo)", flush=True)
    while True:
        for ativo in ativos:
            print(f"\n📥 Baixando {ativo} ...", flush=True)
            df = baixar_dados_robusto(ativo)
            if df.empty:
                continue
            try:
                sinal = gerar_sinal(df, ativo)
                if sinal:
                    print(f"📡 Enviando: {sinal}", flush=True)
                    send_telegram_message(sinal)
                else:
                    print(f"⏸ Sem sinal para {ativo}", flush=True)
            except Exception as e:
                print(f"Erro processando {ativo}: {e}", flush=True)
        print("\n⏳ Aguardando 15 minutos para próxima varredura...\n", flush=True)
        time.sleep(15 * 60)

# ===================== FLASK (Keep Alive) =====================
app = Flask("mayim_bot")

@app.route("/")
def home():
    return "Mayim Bot com ML está online!"

def start_threads_and_flask():
    thread = threading.Thread(target=loop_bot, daemon=True)
    thread.start()
    port = int(os.environ.get("PORT", 0)) or 5000
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_threads_and_flask()
