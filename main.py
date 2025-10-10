# ===============================================
# INDICADOR TRADING S1000
# ===============================================
import os, io, ssl, time, threading, json, requests, datetime as dt
import pandas as pd, numpy as np
import yfinance as yf
from flask import Flask

# TensorFlow (modelo leve)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ================== TELEGRAM =====================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

def telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception as e:
        print("Telegram erro:", e)

# ================== CONFIG =====================
ativos = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X"]
INTERVAL = "15m"
PERIOD = "1d"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
LOG_PATH = os.path.join(CACHE_DIR, "logs.csv")

# ================== SSL FIX =====================
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

yf.set_backend("requests")

# ================== ML MODEL =====================
def criar_modelo():
    m = Sequential()
    m.add(Dense(32, activation="relu", input_shape=(5,)))
    m.add(Dense(16, activation="relu"))
    m.add(Dense(1, activation="tanh"))  # saída -1 a 1
    m.compile(optimizer="adam", loss="mse")
    return m

modelo = criar_modelo()

# ================== INDICADORES =====================
def indicadores(df):
    df = df.copy()
    df["EMA8"]  = df["Close"].ewm(span=8).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD"]  = df["Close"].rolling(20).std()
    df["Upper"] = df["MA20"] + 2 * df["STD"]
    df["Lower"] = df["MA20"] - 2 * df["STD"]
    return df.dropna()

# ================== CACHE + DOWNLOAD =====================
def cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol.replace('=','_')}.csv")

def baixar_dados(ativo):
    path = cache_path(ativo)
    # tenta cache
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        if time.time() - mtime < 900:  # 15min
            try:
                return pd.read_csv(path, index_col=0, parse_dates=True)
            except: pass
    try:
        df = yf.download(ativo, period=PERIOD, interval=INTERVAL, progress=False, threads=False)
        if not df.empty:
            df.to_csv(path)
            return df
    except Exception as e:
        print(f"yfinance erro {ativo}:", e)
    return pd.DataFrame()

# ================== GERAÇÃO DE SINAL =====================
historico = {}
logs = []

def gerar_sinal(df, ativo):
    df = indicadores(df)
    if len(df) < 25: return None
    row = df.iloc[-1]
    features = np.array([
        row["RSI"], row["EMA8"], row["EMA20"],
        row["Upper"], row["Lower"]
    ]).reshape(1, -1)
    pred = float(modelo.predict(features, verbose=0)[0][0])

    close = row["Close"]
    ema8, ema20, ema50 = row["EMA8"], row["EMA20"], row["EMA50"]
    rsi, upper, lower = row["RSI"], row["Upper"], row["Lower"]

    sinal = None
    if ema8 > ema20 > ema50 and rsi < 40 and close < lower and pred > 0:
        sinal = f"🔵 COMPRA | {ativo} | RSI {rsi:.1f}"
    elif ema8 < ema20 < ema50 and rsi > 60 and close > upper and pred < 0:
        sinal = f"🔴 VENDA | {ativo} | RSI {rsi:.1f}"

    chave = f"{ativo}_{INTERVAL}"
    if sinal and historico.get(chave) != sinal:
        historico[chave] = sinal
        logs.append({"hora": dt.datetime.now(), "ativo": ativo, "sinal": sinal})
        return sinal
    return None

# ================== LOOP PRINCIPAL =====================
ultima_limpeza = time.time()
ultima_sintese = time.time()

def loop():
    global ultima_limpeza, ultima_sintese
    print("🚀 Indicador Trading S1000 iniciado.")
    telegram("🚀 Indicador Trading S1000 iniciado.")
    while True:
        for ativo in ativos:
            print(f"📥 {ativo}")
            df = baixar_dados(ativo)
            if df.empty: continue
            try:
                s = gerar_sinal(df, ativo)
                if s:
                    print("📡", s)
                    telegram(s)
                else:
                    print("⏸ Sem sinal para", ativo)
            except Exception as e:
                print("Erro ativo", ativo, ":", e)
        # resumo a cada 30 min
        if time.time() - ultima_sintese > 1800:
            ultima_sintese = time.time()
            if logs:
                resumo = pd.DataFrame(logs).tail(10)
                texto = "🧾 Resumo últimas 10 operações:\n"
                for _,r in resumo.iterrows():
                    texto += f"{r['hora']:%H:%M} | {r['sinal']}\n"
                telegram(texto)
                pd.DataFrame(logs).to_csv(LOG_PATH, index=False)
        # limpeza a cada 24h
        if time.time() - ultima_limpeza > 86400:
            ultima_limpeza = time.time()
            logs.clear()
            open(LOG_PATH,"w").close()
            telegram("🧹 Logs zerados (reset diário).")
        time.sleep(15*60)

# ================== FLASK KEEPALIVE =====================
app = Flask("IndicadorTradingS1000")

@app.route("/")
def home():
    return "Indicador Trading S1000 online"

def start():
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 0)) or 5000
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    start()
