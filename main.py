# main.py - versão robusta de download (yfinance + JSON fallback) + indicadores + TF opcional
import os
import threading
import time
import ssl
import io
import traceback
import requests
import pandas as pd
import numpy as np
from flask import Flask
import yfinance as yf

# Tentativa de importar TensorFlow (se houver problema: continuamos sem ML)
TF_OK = True
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
except Exception as e:
    print(f"⚠️ TensorFlow import falhou: {e}", flush=True)
    TF_OK = False

# ===================== CONFIG TELEGRAM =====================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}", flush=True)

# ===================== SSL / CERTIFI =====================
# ajuda com certificados em alguns ambientes (Render)
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except Exception:
    pass
# evita bloquear por causa de SSL problems (apenas como fallback)
ssl._create_default_https_context = ssl._create_unverified_context

# ===================== NÃO use backend/caches problemáticos do yfinance =====================
# NÃO definir set_backend(...) nem set_tz_cache_location(None) — gerou TypeError nos logs.
# Se quiser forçar cache, usamos /tmp, mas por padrão não mexemos.
try:
    os.environ["YFINANCE_CACHE_DIR"] = "/tmp"
    # opcional (apenas se disponível)
    try:
        yf.set_tz_cache_location("/tmp")
    except Exception:
        pass
except Exception:
    pass

# ===================== ATIVOS =====================
ativos = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "AUDUSD=X", "USDCAD=X", "USDCHF=X",
    "AUDJPY=X", "EURJPY=X", "GBPJPY=X",
    "NZDUSD=X", "EURGBP=X", "AUDCAD=X",
    "GBPCHF=X", "EURCHF=X", "USDMXN=X"
]

INTERVAL = "15m"
PERIOD = "1d"

# ===================== Modelo (TF pequeno opcional) =====================
modelo = None
if TF_OK:
    try:
        def criar_modelo_pequeno():
            m = Sequential()
            m.add(Dense(16, activation="relu", input_shape=(10, 1)))
            m.add(Dense(1, activation="linear"))
            m.compile(optimizer="adam", loss="mse")
            return m
        modelo = criar_modelo_pequeno()
        print("✅ Modelo TensorFlow leve inicializado.", flush=True)
    except Exception as e:
        print(f"⚠️ Falha ao inicializar modelo TF: {e}", flush=True)
        modelo = None

# ===================== Sessão requests com Retry =====================
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET","POST"])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36",
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
})

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

# ===================== PREVISÃO (se houver modelo) =====================
def prever_proximo_candle(df):
    if modelo is None:
        return None
    if len(df) < 11:
        return None
    try:
        arr = df["Close"].values[-11:-1].reshape(1, 10, 1)
        pred = modelo.predict(arr, verbose=0)
        return float(pred[0][0])
    except Exception as e:
        print(f"Erro na previsão ML: {e}", flush=True)
        return None

# ===================== Fallback: JSON endpoint do Yahoo (mais confiável que CSV em muitos casos) =====================
def baixar_json_yahoo(ativo):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ativo}?range={PERIOD}&interval={INTERVAL}"
        r = session.get(url, timeout=12)
        if r.status_code != 200:
            # tenta query2
            url2 = f"https://query2.finance.yahoo.com/v8/finance/chart/{ativo}?range={PERIOD}&interval={INTERVAL}"
            r = session.get(url2, timeout=12)
        if r.status_code == 200:
            j = r.json()
            if j.get("chart") and j["chart"].get("result"):
                res = j["chart"]["result"][0]
                ts = res.get("timestamp")
                indicators = res.get("indicators", {}).get("quote", [])
                if not ts or not indicators:
                    return pd.DataFrame()
                q = indicators[0]
                df = pd.DataFrame({
                    "Open": q.get("open"),
                    "High": q.get("high"),
                    "Low": q.get("low"),
                    "Close": q.get("close"),
                    "Volume": q.get("volume")
                }, index=pd.to_datetime(ts, unit="s"))
                # drop rows with None closes
                df = df.dropna(subset=["Close"])
                return df
    except Exception as e:
        print(f"JSON fallback erro para {ativo}: {e}", flush=True)
        # mostra trace curta
        print(traceback.format_exc(limit=1), flush=True)
    return pd.DataFrame()

# ===================== Fallback CSV (último recurso) =====================
def baixar_csv_fallback(ativo):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ativo}?range={PERIOD}&interval={INTERVAL}&events=history"
    try:
        r = session.get(url, timeout=12)
        if r.status_code == 200 and len(r.text) > 100:
            df = pd.read_csv(io.StringIO(r.text))
            if "Date" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Date"])
                df.set_index("Datetime", inplace=True)
                # normaliza colunas para o formato esperado
                expected = ["Open","High","Low","Close","Adj Close","Volume"]
                for c in expected:
                    if c not in df.columns:
                        df[c] = np.nan
                return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    except Exception as e:
        print(f"CSV fallback erro para {ativo}: {e}", flush=True)
        print(traceback.format_exc(limit=1), flush=True)
    return pd.DataFrame()

# ===================== TENTATIVAS: yfinance -> JSON -> CSV =====================
def baixar_dados_robusto(ativo, tentativas=3):
    for tentativa in range(1, tentativas + 1):
        try:
            print(f"🔁 Tentativa {tentativa}/{tentativas} para {ativo} via yfinance...", flush=True)
            df = yf.download(ativo, period=PERIOD, interval=INTERVAL, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"✅ {ativo} via yfinance ({len(df)} candles)", flush=True)
                return df
            else:
                print(f"⚠️ yfinance retornou vazio para {ativo}.", flush=True)
        except Exception as e:
            print(f"yfinance erro para {ativo}: {e}", flush=True)
            print(traceback.format_exc(limit=1), flush=True)

        # tenta JSON chart API
        print(f"⚠️ Tentando fallback JSON para {ativo}...", flush=True)
        df = baixar_json_yahoo(ativo)
        if not df.empty:
            print(f"✅ {ativo} via JSON fallback ({len(df)} candles)", flush=True)
            return df

        # tenta CSV fallback (última alternativa)
        print(f"⚠️ Tentando fallback CSV para {ativo}...", flush=True)
        df = baixar_csv_fallback(ativo)
        if not df.empty:
            print(f"✅ {ativo} via CSV fallback ({len(df)} candles)", flush=True)
            return df

        time.sleep(1 + tentativa * 0.5)

    print(f"🚫 Falha total ao baixar {ativo} após {tentativas} tentativas.", flush=True)
    return pd.DataFrame()

# ===================== EVITA SINAIS DUPLICADOS =====================
historico_sinais = {}

def gerar_sinal(df, ativo):
    df = calcular_indicadores(df)
    close = float(df["Close"].iloc[-1])
    ema8 = float(df["EMA8"].iloc[-1])
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    upper = float(df["Upper"].iloc[-1])
    lower = float(df["Lower"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])
    pred = prever_proximo_candle(df)

    sinal = None
    # critério estrito
    if ema8 > ema20 > ema50 and rsi < 40 and pred is not None and pred > ema8 and close < lower:
        sinal = f"🔵 COMPRA prevista | {ativo} | RSI {rsi:.1f}"
    elif ema8 < ema20 < ema50 and rsi > 60 and pred is not None and pred < ema8 and close > upper:
        sinal = f"🔴 VENDA prevista | {ativo} | RSI {rsi:.1f}"

    chave = f"{ativo}_{INTERVAL}"
    ultimo = historico_sinais.get(chave)
    if sinal and ultimo == sinal:
        return None
    if sinal:
        historico_sinais[chave] = sinal
    return sinal

# ===================== LOOP PRINCIPAL =====================
def loop_bot():
    print("🚀 Bot iniciado: loop de análise ativo", flush=True)
    while True:
        for ativo in ativos:
            print(f"\n📥 Baixando {ativo} ...", flush=True)
            df = baixar_dados_robusto(ativo, tentativas=3)
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
                print(traceback.format_exc(limit=1), flush=True)
            time.sleep(0.3)  # evita burst entre tickers
        print("\n⏳ Aguardando 15 minutos para próxima varredura...\n", flush=True)
        time.sleep(15 * 60)

# ===================== FLASK KEEPALIVE =====================
app = Flask("mayim_bot")

@app.route("/")
def home():
    return "Mayim bot online"

def start_threads_and_flask():
    t = threading.Thread(target=loop_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 0)) or 5000
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_threads_and_flask()
