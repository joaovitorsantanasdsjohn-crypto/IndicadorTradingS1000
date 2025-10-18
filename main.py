#!/usr/bin/env python3
# main.py - Indicador Trading S1000
# Rodar com Python >=3.10 (ajuste conforme sua infra). Use variáveis de ambiente para token/chat.

import os
import threading
import time
import io
import logging
from logging.handlers import TimedRotatingFileHandler
import random
import json
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from flask import Flask
import yfinance as yf

# ----------------- CONFIGURAÇÕES (ajustáveis) -----------------
INTERVAL = "15m"
PERIOD = "1d"
ATIVOS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X",
    "AUDUSD=X","USDCAD=X","NZDUSD=X","EURGBP=X",
    "EURJPY=X","GBPJPY=X","AUDJPY=X","AUDCAD=X",
    "GBPCHF=X","EURCHF=X","USDMXN=X"
]

# Critérios rígidos (você pode ajustar)
RSI_BUY_THRESHOLD = 40.0   # RSI < esse => condição de sobrevenda para compra
RSI_SELL_THRESHOLD = 60.0  # RSI > esse => condição de sobrecompra para venda

# Tempo entre varreduras (em segundos) - 15 minutos
SCAN_INTERVAL = 15 * 60

# Resumo por Telegram a cada X minutos
SUMMARY_INTERVAL_MINUTES = 30

# ----------------- VARIÁVEIS DE AMBIENTE (NÃO COMITAR TOKEN) -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
PORT = int(os.getenv("PORT", 5000))

# ----------------- LOGGING -----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("indicador_s1000")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler(f"{LOG_DIR}/indicador.log", when="midnight", backupCount=7, utc=True)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# Also log to console
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# In-memory history (for summaries)
signals_sent = []        # tuples (timestamp, ativo, sinal_text)
errors_logged = []       # tuples (timestamp, ativo, error_text)

# ----------------- TENTAR IMPORTAR TENSORFLOW (ML opcional) -----------------
TF_OK = False
modelo = None
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    TF_OK = True
    # criar modelo muito leve (apenas para demonstração)
    def criar_modelo_pequeno():
        m = Sequential()
        # Usaremos Dense simples em vez de LSTM para reduzir chance de erro em ambiente restrito
        m.add(Dense(16, activation="relu", input_shape=(10, 1)))
        m.add(Dense(1, activation="linear"))
        m.compile(optimizer="adam", loss="mse")
        return m
    modelo = criar_modelo_pequeno()
    logger.info("✅ TensorFlow disponível — modelo leve inicializado.")
except Exception as e:
    logger.warning(f"⚠️ TensorFlow não disponível ou falhou import: {e}. O bot rodará sem ML.")

# ----------------- Função de envio Telegram -----------------
def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.error("TELEGRAM_TOKEN ou CHAT_ID não configurados. Mensagem não será enviada.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            logger.warning(f"Telegram returned {r.status_code}: {r.text}")
            return False
        return True
    except Exception as e:
        logger.exception(f"Erro ao enviar Telegram: {e}")
        return False

# ----------------- Indicadores -----------------
def calcular_indicadores(df):
    df = df.copy()
    df["EMA8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD"] = df["Close"].rolling(20).std()
    df["Upper"] = df["MA20"] + 2 * df["STD"]
    df["Lower"] = df["MA20"] - 2 * df["STD"]

    return df.dropna()

# ----------------- Previsão (usando modelo leve quando disponível) -----------------
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
        logger.exception(f"Erro na previsão ML: {e}")
        return None

# ----------------- DOWNLOAD ROBUSTO -----------------
def tentar_yfinance(ativo):
    try:
        # threads=False evita paralelismo que às vezes causa problemas em ambientes serverless
        df = yf.download(ativo, period=PERIOD, interval=INTERVAL, progress=False, threads=False)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        # captura erros de yfinance
        logger.debug(f"yfinance erro para {ativo}: {e}")
    return pd.DataFrame()

def fallback_csv(ativo):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ativo}?range={PERIOD}&interval={INTERVAL}&events=history"
    headers = {"User-Agent": random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64)"
    ])}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code == 200 and len(r.text) > 50:
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
            df = df.set_index(pd.to_datetime(df["Date"]))
            # manter colunas com nomes esperados
            # renomear caso necessário
            return df
    except Exception as e:
        logger.debug(f"fallback_csv erro para {ativo}: {e}")
    return pd.DataFrame()

def baixar_dados(ativo, tentativas=3):
    backoff = 1
    for i in range(1, tentativas + 1):
        logger.info(f"🔁 Tentativa {i}/{tentativas} para {ativo} via yfinance...")
        df = tentar_yfinance(ativo)
        if not df.empty:
            logger.info(f"✅ {ativo} obtido via yfinance ({len(df)} candles)")
            return df
        logger.info(f"⚠️ yfinance falhou para {ativo}. Tentando fallback CSV...")
        df = fallback_csv(ativo)
        if not df.empty:
            logger.info(f"✅ {ativo} obtido via CSV fallback ({len(df)} candles)")
            return df
        # se receber 429 da Yahoo, aumentar backoff
        logger.info(f"⏳ Aguardando {backoff}s antes da próxima tentativa para {ativo}...")
        time.sleep(backoff)
        backoff *= 2
    logger.error(f"🚫 Falha total ao baixar {ativo} após {tentativas} tentativas.")
    errors_logged.append((datetime.utcnow(), ativo, "Falha download"))
    return pd.DataFrame()

# ----------------- Geração de sinal (condições rígidas) -----------------
historico_sinais = {}  # evita repetição idêntica por ativo+intervalo

def gerar_sinal_para_df(df, ativo):
    try:
        df_ind = calcular_indicadores(df)
    except Exception as e:
        logger.exception(f"Erro ao calcular indicadores para {ativo}: {e}")
        errors_logged.append((datetime.utcnow(), ativo, f"calc_indicator:{e}"))
        return None

    close = float(df_ind["Close"].iloc[-1])
    ema8 = float(df_ind["EMA8"].iloc[-1])
    ema20 = float(df_ind["EMA20"].iloc[-1])
    ema50 = float(df_ind["EMA50"].iloc[-1])
    upper = float(df_ind["Upper"].iloc[-1])
    lower = float(df_ind["Lower"].iloc[-1])
    rsi = float(df_ind["RSI"].iloc[-1])
    pred = prever_proximo_candle(df_ind)

    sinal = None
    # Condições estritas — concordância entre EMAs (trend), RSI e posição em relação às bandas
    # COMPRA: EMAs alinhadas (curto > médio > longo), RSI em zona de compra (baixo),
    # previsão (se houver) indicando subida, e preço perto/inferior da banda -> sinal conservador
    if (ema8 > ema20 > ema50) and (rsi < RSI_BUY_THRESHOLD) and (pred is not None and pred > ema8) and (close <= lower * 1.002):
        sinal = f"🔵 COMPRA prevista | {ativo} | RSI {rsi:.1f} | close {close:.5f}"
    # VENDA: EMAs alinhadas de baixa, RSI alto, previsão indicando queda, e preço perto/superior da banda
    elif (ema8 < ema20 < ema50) and (rsi > RSI_SELL_THRESHOLD) and (pred is not None and pred < ema8) and (close >= upper * 0.998):
        sinal = f"🔴 VENDA prevista | {ativo} | RSI {rsi:.1f} | close {close:.5f}"

    if sinal:
        chave = f"{ativo}_{INTERVAL}"
        # evita enviar sinal idêntico repetido
        if historico_sinais.get(chave) == sinal:
            return None
        historico_sinais[chave] = sinal
        # guarda histórico pra resumo
        signals_sent.append((datetime.utcnow(), ativo, sinal))
        logger.info(f"📡 SINAL GERADO: {sinal}")
        return sinal
    else:
        logger.debug(f"Sem sinal para {ativo} (RSI={rsi:.1f}, close={close:.5f})")
    return None

# ----------------- Loop principal -----------------
def loop_bot():
    logger.info("🚀 Bot iniciado: loop de análise ativo")
    next_summary = datetime.utcnow() + timedelta(minutes=SUMMARY_INTERVAL_MINUTES)
    while True:
        for ativo in ATIVOS:
            logger.info(f"📥 Baixando dados de {ativo}...")
            df = baixar_dados(ativo, tentativas=3)
            if df.empty:
                logger.warning(f"Dados vazios para {ativo}, pulando...")
                continue
            try:
                sinal = gerar_sinal_para_df(df, ativo)
                if sinal:
                    send_telegram_message(sinal)
            except Exception as e:
                logger.exception(f"Erro processando {ativo}: {e}")
                errors_logged.append((datetime.utcnow(), ativo, str(e)))
            # pequeno sleep entre ativos para reduzir taxa de requisições
            time.sleep(1 + random.random() * 0.5)

        # Enviar resumo se chegou o horário
        if datetime.utcnow() >= next_summary:
            enviar_resumo()
            next_summary = datetime.utcnow() + timedelta(minutes=SUMMARY_INTERVAL_MINUTES)

        logger.info(f"⏳ Aguardando {SCAN_INTERVAL} segundos para próxima varredura...")
        time.sleep(SCAN_INTERVAL)

# ----------------- Resumo (a cada X minutos) -----------------
def enviar_resumo():
    # montar resumo dos últimos SUMMARY_INTERVAL_MINUTES
    agora = datetime.utcnow()
    janela = agora - timedelta(minutes=SUMMARY_INTERVAL_MINUTES)
    sinais_ult = [s for s in signals_sent if s[0] >= janela]
    erros_ult = [e for e in errors_logged if e[0] >= janela]

    texto = f"📊 Resumo {agora.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
    texto += f"- Período: últimos {SUMMARY_INTERVAL_MINUTES} minutos\n"
    texto += f"- Sinais enviados: {len(sinais_ult)}\n"
    if len(sinais_ult) > 0:
        for t,a,s in sinais_ult[-5:]:
            texto += f"  • {t.strftime('%H:%M')} {a} -> {s.split('|')[0].strip()}\n"
    texto += f"- Erros recentes: {len(erros_ult)}\n"
    if len(erros_ult) > 0:
        for t,a,e in erros_ult[-5:]:
            texto += f"  • {t.strftime('%H:%M')} {a} -> {e}\n"

    # enviar via Telegram e logar localmente
    logger.info("Enviando resumo por Telegram...")
    ok = send_telegram_message(texto)
    if ok:
        logger.info("✅ Resumo enviado.")
    else:
        logger.warning("❌ Falha ao enviar resumo por Telegram.")

# ----------------- FLASK para keepalive -----------------
app = Flask("indicador_s1000")

@app.route("/")
def home():
    return "Indicador Trading S1000 ativo. (Keepalive ping)"

# ----------------- Start threads -----------------
def start():
    t = threading.Thread(target=loop_bot, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    logger.info("Inicializando Indicador Trading S1000...")
    start()
