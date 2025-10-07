import os
import threading
import time
import ssl
import random
import yfinance as yf
import pandas as pd
import requests
from flask import Flask
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from json.decoder import JSONDecodeError

# ===================== CONFIGURAÇÕES DO TELEGRAM =====================
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
CHAT_ID = "6370166264"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# ===================== CONFIGURAÇÃO SSL (RENDER) =====================
# (apenas evita erros de verificação em alguns ambientes hospedados)
ssl._create_default_https_context = ssl._create_unverified_context

# ===================== LISTA DE ATIVOS (15 pares) =====================
ativos = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "AUDCAD=X",
    "GBPCHF=X", "EURCHF=X", "USDMXN=X"
]

# ===================== HISTÓRICO SIMPLES DE SINAIS (para evitar duplicados) ====
# chave: ativo ; valor: (sinal, timestamp)
historico_sinais = {}

# tempo mínimo (segundos) entre sinais iguais para o mesmo ativo
MIN_SECONDS_BETWEEN_SAME_SIGNAL = 15 * 60  # 15 minutos

# ===================== MODELO LSTM (pequeno) =====================
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
    df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
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
    # ATR simples (high-low) para filtragem de baixa volatilidade
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    return df

# ===================== PREVISÃO (usando LSTM) =====================
def prever_proximo_candle(df):
    if len(df) < 11:
        return None
    data = df["Close"].values[-11:-1]
    data = data.reshape((1, 10, 1))
    try:
        pred = modelo.predict(data, verbose=0)
        return float(pred[0][0])
    except Exception as e:
        print(f"Erro na previsão do modelo: {e}")
        return None

# ===================== FUNÇÃO DE DOWNLOAD ROBUSTA =====================
def baixar_dados_com_retries(ativo, interval="15m"):
    attempts = 0
    periods = ["1d", "5d", "10d"]  # fallback sequence
    last_exc = None

    while attempts < 5:
        # escolhe period com round-robin: primeiro 1d, depois fallback
        period_choice = periods[0] if attempts < 2 else (periods[1] if attempts < 4 else periods[2])
        try:
            print(f"📥 Tentativa {attempts+1}: baixando {ativo} period={period_choice} interval={interval}")
            df = yf.download(
                tickers=ativo,
                period=period_choice,
                interval=interval,
                progress=False,
                threads=False
            )

            # yfinance pode devolver DataFrame vazio ou com colunas; checar
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df

            # se veio vazio, registrar e tentar novamente com outro period
            print(f"⚠️ yfinance retornou vazio para {ativo} com period={period_choice}")

        except JSONDecodeError as e:
            last_exc = e
            print(f"⚠️ JSONDecodeError ao baixar {ativo}: {e}")
        except Exception as e:
            last_exc = e
            print(f"⚠️ Erro ao baixar {ativo}: {e}")

        attempts += 1
        sleep_for = 1 + attempts * 1.5 + random.random()
        print(f"🔁 Esperando {sleep_for:.1f}s antes da próxima tentativa...")
        time.sleep(sleep_for)

    # após tentativas, retorna DataFrame vazio e loga
    print(f"🚫 Falha ao baixar dados de {ativo} após várias tentativas. Último erro: {last_exc}")
    return pd.DataFrame()

# ===================== LÓGICA DE SINAL (com filtro de duplicados e volatilidade) ====
def deve_enviar_sinal(ativo, novo_sinal):
    info = historico_sinais.get(ativo)
    agora = time.time()

    if info is None:
        return True

    sinal_antigo, ts = info
    # se mesmo sinal e dentro do tempo mínimo, não enviar
    if sinal_antigo == novo_sinal and (agora - ts) < MIN_SECONDS_BETWEEN_SAME_SIGNAL:
        return False

    # se sinal diferente, permitir (mas podemos também impor um pequeno cooldown)
    return True

def registrar_sinal(ativo, sinal):
    historico_sinais[ativo] = (sinal, time.time())

# ===================== LOOP PRINCIPAL DE ANÁLISE E ENVIO =====================
def analisar_e_enviar_sinais():
    while True:
        for ativo in ativos:
            # pequena pausa entre ativos para reduzir pressão no Yahoo
            time.sleep(0.8)

            df = baixar_dados_com_retries(ativo, interval="15m")
            if df.empty:
                # pula ativo se sem dados
                continue

            try:
                df = calcular_indicadores(df)

                # valores mais recentes
                close = float(df["Close"].iloc[-1])
                ema = float(df["EMA"].iloc[-1])
                upper = float(df["Upper"].iloc[-1])
                lower = float(df["Lower"].iloc[-1])
                rsi = float(df["RSI"].iloc[-1])

                # volatilidade mínima (filtra mercados sem movimento)
                atr = df["ATR"].iloc[-1] if "ATR" in df.columns and not df["ATR"].isna().all() else None
                if atr is not None:
                    # thresholds simples: para pares com USD (menor pip), ajuste mais fino
                    if "USD" in ativo and atr < 0.0003:
                        # pouca volatilidade -> pular
                        print(f"⚠️ Baixa volatilidade em {ativo} (ATR {atr:.6f}), pulando...")
                        continue
                    elif "USD" not in ativo and atr < 0.01:
                        print(f"⚠️ Baixa volatilidade em {ativo} (ATR {atr:.4f}), pulando...")
                        continue

                pred_close = prever_proximo_candle(df)
                if pred_close is None:
                    continue

                # definição do sinal com critérios exigentes (tudo em concordância)
                sinal = None
                if (rsi < 40) and (pred_close < lower) and (pred_close > ema):
                    sinal = f"🔵 COMPRA prevista em {ativo} | RSI: {rsi:.2f}"
                elif (rsi > 60) and (pred_close > upper) and (pred_close < ema):
                    sinal = f"🔴 VENDA prevista em {ativo} | RSI: {rsi:.2f}"

                if sinal:
                    if deve_enviar_sinal(ativo, sinal):
                        print(f"📡 Enviando sinal: {sinal}")
                        send_telegram_message(sinal)
                        registrar_sinal(ativo, sinal)
                    else:
                        print(f"⛔ Sinal repetido para {ativo} detectado; não enviado.")

            except Exception as e:
                print(f"❌ Erro ao processar {ativo}: {e}")
                # não interrompe o loop; passa pro próximo ativo
                continue

        print("⏳ Aguardando 15 minutos para nova análise...")
        time.sleep(15 * 60)  # 15 minutos

# ===================== FLASK (keep-alive) =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Bot de Trading com ML e previsão de candle 15m rodando!"

# ===================== RODAR EM THREAD + FLASK =====================
if __name__ == "__main__":
    t = threading.Thread(target=analisar_e_enviar_sinais, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
