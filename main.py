# main.py  (versão otimizada para Render free + ML opcional)
import os
import time
import gc
import traceback
from threading import Thread

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from flask import Flask

# Tentativa de importar TensorFlow; se não tiver, o bot continuará em modo só-regras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# resource para checar uso de memória (Unix)
try:
    import resource
    RESOURCE_AVAILABLE = True
except Exception:
    RESOURCE_AVAILABLE = False

# ============================
# CONFIGURAÇÕES (pode ajustar via ENV)
# ============================
# ativos (pode reduzir a lista se precisar economizar mais memória)
ATIVOS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "CHFJPY=X", "EURGBP=X",
    "EURCAD=X", "AUDJPY=X", "USDCHF=X", "EURNZD=X", "CADJPY=X"
]

# quantos ativos processar por ciclo (reduz uso de memória) — padrão 5
ASSETS_PER_CYCLE = int(os.getenv("ASSETS_PER_CYCLE", "5"))

# timeframes
TF_SHORT = "5m"
TF_LONG = "15m"

# Telegram: prefira setar como variáveis de ambiente no Render
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6370166264")

# ML: habilite definindo ENABLE_ML=true no ambiente (por padrão desativado aqui para segurança)
ENABLE_ML = os.getenv("ENABLE_ML", "false").lower() == "true" and TF_AVAILABLE

# thresholds de segurança
MEM_THRESHOLD_KB = int(os.getenv("MEM_THRESHOLD_KB", "350000"))  # 350 MB por padrão (ajuste se souber)
ML_TRAIN_EPOCHS = int(os.getenv("ML_TRAIN_EPOCHS", "2"))         # poucas épocas para economizar CPU/RAM
TRAIN_HISTORY = int(os.getenv("TRAIN_HISTORY", "150"))          # janelas para treino (reduzir para economizar)
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.60"))

# rate limit Telegram (mensagens por minuto)
TG_RATE_PER_MIN = int(os.getenv("TG_RATE_PER_MIN", "6"))

# Histórico de sinais e controle de envio
historico_sinais = {}
telegram_msgs_last_min = []
last_cycle_idx = 0

# ============================
# UTILITÁRIOS E INDICADORES
# ============================
def calcular_rsi(series, period=14):
    delta = series.diff()
    ganho = delta.clip(lower=0)
    perda = -1 * delta.clip(upper=0)
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
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["RSI"] = calcular_rsi(df["Close"], 14)
    df["Upper"] = df["Close"].rolling(window=20).mean() + 2 * df["Close"].rolling(window=20).std()
    df["Lower"] = df["Close"].rolling(window=20).mean() - 2 * df["Close"].rolling(window=20).std()
    df["ATR"] = (df["High"] - df["Low"]).rolling(window=14).mean().fillna(method="bfill")
    df["MACD"], df["MACD_SIGNAL"] = calcular_macd(df["Close"])
    df.fillna(method="bfill", inplace=True)
    return df

# features simples para ML
def extrair_features_row(row):
    return np.array([
        row["Close"],
        row["EMA9"],
        row["EMA21"],
        row["RSI"],
        row["MACD"],
        row["MACD_SIGNAL"],
        row["Upper"],
        row["Lower"],
        row["ATR"]
    ], dtype=np.float32)

def construir_dataset_para_ml(df, lookback=TRAIN_HISTORY):
    if len(df) < 12:
        return None, None
    df = df.copy().reset_index(drop=True)
    X = []
    y = []
    # usa janelas pequenas: penúltima vela para prever próxima
    max_i = min(len(df)-1, lookback)
    for i in range(1, max_i):
        row = df.iloc[i-1]
        next_close = df.iloc[i]["Close"]
        current_close = row["Close"]
        feat = extrair_features_row(row)
        label = 1 if next_close > current_close else 0
        X.append(feat)
        y.append(label)
    if not X:
        return None, None
    return np.vstack(X), np.array(y)

# modelo pequeno
def build_small_model(input_dim):
    model = Sequential([
        Dense(48, activation="relu", input_shape=(input_dim,)),
        Dropout(0.15),
        Dense(24, activation="relu"),
        Dropout(0.10),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# função segura para coleta de lixo
def gc_collect():
    try:
        gc.collect()
    except Exception:
        pass

# checa uso de memória; se acima do limiar retorna True
def memory_exceeded():
    if not RESOURCE_AVAILABLE:
        return False
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on linux
        return usage >= MEM_THRESHOLD_KB
    except Exception:
        return False

# ============================
# REGRAS (mesma lógica criteriosa)
# ============================
def obter_sinal_regras(df, timeframe, ativo):
    if len(df) < 4:
        return None
    ultimo = df.iloc[-2]  # candle fechado
    close = float(ultimo["Close"])
    ema9 = float(ultimo["EMA9"])
    ema21 = float(ultimo["EMA21"])
    rsi = float(ultimo["RSI"])
    upper = float(ultimo["Upper"])
    lower = float(ultimo["Lower"])
    atr = float(df["ATR"].tail(5).mean())
    macd = float(ultimo["MACD"])
    macd_signal = float(ultimo["MACD_SIGNAL"])

    # filtro de volatilidade
    if atr < (0.0005 if "USD" in ativo else 0.05):
        return None

    # regras criteriosas
    if ema9 > ema21 and close <= lower * 1.01 and rsi < 65 and macd > macd_signal:
        return "CALL"
    if ema9 < ema21 and close >= upper * 0.99 and rsi > 35 and macd < macd_signal:
        return "PUT"
    return None

# ============================
# ML: treino leve e predição
# ============================
def treinar_modelo_local(df):
    try:
        X, y = construir_dataset_para_ml(df, lookback=TRAIN_HISTORY)
        if X is None:
            return None
        # normalização min-max
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        denom = (X_max - X_min)
        denom[denom == 0] = 1.0
        Xn = (X - X_min) / denom
        model = build_small_model(Xn.shape[1])
        model.fit(Xn, y, epochs=ML_TRAIN_EPOCHS, batch_size=max(8, min(32, len(Xn)//2)), verbose=0)
        # guardar normalização no modelo
        model._x_min = X_min
        model._x_max = X_max
        return model
    except Exception:
        return None

def predict_model_prob(model, feat_vec):
    try:
        X_min = getattr(model, "_x_min", None)
        X_max = getattr(model, "_x_max", None)
        if X_min is None or X_max is None:
            return 0.0
        denom = (X_max - X_min)
        denom[denom == 0] = 1.0
        xn = (feat_vec - X_min) / denom
        xn = xn.reshape(1, -1)
        prob = float(model.predict(xn, verbose=0)[0][0])
        return prob
    except Exception:
        return 0.0

# ============================
# TELEGRAM (rate-limit simples)
# ============================
def enviar_telegram(mensagem):
    global telegram_msgs_last_min
    # limpa lista de timestamps com mais de 60s
    now = time.time()
    telegram_msgs_last_min = [t for t in telegram_msgs_last_min if now - t < 60]
    if len(telegram_msgs_last_min) >= TG_RATE_PER_MIN:
        print("Rate limit Telegram atingido — mensagem descartada temporariamente")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            telegram_msgs_last_min.append(now)
            return True
        else:
            print("Erro Telegram:", r.status_code, r.text)
            return False
    except Exception as e:
        print("Exception Telegram:", e)
        return False

# ============================
# LOOP PRINCIPAL OTIMIZADO
# ============================
def executar_bot():
    global last_cycle_idx

    # ML global: vamos treinar um modelo pequeno combinado se ENABLE_ML e houver recursos
    model = None
    ml_enabled_runtime = False
    if ENABLE_ML and TF_AVAILABLE and not memory_exceeded():
        print("Tentando treinar modelo ML inicial (leve)...")
        # treino rápido por ativo até formar um modelo simples (pode falhar se pouca RAM)
        X_parts = []
        y_parts = []
        for ativo in ATIVOS[:ASSETS_PER_CYCLE]:
            try:
                period = "2d" if TF_SHORT == "5m" else "5d"
                df_tmp = yf.download(ativo, period=period, interval=TF_SHORT, progress=False)
                if df_tmp.empty:
                    continue
                df_tmp = indicadores(df_tmp)
                X, y = construir_dataset_para_ml(df_tmp, lookback=TRAIN_HISTORY)
                if X is not None:
                    X_parts.append(X)
                    y_parts.append(y)
                del df_tmp, X, y
                gc_collect()
                if memory_exceeded():
                    print("Memória alta durante coleta inicial ML — abortando ML")
                    break
            except Exception:
                continue
        if X_parts and not memory_exceeded():
            try:
                X_comb = np.vstack(X_parts)
                y_comb = np.concatenate(y_parts)
                X_min = X_comb.min(axis=0)
                X_max = X_comb.max(axis=0)
                denom = (X_max - X_min)
                denom[denom == 0] = 1.0
                Xn = (X_comb - X_min) / denom
                model = build_small_model(Xn.shape[1])
                model.fit(Xn, y_comb, epochs=ML_TRAIN_EPOCHS, batch_size=max(8, min(32, len(Xn)//2)), verbose=0)
                model._x_min = X_min
                model._x_max = X_max
                ml_enabled_runtime = True
                print("Modelo ML inicial treinado (modo leve).")
            except Exception:
                model = None
                ml_enabled_runtime = False
        else:
            model = None
            ml_enabled_runtime = False
    else:
        print("ML não habilitado ou TensorFlow ausente ou memória insuficiente. Rodando só com regras.")

    # ciclo principal com rotação de ativos (processa ASSETS_PER_CYCLE por iteração)
    n = len(ATIVOS)
    start_idx = 0
    while True:
        try:
            print("="*60)
            print("BOT: ciclo start - rotacionando ativos")
            print(f"ML ativo no runtime: {ml_enabled_runtime}")
            print(f"Processando até {ASSETS_PER_CYCLE} ativos neste ciclo")
            print("="*60)
        except Exception:
            pass

        # processar apenas um lote por ciclo para economizar memória
        for i in range(ASSETS_PER_CYCLE):
            idx = (start_idx + i) % n
            ativo = ATIVOS[idx]
            try:
                # usar períodos compactos para economizar
                df_short = yf.download(ativo, period="1d", interval=TF_SHORT, progress=False)
                df_long = yf.download(ativo, period="5d", interval=TF_LONG, progress=False)
                if df_short.empty or df_long.empty:
                    # cleanup e next
                    for obj in ("df_short", "df_long"):
                        try:
                            del globals()[obj]
                        except Exception:
                            pass
                    gc_collect()
                    continue

                df_short = indicadores(df_short)
                df_long = indicadores(df_long)

                # sinal pelas regras (no timeframe curto), confirmado pelo longo (multi-timeframe)
                sinal_short = obter_sinal_regras(df_short, TF_SHORT, ativo)
                sinal_long = obter_sinal_regras(df_long, TF_LONG, ativo)

                sinal_final = None
                if sinal_short and sinal_long and sinal_short == sinal_long:
                    # se ML ativo, checar probabilidade
                    if ml_enabled_runtime and model is not None:
                        feat = extrair_features_row(df_short.iloc[-2])
                        prob_up = predict_model_prob(model, feat)
                        if sinal_short == "CALL" and prob_up >= ML_CONFIDENCE_THRESHOLD:
                            sinal_final = "CALL"
                        elif sinal_short == "PUT" and (1 - prob_up) >= ML_CONFIDENCE_THRESHOLD:
                            sinal_final = "PUT"
                        else:
                            # ML descartou — não emitir
                            sinal_final = None
                    else:
                        sinal_final = sinal_short

                # checar duplicados e enviar
                chave = f"{ativo}_{TF_SHORT}"
                ultimo = historico_sinais.get(chave)
                if sinal_final and ultimo != sinal_final:
                    enviado = enviar_telegram(f"{ativo} | {TF_SHORT}/{TF_LONG} | {sinal_final}")
                    if enviado:
                        historico_sinais[chave] = sinal_final

                # cleanup de dataframes
                try:
                    del df_short, df_long
                except Exception:
                    pass
                gc_collect()

                # segurança: se memória subir demais, desligar ML e reduzir carga
                if memory_exceeded():
                    print("Memória alta detectada durante execução. Desativando ML e reduzindo carga.")
                    ml_enabled_runtime = False
                    model = None
                    # reduzir ASSETS_PER_CYCLE temporariamente para 3
                    # (não alteramos a variável global, só imprimimos sugestão)
                    # continue loop normalmente

            except Exception as e:
                print(f"Erro ao processar {ativo}: {e}")
                # traceback.print_exc()
                try:
                    del df_short, df_long
                except Exception:
                    pass
                gc_collect()
                continue

        # avançar o índice de rotação
        start_idx = (start_idx + ASSETS_PER_CYCLE) % n

        # retrain leve opcional (só se ML estiver ativo e memória ok)
        if ml_enabled_runtime and model is not None and not memory_exceeded():
            try:
                # reconstrói dataset leve para atualizar o modelo (período curto)
                # atenção: custo de retrain é pequeno (epocas=1)
                X_acc = []
                y_acc = []
                for ativo_r in ATIVOS[:ASSETS_PER_CYCLE]:
                    try:
                        df_r = yf.download(ativo_r, period="2d", interval=TF_SHORT, progress=False)
                        if df_r.empty:
                            continue
                        df_r = indicadores(df_r)
                        Xr, yr = construir_dataset_para_ml(df_r, lookback=TRAIN_HISTORY//2)
                        if Xr is not None:
                            X_acc.append(Xr)
                            y_acc.append(yr)
                        del df_r, Xr, yr
                        gc_collect()
                    except Exception:
                        continue
                if X_acc:
                    Xc = np.vstack(X_acc)
                    yc = np.concatenate(y_acc)
                    X_min = Xc.min(axis=0)
                    X_max = Xc.max(axis=0)
                    denom = (X_max - X_min)
                    denom[denom == 0] = 1.0
                    Xn = (Xc - X_min) / denom
                    # fit rápido
                    model.fit(Xn, yc, epochs=1, batch_size=max(8, min(32, len(Xn)//2)), verbose=0)
                    # atualizar normalização (simples)
                    model._x_min = X_min
                    model._x_max = X_max
            except Exception:
                # se falhar, desliga ML runtime
                ml_enabled_runtime = False
                model = None

        # esperar antes do próximo lote
        time.sleep(60)
        gc_collect()

# ============================
# FLASK (keep-alive)
# ============================
app = Flask("bot")

@app.route("/")
def home():
    return "Bot otimizado rodando."

@app.route("/ping")
def ping():
    return "pong"

def run_server():
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    # roda Flask em thread e o bot no principal
    t = Thread(target=run_server, daemon=True)
    t.start()
    executar_bot()
