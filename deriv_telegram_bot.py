# Indicador Trading S1000

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging
from flask import Flask
import threading
from typing import Optional

# ---------------- ML availability ----------------

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------- Inicialização ----------------

load_dotenv()

# ---------------- Configurações ----------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")  # (não é obrigatório p/ candles, mas pode estabilizar)
APP_ID = os.getenv("DERIV_APP_ID", "111022")

CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
GRANULARITY_SECONDS = CANDLE_INTERVAL * 60

SIGNAL_ADVANCE_MINUTES = 5

WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# watchdog para reconectar se WS ficar "mudo"
WS_CANDLE_TIMEOUT_SECONDS = int(os.getenv("WS_CANDLE_TIMEOUT_SECONDS", "300"))

SYMBOLS = [
    "frxEURUSD","frxUSDJPY","frxGBPUSD","frxUSDCHF","frxAUDUSD",
    "frxUSDCAD","frxNZDUSD","frxEURJPY","frxGBPJPY","frxEURGBP",
    "frxEURAUD","frxAUDJPY","frxGBPAUD","frxGBPCAD","frxAUDNZD","frxEURCAD"
]

# ---------------- Parâmetros Técnicos ----------------

EMA_FAST = 9
EMA_MID  = 21
EMA_SLOW = 34

BB_PERIOD = 20
BB_STD    = 2.3

MFI_PERIOD = 14
RSI_PERIOD = 14

# ---------------- ML ----------------

ML_ENABLED = SKLEARN_AVAILABLE
ML_MIN_TRAINED_SAMPLES = 300
ML_MAX_SAMPLES = 2000
ML_CONF_THRESHOLD = float(os.getenv("ML_CONF_THRESHOLD", "0.55"))
ML_N_ESTIMATORS = 60
ML_MAX_DEPTH = 5

# ---------------- Otimização Memória ----------------
# >>> ESSAS CONFIGS EVITAM ESTOURO NO RENDER FREE (512MB)

MAX_CANDLES_RAM = int(os.getenv("MAX_CANDLES_RAM", "1300"))   # limite por símbolo
INDICATOR_WINDOW = int(os.getenv("INDICATOR_WINDOW", "220"))  # recalcula só o final
TRAIN_EVERY_N_CANDLES = int(os.getenv("TRAIN_EVERY_N_CANDLES", "3"))  # treina ML a cada N velas

# ---------------- Estado ----------------

candles = {s: pd.DataFrame() for s in SYMBOLS}
ml_models = {}
ml_model_ready = {}
last_signal_epoch = {s: None for s in SYMBOLS}
last_processed_epoch = {s: None for s in SYMBOLS}

# contador de velas por símbolo (pra decidir quando treinar ML)
candle_counter = {s: 0 for s in SYMBOLS}

# ---------------- Logging ----------------

logger = logging.getLogger("IndicadorTradingS1000")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    brt = (datetime.utcnow() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S BRT")
    utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    full = f"{utc} | {brt} | {msg}"
    if level == "info":
        logger.info(full)
    elif level == "warning":
        logger.warning(full)
    elif level == "error":
        logger.error(full)

# ---------------- Telegram ----------------

def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        log(f"Erro Telegram: {e}", "error")

# ---------------- Indicadores ----------------

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) < EMA_SLOW + 10:
        return df

    # garante tipos numéricos
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_mid"]  = EMAIndicator(df["close"], EMA_MID).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], RSI_PERIOD).rsi()

    bb = BollingerBands(df["close"], BB_PERIOD, BB_STD)
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # MFI (volume neutro em Forex)
    if "volume" not in df.columns:
        df["volume"] = 1

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1)

    df["mfi"] = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=MFI_PERIOD
    ).money_flow_index()

    return df

def update_indicators_light(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula indicadores só na cauda do DF, pra economizar RAM/CPU.
    """
    if df.empty:
        return df
    tail = df.tail(INDICATOR_WINDOW).copy().reset_index(drop=True)
    tail = calcular_indicadores(tail)

    # escreve de volta somente nas linhas da cauda
    start = max(len(df) - len(tail), 0)
    for col in ["ema_fast","ema_mid","ema_slow","rsi","bb_mid","bb_upper","bb_lower","mfi","volume"]:
        if col in tail.columns:
            df.loc[df.index[start:], col] = tail[col].values

    return df

def trim_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limita o dataframe por símbolo pra não estourar RAM.
    """
    if len(df) > MAX_CANDLES_RAM:
        df = df.iloc[-MAX_CANDLES_RAM:].reset_index(drop=True)
    return df

# ---------------- ML ----------------

def build_ml_dataset(df: pd.DataFrame):
    df = df.dropna().copy()

    # features que ML deve usar
    df["future"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # remove epoch e future
    X = df.drop(columns=["future", "epoch"], errors="ignore").iloc[:-1]
    y = df["future"].iloc[:-1]

    return X.tail(ML_MAX_SAMPLES), y.tail(ML_MAX_SAMPLES)

def train_ml(symbol: str):
    if not ML_ENABLED:
        ml_model_ready[symbol] = False
        return

    df = candles[symbol]
    if len(df) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return

    X, y = build_ml_dataset(df)
    if len(X) < ML_MIN_TRAINED_SAMPLES:
        ml_model_ready[symbol] = False
        return

    model = RandomForestClassifier(
        n_estimators=ML_N_ESTIMATORS,
        max_depth=ML_MAX_DEPTH,
        random_state=42,
        n_jobs=1  # IMPORTANT: não explode CPU/RAM no Render
    )
    model.fit(X, y)

    ml_models[symbol] = (model, X.columns.tolist())
    ml_model_ready[symbol] = True

def ml_predict(symbol: str, row: pd.Series) -> Optional[float]:
    if not ML_ENABLED:
        return None
    if not ml_model_ready.get(symbol):
        return None

    model, cols = ml_models[symbol]
    try:
        vals = [float(row[c]) for c in cols]
    except Exception:
        return None

    try:
        return model.predict_proba([vals])[0][1]
    except Exception:
        return None

# ---------------- SETUP / CONTEXTO DE MERCADO ----------------

def has_market_context(row: pd.Series, direction: str) -> bool:
    """
    Só manda sinal quando existir SETUP com contexto,
    usando EMA + RSI + Bollinger + MFI.
    """

    try:
        close = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_mid = float(row["ema_mid"])
        ema_slow = float(row["ema_slow"])
        rsi = float(row["rsi"])
        mfi = float(row["mfi"])
        bb_mid = float(row["bb_mid"])
    except Exception:
        return False

    # COMPRA
    if direction == "COMPRA":
        if not (ema_fast > ema_mid > ema_slow):
            return False

        if not (35 <= rsi <= 55):
            return False

        if close > bb_mid:
            return False

        if mfi >= 75:
            return False

        return True

    # VENDA
    if direction == "VENDA":
        if not (ema_fast < ema_mid < ema_slow):
            return False

        if not (45 <= rsi <= 65):
            return False

        if close < bb_mid:
            return False

        if mfi <= 25:
            return False

        return True

    return False

# ---------------- SINAL ----------------

def avaliar_sinal(symbol: str):
    df = candles[symbol]
    if len(df) < EMA_SLOW + 50:
        return

    row = df.iloc[-1]

    direction = "COMPRA" if float(row.get("ema_fast", 0)) >= float(row.get("ema_mid", 0)) else "VENDA"

    # 1) contexto
    if not has_market_context(row, direction):
        return

    # 2) ML filtra probabilidade
    ml_prob = ml_predict(symbol, row)
    if ml_prob is None or ml_prob < ML_CONF_THRESHOLD:
        return

    epoch = int(row["epoch"])

    # evita duplicar sinal na mesma vela
    if last_signal_epoch[symbol] == epoch:
        return
    last_signal_epoch[symbol] = epoch

    entry_time = datetime.utcfromtimestamp(epoch) - timedelta(hours=3)
    entry_time += timedelta(minutes=SIGNAL_ADVANCE_MINUTES)

    ativo = symbol.replace("frx", "")

    msg = (
        f"📊 <b>ATIVO:</b> {ativo}\n"
        f"📈 <b>DIREÇÃO:</b> {direction}\n"
        f"⏰ <b>ENTRADA:</b> {entry_time.strftime('%H:%M')}\n"
        f"🤖 <b>ML:</b> {ml_prob*100:.0f}%"
    )

    send_telegram(msg)
    log(f"{symbol} — sinal enviado {direction} (ML {ml_prob*100:.0f}%)")

# ---------------- WebSocket ----------------

async def ws_loop(symbol: str):
    while True:
        try:
            log(f"{symbol} WS conectando...", "info")

            async with websockets.connect(
                WS_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT
            ) as ws:

                log(f"{symbol} WS conectado ✅", "info")

                # (Opcional) autoriza para estabilizar algumas contas/requests
                if DERIV_TOKEN:
                    try:
                        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                        auth_raw = await ws.recv()
                        auth_data = json.loads(auth_raw)
                        if "error" in auth_data:
                            log(f"{symbol} authorize falhou: {auth_data.get('error')}", "warning")
                    except Exception as e:
                        log(f"{symbol} authorize erro: {e}", "warning")

                # 1) histórico
                req_hist = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1200,
                    "end": "latest",
                    "granularity": GRANULARITY_SECONDS,
                    "style": "candles"
                }
                await ws.send(json.dumps(req_hist))
                log(f"{symbol} Histórico solicitado 📥", "info")

                hist_raw = await ws.recv()
                hist_data = json.loads(hist_raw)

                if "error" in hist_data:
                    log(f"{symbol} WS retornou erro: {hist_data.get('error')}", "error")
                    await ws.close()
                    continue

                df = pd.DataFrame(hist_data.get("candles", []))
                if df.empty:
                    log(f"{symbol} Histórico vazio — reconectando 🔁", "warning")
                    await ws.close()
                    continue

                # organiza colunas principais
                for col in ["open", "high", "low", "close", "epoch"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                df = calcular_indicadores(df)
                df = trim_df(df)
                candles[symbol] = df

                # treina ML uma vez no start (se der)
                train_ml(symbol)

                # 2) stream candles (chave correta é "candles" com subscribe)
                req_stream = {
                    "candles": symbol,
                    "subscribe": 1,
                    "granularity": GRANULARITY_SECONDS
                }
                await ws.send(json.dumps(req_stream))
                log(f"{symbol} Stream (candles) ligado 🔴", "info")

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_CANDLE_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        log(f"{symbol} Sem candles por {WS_CANDLE_TIMEOUT_SECONDS}s — reconectando 🔁", "warning")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    data = json.loads(raw)

                    if "error" in data:
                        log(f"{symbol} WS retornou erro: {data.get('error')}", "error")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    # candle update
                    if "candles" in data and data["candles"]:
                        new_row = data["candles"][0]
                        df = candles[symbol]

                        # normaliza numérico
                        for col in ["open", "high", "low", "close", "epoch"]:
                            if col in new_row:
                                try:
                                    new_row[col] = float(new_row[col]) if col != "epoch" else int(new_row[col])
                                except Exception:
                                    pass

                        if df.empty:
                            df = pd.DataFrame([new_row])
                        else:
                            last_epoch = int(df.iloc[-1]["epoch"])
                            new_epoch = int(new_row["epoch"])

                            if new_epoch == last_epoch:
                                # atualiza somente chaves base
                                for k, v in new_row.items():
                                    df.at[df.index[-1], k] = v
                            else:
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                        # limita DF pra não estourar RAM
                        df = trim_df(df)

                        # atualiza indicadores somente na cauda
                        df = update_indicators_light(df)
                        candles[symbol] = df

                        # processar só quando epoch muda
                        try:
                            current_epoch = int(df.iloc[-1]["epoch"])
                        except Exception:
                            continue

                        if last_processed_epoch[symbol] != current_epoch:
                            last_processed_epoch[symbol] = current_epoch

                            candle_counter[symbol] += 1

                            # treina ML só a cada N velas (economiza RAM/CPU e evita reset)
                            if candle_counter[symbol] % TRAIN_EVERY_N_CANDLES == 0:
                                train_ml(symbol)

                            avaliar_sinal(symbol)

        except Exception as e:
            log(f"{symbol} WS erro: {e}", "error")
            await asyncio.sleep(5)

# ---------------- Flask ----------------

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def health():
    return "OK", 200

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

# ---------------- MAIN ----------------

async def main():
    send_telegram("🚀 BOT INICIADO — M5 ATIVO")
    await asyncio.gather(*(ws_loop(s) for s in SYMBOLS))

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
