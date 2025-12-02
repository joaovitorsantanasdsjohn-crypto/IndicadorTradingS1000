# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (ajustada) — 1 WS por par (Opção A)
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações principais ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos (5m)
APP_ID = os.getenv("DERIV_APP_ID", "111022")

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = [
    "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
    "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
    "frxEURAUD", "frxAUDJPY", "frxGBPAUD", "frxGBPCAD", "frxAUDNZD",
    "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
]

DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Parâmetros de sinal/freq ----------------
SIGNAL_MIN_INTERVAL_SECONDS = 1800  # cooldown por par (ajustável)
RSI_STRICT_BUY = 40
RSI_STRICT_SELL = 60
RSI_RELAX = 45
USE_MACD_CONFIRMATION = True
MACD_DIFF_RELAX = 0.0001
BB_PROXIMITY_PCT = 0.02  # 2%

# ---------------- Estado / controle ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
sent_download_message = {s: False for s in SYMBOLS}
last_notify_time = {}  # throttle telegram por par

# ---------------- Logging ----------------
logger = logging.getLogger("indicador")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

def log(msg: str, level: str = "info"):
    if level == "info":
        logger.info(msg)
        print(msg, flush=True)
    elif level == "warning":
        logger.warning(msg)
        print(msg, flush=True)
    elif level == "error":
        logger.error(msg)
        print(msg, flush=True)
    else:
        logger.debug(msg)
        print(msg, flush=True)

# ---------------- Telegram helper ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()
    if symbol:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            log(f"Telegram rate limit skipped for {symbol}", "warning")
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log(f"⚠️ Telegram não configurado. Mensagem: {message}", "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
        else:
            log(f"Telegram enviado: {message}")
    except Exception as e:
        log(f"❌ Erro ao enviar Telegram: {e}\n{traceback.format_exc()}", "error")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    try:
        macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd"] = pd.NA
        df["macd_signal"] = pd.NA
        df["macd_diff"] = pd.NA

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df

# ---------------- Lógica de Sinal ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    try:
        if len(df) < 2:
            return None

        now_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        epoch = int(now_row["epoch"])
        close = float(now_row["close"])

        ema20_now = now_row["ema20"]
        ema50_now = now_row["ema50"]
        ema20_prev = prev_row["ema20"]
        ema50_prev = prev_row["ema50"]
        rsi_now = now_row["rsi"]
        bb_lower = now_row["bb_lower"]
        bb_upper = now_row["bb_upper"]
        bb_mavg = now_row["bb_mavg"]
        macd_diff = now_row["macd_diff"]

        if any(pd.isna([ema20_prev, ema50_prev, ema20_now, ema50_now, rsi_now, bb_lower, bb_upper])):
            log(f"[{symbol}] Indicadores incompletos (NaN) — aguardando mais candles.")
            return None

        try:
            log(
                f"[{symbol}] close={close:.5f} EMA20_prev={ema20_prev:.5f} EMA50_prev={ema50_prev:.5f} "
                f"EMA20_now={ema
