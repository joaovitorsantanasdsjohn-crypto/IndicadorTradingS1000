import os
import threading
import time
import ssl
import yfinance as yf
import pandas as pd
import requests
from flask import Flask
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

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

# ===================== CONFIGURAÇÃO SSL =====================
ssl._create_default_https_context = ssl._create_unverified_context

# ===================== LISTA DE ATIVOS =====================
ativos = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "AUDCAD=X",
    "GBPCHF=X", "EURCHF=X", "USDMXN=X"
]

# ===================== MODELO LSTM =====================
def criar_modelo():
    model = Sequential()
   
