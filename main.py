import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Configurações do Telegram
TELEGRAM_TOKEN = "7964245740:AAH7yN95r_NNQaq3OAJU43S4nagIAcgK2w0"
TELEGRAM_CHAT_ID = "6370166264"

# Ativos e timeframes
ativos = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
timeframes = ["5m", "15m"]

# Histórico de sinais
historico_sinais = {}

# Inicializa o modelo ML (rede neural simples)
scaler = MinMaxScaler()
model = Sequential([
    Dense(64, activation='relu', input_dim=5),  # 5 indicadores principais
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Função de envio Telegram
def enviar_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": mensagem})
    except:
        pass

# Funções de indicadores
def calcular_indicadores(df):
    # EMA
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    df['BB_M'] = df['Close'].rolling(20).mean()
    df['BB_U'] = df['BB_M'] + 2*df['Close'].rolling(20).std()
    df['BB_L'] = df['BB_M'] - 2*df['Close'].rolling(20).std()
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    return df

# Função que decide o sinal
def obter_sinal(df, ativo, timeframe):
    ultimo = df.iloc[-1]
    # Critérios máximos de assertividade
    if (ultimo['Close'] > ultimo['EMA_9'] > ultimo['EMA_21'] and
        ultimo['RSI'] > 55 and
        ultimo['Close'] > ultimo['BB_U'] and
        ultimo['MACD'] > ultimo['Signal']):
        return "CALL"
    elif (ultimo['Close'] < ultimo['EMA_9'] < ultimo['EMA_21'] and
          ultimo['RSI'] < 45 and
          ultimo['Close'] < ultimo['BB_L'] and
          ultimo['MACD'] < ultimo['Signal']):
        return "PUT"
    else:
        return None

# Treinamento ML básico com cada sinal novo
def treinar_modelo(indicadores, sinal):
    X = scaler.fit_transform(indicadores)
    y = np.array([1 if sinal=="CALL" else 0])
    model.fit(X, y, epochs=1, verbose=0)

# Loop principal
def executar_bot():
    while True:
        for ativo in ativos:
            for timeframe in timeframes:
                try:
                    df = yf.download(ativo, period="2d", interval=timeframe, progress=False)
                    if df.empty:
                        continue
                    df = calcular_indicadores(df)
                    sinal = obter_sinal(df, ativo, timeframe)
                    if sinal:
                        enviar_telegram(f"{ativo} | {timeframe.upper()} | {sinal}")
                        # Treina ML com indicadores do último candle
                        ultimos_ind = df[['EMA_9','EMA_21','RSI','MACD','ATR']].iloc[-1].values.reshape(1,-1)
                        treinar_modelo(ultimos_ind, sinal)
                except Exception as e:
                    print(f"Erro em {ativo}: {e}")
        time.sleep(60)

if __name__ == "__main__":
    executar_bot()
