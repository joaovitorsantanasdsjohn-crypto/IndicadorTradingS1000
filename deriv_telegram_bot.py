import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import threading
from flask import Flask
import time
from pathlib import Path

load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
APP_ID = os.getenv("DERIV_APP_ID", "1089")
# Lê os pares da variável de ambiente SYMBOLS (separados por vírgula)
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]

if not SYMBOLS:
    SYMBOLS = [
        "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
        "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
        "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
        "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
    ]

# Diretório para armazenar candles CSV
DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Telegram ----------------
def send_telegram(message):
    """Envia mensagem ao Telegram (sem throttling)."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram não configurado (TELEGRAM_TOKEN/CHAT_ID). Mensagem:", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e} -- Mensagem: {message}")

# Throttle simples para evitar flood de mensagens por símbolo/tipo
_last_message_time = {}  # chave -> timestamp
def send_telegram_throttled(key: str, message: str, min_interval_s: int = 300):
    """Envia uma mensagem ao Telegram somente se não foi enviada a mesma chave recentemente."""
    now = time.time()
    last = _last_message_time.get(key, 0)
    if now - last >= min_interval_s:
        _last_message_time[key] = now
        send_telegram(message)
    else:
        # só log no terminal para debug
        print(f"[throttled] {key} ({int(now-last)}s since last). Mensagem suprimida.")

# ---------------- Indicadores (RSI + Bollinger) ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula RSI e Bandas de Bollinger.
    Mantém todas colunas originais da resposta do WS.
    """
    # garante ordenação por timestamp (caso)
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    # RSI
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    # Bollinger
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- Lógica de Sinal (RSI + Bollinger concordando) ----------------
def gerar_sinal_rsi_bb(df: pd.DataFrame):
    """
    Regras criteriosas:
    - COMPRA: fechamento <= banda inferior E RSI <= 30
    - VENDA : fechamento >= banda superior E RSI >= 70
    Ajuste thresholds conforme quiser (30/70 são clássicos).
    """
    ultima = df.iloc[-1]
    close = float(ultima['close'])
    rsi = float(ultima['rsi']) if not pd.isna(ultima['rsi']) else None
    bb_low = float(ultima['bb_lower']) if not pd.isna(ultima['bb_lower']) else None
    bb_up = float(ultima['bb_upper']) if not pd.isna(ultima['bb_upper']) else None

    # exige que indicadores existam
    if rsi is None or bb_low is None or bb_up is None:
        return None

    # COMPRA - preço tocou/está abaixo da banda inferior e RSI indica sobrevenda
    if close <= bb_low and rsi <= 30:
        return "COMPRA"
    # VENDA - preço tocou/está acima da banda superior e RSI indica sobrecompra
    if close >= bb_up and rsi >= 70:
        return "VENDA"

    return None

# ---------------- Salvamento de candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str, max_rows: int = 200):
    """Salva as últimas `max_rows` velas em CSV (sobrescreve)."""
    path = DATA_DIR / f"candles_{symbol}.csv"
    df_to_save = df.tail(max_rows).copy()
    # Se existir coluna epoch, converte para datetime antes de salvar pra facilitar leitura
    if 'epoch' in df_to_save.columns:
        try:
            df_to_save['timestamp_utc'] = pd.to_datetime(df_to_save['epoch'], unit='s', utc=True)
        except Exception:
            pass
    df_to_save.to_csv(path, index=False)

# ---------------- Utilitários de tempo ----------------
def seconds_to_next_candle(interval_minutes: int):
    """Calcula quantos segundos faltam até o próximo fechamento da vela do intervalo."""
    now = datetime.now(timezone.utc)
    total_seconds = int(now.timestamp())
    period = interval_minutes * 60
    seconds_passed = total_seconds % period
    return period - seconds_passed if seconds_passed != 0 else 0

# ---------------- Monitoramento WebSocket (por símbolo) ----------------
async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    """
    Abre conexão WebSocket para o símbolo, solicita histórico no fechamento de cada vela,
    salva candles e envia sinais quando RSI+BB concordarem.
    """
    await asyncio.sleep(start_delay)  # stagger para evitar flood de conexões
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

    # chaves para throttling
    error_key = f"{symbol}_error"
    timeout_key = f"{symbol}_timeout"
    conn_key = f"{symbol}_connected"

    while True:
        try:
            print(f"[{symbol}] Tentando conectar ao WS (url={url})")
            # open_timeout reduz chance de ficar preso na handshake
            async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=15) as ws:
                print(f"[{symbol}] WebSocket aberto. Solicitando primeiro histórico assim que possível.")
                # Para garantir que confirmamos "conexão ativa" apenas quando receber candles,
                # vamos fazer o primeiro request imediatamente (se estiver alinhado com vela) ou esperar o fechamento.
                # Requisitamos o histórico no final da vela para obter vela completa.
                # Loop de requisições: após receber history, esperamos até próximo fechamento e requisitamos novamente.
                first_response = True

                while True:
                    # espera até o fechamento da vela (ou pequena margem) antes de pedir
                    seconds_wait = seconds_to_next_candle(CANDLE_INTERVAL)
                    if seconds_wait > 0:
                        # espera até fechamento + 1 segundo para garantir vela fechada no servidor
                        await asyncio.sleep(seconds_wait + 1)

                    req = {
                        "ticks_history": symbol,
                        "count": 500,   # buscamos 500, mas iremos armazenar apenas últimas 200
                        "granularity": CANDLE_INTERVAL * 60,
                        "style": "candles"
                    }
                    try:
                        await ws.send(json.dumps(req))
                    except Exception as e:
                        # falha ao enviar -> força reconexão
                        send_telegram_throttled(error_key, f"❌ Erro ao enviar request de candles para {symbol}: {e}", min_interval_s=120)
                        print(f"[{symbol}] Erro ao enviar request: {e}")
                        break

                    try:
                        # Recebe resposta (timeout generoso)
                        response = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        send_telegram_throttled(timeout_key, f"⚠️ Timeout sem receber dados do WebSocket para {symbol}", min_interval_s=300)
                        print(f"[{symbol}] Timeout aguardando resposta do WS.")
                        break
                    except Exception as e:
                        send_telegram_throttled(error_key, f"❌ Erro no WebSocket para {symbol}: {e}", min_interval_s=120)
                        print(f"[{symbol}] Erro recebendo resposta: {e}")
                        break

                    # processa a mensagem
                    try:
                        data = json.loads(response)
                    except Exception as e:
                        print(f"[{symbol}] Não foi possível parsear JSON: {e}")
                        continue

                    if "history" in data and "candles" in data["history"]:
                        candles = data["history"]["candles"]
                        if not candles:
                            print(f"[{symbol}] history veio vazio.")
                            continue

                        df = pd.DataFrame(candles)
                        # garante colunas numéricas
                        df['close'] = df['close'].astype(float)
                        if 'open' in df.columns:
                            df['open'] = df['open'].astype(float)
                        # salva último conjunto (até 500) e também grava últimas 200 em CSV
                        try:
                            df_ind = calcular_indicadores(df)
                        except Exception as e:
                            print(f"[{symbol}] Erro ao calcular indicadores: {e}")
                            df_ind = df  # fallback

                        # salva as últimas 200 velas em CSV (sobrescreve)
                        try:
                            save_last_candles(df_ind, symbol, max_rows=200)
                        except Exception as e:
                            print(f"[{symbol}] Erro ao salvar candles CSV: {e}")

                        # envie confirmação concreta DEPOIS que recebemos a primeira history
                        if first_response:
                            send_telegram_throttled(conn_key, f"✅ Conexão ATIVA e candles recebidos para {symbol} — primeiro download concluído.", min_interval_s=2)
                            print(f"[{symbol}] Primeira resposta de candles recebida e confirmada.")
                            first_response = False

                        # gera sinal com RSI + Bollinger
                        sinal = gerar_sinal_rsi_bb(df_ind)
                        if sinal:
                            # envia sinal com pouco (ou nenhum) throttle — sinais são importantes
                            # porém ainda vamos evitar spam repetido absoluto por símbolo: 60s
                            send_telegram_throttled(f"{symbol}_signal", f"💹 Sinal *{sinal}* detectado para *{symbol}* (vela {CANDLE_INTERVAL} min). Fechamento: {df_ind.iloc[-1]['close']}", min_interval_s=60)
                            print(f"[{symbol}] Sinal {sinal} enviado.")
                        else:
                            print(f"[{symbol}] Nenhum sinal (RSI+BB).")

                    else:
                        # resposta que não é 'history' (pode ser erro)
                        if "error" in data:
                            msg = data.get("error", {}).get("message", str(data))
                            send_telegram_throttled(error_key, f"❌ Erro (history) do WS para {symbol}: {msg}", min_interval_s=120)
                            print(f"[{symbol}] Resposta de erro do WS: {msg}")
                        else:
                            print(f"[{symbol}] Mensagem não esperada do WS: {data}")

                    # loop: aguardaremos até próxima vela (a espera está no topo do loop)

        except Exception as e:
            # Throttle reconnection/error messages para evitar flood
            send_telegram_throttled(f"{symbol}_reconnect", f"🔄 Tentando reconectar WebSocket para {symbol} após erro: {e}", min_interval_s=120)
            print(f"[{symbol}] Exceção na conexão: {e} — reconectando em 5s.")
            await asyncio.sleep(5)

# ---------------- Flask Web Service ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    # Usar porta definida pelo Render
    port = int(os.environ.get("PORT", 10000))
    # roda servidor flask (desenvolvimento) apenas para manter o processo ativo no Render web service
    app.run(host="0.0.0.0", port=port)

# ---------------- Função Principal ----------------
async def main():
    # inicia Flask em thread separada
    threading.Thread(target=run_flask, daemon=True).start()
    # informa no Telegram que o bot iniciou (sem flood)
    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    # informa quais pares serão monitorados
    send_telegram("📊 Começando a monitorar: " + ", ".join(SYMBOLS))

    # cria tarefas com rodízio (stagger) de 1s entre cada para reduzir handshakes simultâneos
    tasks = []
    for i, sym in enumerate(SYMBOLS):
        start_delay = i * 1.0  # 1s de diferença entre cada conexão
        tasks.append(asyncio.create_task(monitor_symbol(sym, start_delay=start_delay)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Encerrando bot.")
