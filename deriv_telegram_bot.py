# deriv_telegram_bot.py
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
from pathlib import Path
import time

load_dotenv()

# ---------------- Configurações ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
APP_ID = os.getenv("DERIV_APP_ID", "1089")

# Lista de pares (puxa de env SYMBOLS ou usa lista padrão)
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]
if not SYMBOLS:
    SYMBOLS = [
        "frxEURUSD", "frxUSDJPY", "frxGBPUSD", "frxUSDCHF", "frxAUDUSD",
        "frxUSDCAD", "frxNZDUSD", "frxEURJPY", "frxGBPJPY", "frxEURGBP",
        "frxEURAUD", "frxAUDJPY", "frxCHFJPY", "frxCADJPY", "frxGBPAUD",
        "frxGBPCAD", "frxAUDNZD", "frxEURCAD", "frxUSDNOK", "frxUSDSEK"
    ]

# Diretório para armazenar candles
DATA_DIR = Path("./candles_data")
DATA_DIR.mkdir(exist_ok=True)

# Limite de conexões concorrentes ao WebSocket (correção 1)
MAX_CONCURRENT_WS = 3
ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)

# ---------------- Telegram ----------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram não configurado. Mensagem:", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print("📨 Telegram:", message[:120])
        else:
            print(f"❌ Telegram HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    # RSI + BollingerBands (conforme pedido)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

# ---------------- Gerar Sinal ----------------
def gerar_sinal(df: pd.DataFrame):
    ultima = df.iloc[-1]
    close = float(ultima['close'])
    rsi = float(ultima['rsi']) if not pd.isna(ultima['rsi']) else None
    bb_low = float(ultima['bb_lower']) if not pd.isna(ultima['bb_lower']) else None
    bb_up = float(ultima['bb_upper']) if not pd.isna(ultima['bb_upper']) else None

    if rsi is None or bb_low is None or bb_up is None:
        return None

    # Compra: preço ≤ banda inferior E RSI ≤ 30
    if close <= bb_low and rsi <= 30:
        return "COMPRA"
    # Venda: preço ≥ banda superior E RSI ≥ 70
    if close >= bb_up and rsi >= 70:
        return "VENDA"

    return None

# ---------------- Salvamento de candles ----------------
def save_last_candles(df: pd.DataFrame, symbol: str, max_rows: int = 200):
    path = DATA_DIR / f"candles_{symbol}.csv"
    df_to_save = df.tail(max_rows).copy()
    if 'epoch' in df_to_save.columns:
        try:
            df_to_save['timestamp_utc'] = pd.to_datetime(df_to_save['epoch'], unit='s', utc=True)
        except Exception:
            pass
    df_to_save.to_csv(path, index=False)
    print(f"[{symbol}] Salvas últimas {len(df_to_save)} velas em {path}")

# ---------------- Utilitários ----------------
def seconds_to_next_candle(interval_minutes: int):
    now = datetime.now(timezone.utc)
    total_seconds = int(now.timestamp())
    period = interval_minutes * 60
    seconds_passed = total_seconds % period
    return (period - seconds_passed) if seconds_passed != 0 else 0

# ---------------- Monitoramento WebSocket (com semaphore para 3 conexões) ----------------
async def monitor_symbol(symbol: str, start_delay: float = 0.0):
    await asyncio.sleep(start_delay)

    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    backoff_seconds = 5

    while True:
        await ws_semaphore.acquire()
        try:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=15
                ) as ws:
                    send_telegram(f"✅ Conexão ativa com WebSocket da Deriv para {symbol}")
                    print(f"[{symbol}] WebSocket conectado. (slot reservado)")

                    first_received = False

                    while True:
                        wait = seconds_to_next_candle(CANDLE_INTERVAL)
                        print(f"[{symbol}] Aguardando {wait + 1}s até fechamento da próxima vela.")
                        if wait > 0:
                            await asyncio.sleep(wait + 1)

                        # tentativas para history (retries)
                        history_attempts = 0
                        history_success = False
                        history_data = None
                        while history_attempts < 3 and not history_success:
                            history_attempts += 1
                            req = {
                                "ticks_history": symbol,
                                "count": 500,
                                "end": "latest",
                                "granularity": CANDLE_INTERVAL * 60,
                                "style": "candles"
                            }
                            try:
                                await ws.send(json.dumps(req))
                            except Exception as e:
                                print(f"[{symbol}] Erro ao enviar req history (tentativa {history_attempts}): {e}")
                                await asyncio.sleep(1)
                                continue

                            try:
                                response = await asyncio.wait_for(ws.recv(), timeout=30)
                            except asyncio.TimeoutError:
                                print(f"[{symbol}] Timeout aguardando resposta (history) (tentativa {history_attempts}).")
                                await asyncio.sleep(1)
                                continue
                            except Exception as e:
                                print(f"[{symbol}] Erro recv (history) (tentativa {history_attempts}): {e}")
                                await asyncio.sleep(1)
                                continue

                            try:
                                data = json.loads(response)
                            except Exception as e:
                                print(f"[{symbol}] Resposta inválida JSON (tentativa {history_attempts}): {e}")
                                await asyncio.sleep(1)
                                continue

                            if "history" in data and "candles" in data["history"]:
                                history_success = True
                                history_data = data
                            else:
                                # mensagem de erro no history (ex.: Input validation failed)
                                err_msg = data.get("error", {}).get("message", "sem detalhes")
                                print(f"[{symbol}] history retornou erro (tentativa {history_attempts}): {err_msg}")
                                await asyncio.sleep(1)

                        if not history_success:
                            send_telegram(f"⚠️ Falha ao obter candles para {symbol} após 3 tentativas")
                            print(f"[{symbol}] Falha history após 3 tentativas.")
                            break  # sai do loop interno para reconectar (vai para exceção externa)

                        # processar candles
                        candles = history_data["history"]["candles"]
                        df = pd.DataFrame(candles)
                        if 'close' in df.columns:
                            df['close'] = df['close'].astype(float)
                        if 'open' in df.columns:
                            df['open'] = df['open'].astype(float)

                        # logs de open/close para inspeção no terminal
                        try:
                            open_val = df.iloc[-1].get('open') if 'open' in df.columns else None
                            close_val = float(df.iloc[-1]['close'])
                            print(f"[{symbol}] Última vela -> open: {open_val}  close: {close_val}")
                        except Exception:
                            pass

                        df_ind = calcular_indicadores(df)
                        save_last_candles(df_ind, symbol, max_rows=200)

                        if not first_received:
                            send_telegram(f"📡 [{symbol}] Candles recebidos. Último fechamento: {df_ind.iloc[-1]['close']:.5f}")
                            first_received = True

                        sinal = gerar_sinal(df_ind)
                        if sinal:
                            send_telegram(f"💹 Sinal {sinal} detectado para {symbol} ({CANDLE_INTERVAL} min)")

                        # próxima vela (loop volta e aguarda próximo fechamento)
            except Exception as e:
                send_telegram(f"⚠️ Erro ou desconexão no WebSocket para {symbol}: {e}")
                print(f"[{symbol}] Exceção no WS: {e}")
        finally:
            ws_semaphore.release()
            print(f"[{symbol}] Slot liberado. Reconectando em {backoff_seconds}s …")
            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 120)

# ---------------- Flask Web Service ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo ✅"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    # Note: Render executa o Flask development server aqui; é suficiente para o worker web.
    app.run(host="0.0.0.0", port=port)

# ---------------- Principal ----------------
async def main():
    # Inicia web endpoint para manter serviço ativo no Render
    threading.Thread(target=run_flask, daemon=True).start()

    send_telegram("✅ Bot iniciado com sucesso no Render e pronto para análise!")
    print("Iniciando monitoramento dos símbolos:", SYMBOLS)

    # Mensagem de teste de conexão com o Telegram
    send_telegram("🔍 Teste de conexão Telegram: se você recebeu esta mensagem, o bot está OK ✅")

    tasks = []
    for i, sym in enumerate(SYMBOLS):
        delay = i * 5  # escalona abertura (5s entre pares) para reduzir picos
        tasks.append(asyncio.create_task(monitor_symbol(sym, start_delay=delay)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Encerrando por interrupt.")
