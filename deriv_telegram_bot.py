# ===============================================================
# deriv_telegram_bot.py — LÓGICA B (AJUSTADA) — Opção A — COMPLETO
# (com: anti-duplicação, horário Brasília, backoff/reconexão,
#  validação robusta do histórico, e Força do Sinal)
# ===============================================================

import asyncio
import websockets
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import threading
from flask import Flask
from pathlib import Path
import time
import random
import logging
import traceback
import math

# ---------------- Inicialização ----------------
load_dotenv()

# ---------------- Configurações principais ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutos
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

# ---------------- Parâmetros (mais assertivo) ----------------
BB_PROXIMITY_PCT = 0.20
RSI_BUY_MAX = 52
RSI_SELL_MIN = 48
MACD_TOLERANCE = 0.002

# Anti-spam minimum between signals for the same symbol (seconds).
# We keep 1-sinal-por-candle as primary, but also block very rapid repeats after network reconnects.
MIN_SECONDS_BETWEEN_SIGNALS = 10

# ---------------- Estado ----------------
last_signal_state = {s: None for s in SYMBOLS}
last_signal_candle = {s: None for s in SYMBOLS}
last_signal_time = {s: 0 for s in SYMBOLS}
last_notify_time = {}  # throttle per-symbol tg messages

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
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)
    print(msg, flush=True)

# ---------------- Telegram helper ----------------
def send_telegram(message: str, symbol: str = None):
    now = time.time()

    # per-symbol small throttle to avoid duplicate trigger spam
    if symbol:
        last = last_notify_time.get(symbol, 0)
        if now - last < 3:
            log(f"[TG] throttle skip for {symbol}", "warning")
            return
        last_notify_time[symbol] = now

    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("⚠️ Telegram não configurado.", "warning")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        # Telegram Markdown: wrap in Markdown; user requested Portuguese; keep parse_mode.
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            log(f"❌ Telegram retornou {r.status_code}: {r.text}", "error")
    except Exception as e:
        log(f"[TG] Erro ao enviar: {e}\n{traceback.format_exc()}", "error")

# ---------------- Indicadores ----------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    # EMAs
    df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()

    # RSI
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    # MACD diff (mais leve)
    try:
        macd = MACD(df["close"], 26, 12, 9)
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd_diff"] = pd.NA

    # Bollinger
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()

    return df

# ---------------- Lógica — CORRIGIDA + FORÇA DO SINAL ----------------
def gerar_sinal(df: pd.DataFrame, symbol: str):
    """
    Retorna None ou dict {"tipo": "COMPRA"/"VENDA", "forca": int(0-100)}.
    Regras:
      - 1 sinal por candle (last_signal_candle)
      - validação de NaNs
      - força combinada a partir de distância a Bollinger, RSI e MACD
    """
    try:
        if len(df) < 3:
            return None

        now = df.iloc[-1]
        prev = df.iloc[-2]

        epoch = int(now["epoch"])
        close = float(now["close"])

        # Evitar duplicidade: 1 sinal por candle fechado
        if last_signal_candle.get(symbol) == epoch:
            # já enviamos sinal para este candle
            return None

        # indicadores
        ema20_now, ema50_now = now["ema20"], now["ema50"]
        ema20_prev, ema50_prev = prev["ema20"], prev["ema50"]
        rsi_now = now["rsi"]
        bb_upper, bb_lower = now["bb_upper"], now["bb_lower"]
        macd_diff = now.get("macd_diff")

        if any(pd.isna([ema20_now, ema50_now, ema20_prev, ema50_prev, rsi_now, bb_upper, bb_lower])):
            log(f"[{symbol}] Indicadores incompletos (NaN) — aguardando mais candles.", "warning")
            return None

        # Tendência / cruzamentos
        cruzou_up = (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
        cruzou_down = (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)
        tendencia_up = ema20_now > ema50_now
        tendencia_down = ema20_now < ema50_now

        # Bollinger proximidade
        range_bb = bb_upper - bb_lower
        if range_bb == 0 or math.isclose(range_bb, 0.0):
            return None
        lim_inf = bb_lower + range_bb * BB_PROXIMITY_PCT
        lim_sup = bb_upper - range_bb * BB_PROXIMITY_PCT

        perto_lower = close <= lim_inf
        perto_upper = close >= lim_sup

        # RSI
        buy_rsi_ok = rsi_now <= RSI_BUY_MAX
        sell_rsi_ok = rsi_now >= RSI_SELL_MIN

        # MACD
        macd_buy_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff > -MACD_TOLERANCE)
        macd_sell_ok = True if (macd_diff is None or pd.isna(macd_diff)) else (macd_diff < MACD_TOLERANCE)

        # Condições finais
        cond_buy = (cruzou_up or tendencia_up) and perto_lower and buy_rsi_ok and macd_buy_ok
        cond_sell = (cruzou_down or tendencia_down) and perto_upper and sell_rsi_ok and macd_sell_ok

        # Se nenhum, retorna
        if not (cond_buy or cond_sell):
            return None

        # Calcular força (0-100)
        def calc_forca(is_buy: bool):
            score = 0.0

            # 1) Bollinger proximity contribution (0..40)
            if is_buy:
                # closer to lower band (below lim_inf) -> stronger
                # dist normalized: (lim_inf - close) / range_bb, clamped [0,1]
                dist = max(0.0, min(1.0, (lim_inf - close) / range_bb))
                # if price is much below lim_inf, dist closer to 1
                score += dist * 40.0
            else:
                dist = max(0.0, min(1.0, (close - lim_sup) / range_bb))
                score += dist * 40.0

            # 2) RSI contribution (0..30) — farther into desired zone is stronger
            if is_buy:
                # RSI lower than ideal (RSI_BUY_MAX) -> stronger; normalize over 20 pts
                rsi_strength = max(0.0, min(1.0, (RSI_BUY_MAX - rsi_now) / 20.0))
                score += rsi_strength * 30.0
            else:
                rsi_strength = max(0.0, min(1.0, (rsi_now - RSI_SELL_MIN) / 20.0))
                score += rsi_strength * 30.0

            # 3) EMA separation contribution (0..20) — larger separation implies clearer trend
            ema_sep = abs(ema20_now - ema50_now)
            # normalize: use a small scale — avoid explosion; assume typical pip sizes.
            # We map ema_sep to [0,1] using a heuristic divisor (scale)
            scale = max(1e-6, 0.01)  # 0.01 is a safe heuristic; depends on instrument
            sep_strength = max(0.0, min(1.0, ema_sep / scale))
            score += sep_strength * 20.0

            # 4) MACD (0..10) — if present, reward sign concordance
            if macd_diff is not None and not pd.isna(macd_diff):
                macd_strength = max(0.0, min(1.0, abs(macd_diff) / (MACD_TOLERANCE * 5)))  # soften scale
                score += macd_strength * 10.0

            # clamp 0..100 and return int
            return int(max(0, min(100, round(score))))

        if cond_buy:
            force = calc_forca(is_buy=True)
            # marca para evitar duplicatas e como timestamp
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = time.time()
            last_signal_state[symbol] = "COMPRA"
            return {"tipo": "COMPRA", "forca": force}

        if cond_sell:
            force = calc_forca(is_buy=False)
            last_signal_candle[symbol] = epoch
            last_signal_time[symbol] = time.time()
            last_signal_state[symbol] = "VENDA"
            return {"tipo": "VENDA", "forca": force}

        return None

    except Exception as e:
        log(f"[{symbol}] Erro em gerar_sinal: {e}\n{traceback.format_exc()}", "error")
        return None

# ---------------- Persistência ----------------
def save_last_candles(df: pd.DataFrame, symbol: str):
    path = DATA_DIR / f"candles_{symbol}.csv"
    try:
        df.tail(200).to_csv(path, index=False)
    except Exception as e:
        log(f"[{symbol}] Erro ao salvar candles: {e}", "warning")

# ---------------- Monitor WebSocket (com backoff e validação) ----------------
async def monitor_symbol(symbol: str):
    reconnect_attempt = 0
    while True:
        try:
            reconnect_attempt += 1
            backoff_base = min(60, 2 ** min(6, reconnect_attempt))  # 2,4,8,16,32,64 -> capped 60
            jitter = random.uniform(0.5, 1.5)

            log(f"[{symbol}] Conectando ao WS (attempt {reconnect_attempt})...")
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                log(f"[{symbol}] WS conectado.")
                reconnect_attempt = 0  # reset on success

                # authorize
                try:
                    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                    auth_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except Exception as e:
                    log(f"[{symbol}] Falha ao autorizar/receber authorize: {e}", "error")
                    raise

                # initial history request: tente obter candles robustamente
                df = None
                history_tries = 0
                while True:
                    history_tries += 1
                    try:
                        req_hist = {
                            "ticks_history": symbol,
                            "count": 200,
                            "end": "latest",
                            "granularity": CANDLE_INTERVAL * 60,
                            "style": "candles"
                        }
                        await ws.send(json.dumps(req_hist))
                        raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(raw)

                        # validar resposta
                        if isinstance(data, dict) and "history" in data and isinstance(data["history"], dict):
                            # some endpoints use history.candles
                            history = data["history"]
                            if isinstance(history.get("candles"), list) and len(history["candles"]) > 0:
                                df = pd.DataFrame(history["candles"])
                                break

                        if isinstance(data, dict) and "candles" in data and isinstance(data["candles"], list) and len(data["candles"]) > 0:
                            df = pd.DataFrame(data["candles"])
                            break

                        # se veio msg sem candles (por exemplo tick/control) aguarde e tente novamente
                        log(f"[{symbol}] Histórico inicial sem candles (tentativa {history_tries}), resposta: {list(data.keys()) if isinstance(data, dict) else type(data)}", "warning")
                        await asyncio.sleep(1.0 + random.random() * 1.5)

                    except asyncio.TimeoutError:
                        log(f"[{symbol}] Timeout ao solicitar histórico (tentativa {history_tries}).", "warning")
                        if history_tries >= 3:
                            raise Exception("Falha ao obter histórico após múltiplas tentativas")
                        await asyncio.sleep(1.0 + random.random() * 2.0)
                    except Exception as e:
                        log(f"[{symbol}] Erro ao obter histórico: {e}", "error")
                        if history_tries >= 3:
                            raise
                        await asyncio.sleep(1.0 + random.random() * 2.0)

                # processa dataframe inicial
                df = calcular_indicadores(df)
                save_last_candles(df, symbol)
                log(f"[{symbol}] Histórico inicial carregado ({len(df)} candles).")

                # subscribe ao feed de candles ao vivo
                sub_req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": CANDLE_INTERVAL * 60,
                    "end": "latest",
                    "subscribe": 1
                }
                await ws.send(json.dumps(sub_req))
                log(f"[{symbol}] Inscrito em candles ao vivo.")

                ultimo_candle_time = time.time()

                # Loop de recebimento robusto
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    except asyncio.TimeoutError:
                        # se ficar muito tempo sem candles, força reconexão
                        if time.time() - ultimo_candle_time > 300:
                            log(f"[{symbol}] Nenhum candle por >5min, forçando reconexão.", "warning")
                            raise Exception("Timeout prolongado, reconectar")
                        else:
                            # ping/keepalive scenario
                            log(f"[{symbol}] Timeout curto aguardando mensagem, mantendo conexão...", "info")
                            continue

                    # parse robusto
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        log(f"[{symbol}] Mensagem não JSON recebida, ignorando.", "warning")
                        continue

                    # extrair candle de forma resiliente
                    candle = None
                    if isinstance(msg, dict):
                        if "candle" in msg and isinstance(msg["candle"], dict):
                            candle = msg["candle"]
                        elif "ohlc" in msg and isinstance(msg["ohlc"], dict):
                            candle = msg["ohlc"]
                        elif "history" in msg and isinstance(msg["history"], dict) and isinstance(msg["history"].get("candles"), list):
                            last = msg["history"]["candles"][-1]
                            candle = last
                        elif "candles" in msg and isinstance(msg["candles"], list) and len(msg["candles"]) > 0:
                            candle = msg["candles"][-1]
                        # else: could be control message

                    if candle is None:
                        # mensagem de controle ou tick — log genérico e continue
                        if isinstance(msg, dict) and msg.get("msg_type"):
                            log(f"[{symbol}] msg_type recebida: {msg.get('msg_type')}", "info")
                        continue

                    # normaliza campos do candle
                    try:
                        epoch = int(candle.get("epoch"))
                        open_p = float(candle.get("open"))
                        high_p = float(candle.get("high"))
                        low_p = float(candle.get("low"))
                        close_p = float(candle.get("close"))
                        volume_p = float(candle.get("volume")) if candle.get("volume") is not None else 0.0
                    except Exception:
                        log(f"[{symbol}] Candle com campos inválidos, ignorando: {candle}", "warning")
                        continue

                    candle_time_str = datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
                    log(f"[{symbol}] Novo candle recebido: epoch={epoch} UTC | close={close_p}")

                    ultimo_candle_time = time.time()

                    # só adiciona se for um novo candle (closed)
                    if df.empty or int(df.iloc[-1]["epoch"]) != epoch:
                        df.loc[len(df)] = {
                            "epoch": epoch,
                            "open": open_p,
                            "high": high_p,
                            "low": low_p,
                            "close": close_p,
                            "volume": volume_p
                        }

                        df = calcular_indicadores(df)
                        save_last_candles(df, symbol)

                        # gerar sinal (1 por candle)
                        sinal = gerar_sinal(df, symbol)

                        if sinal:
                            # adicional anti-duplicate by time (safety)
                            now_ts = time.time()
                            last_ts = last_signal_time.get(symbol, 0)
                            if now_ts - last_ts < MIN_SECONDS_BETWEEN_SIGNALS:
                                log(f"[{symbol}] Sinal gerado muito próximo ao anterior ({now_ts-last_ts:.1f}s) — skip.", "warning")
                                continue

                            # monta mensagem
                            tipo = sinal["tipo"]
                            forca = sinal["forca"]
                            arrow = "🟢" if tipo == "COMPRA" else "🔴"
                            price = close_p

                            # Conversão para horário de Brasília (UTC-3). Observação: não lida com DST automaticamente.
                            entrada_br = datetime.utcfromtimestamp(epoch) - timedelta(hours=3)
                            entrada_str = entrada_br.strftime("%Y-%m-%d %H:%M:%S")

                            msg_final = (
                                f"📊 *NOVO SINAL — M{CANDLE_INTERVAL}*\n"
                                f"• Par: {symbol.replace('frx','')}\n"
                                f"• Direção: {arrow} *{tipo}*\n"
                                f"• Força do sinal: *{forca}%*\n"
                                f"• Preço: {price:.5f}\n"
                                f"• Horário de entrada (Brasília): {entrada_str}"
                            )

                            # marca timestamp e envia
                            last_signal_time[symbol] = time.time()
                            send_telegram(msg_final, symbol)

        except websockets.exceptions.ConnectionClosed as e:
            log(f"[WS {symbol}] ConnectionClosed: {e}", "warning")
            # fallthrough to backoff sleep and reconnect
        except Exception as e:
            log(f"[WS {symbol}] erro: {e}\n{traceback.format_exc()}", "error")

        # backoff antes de tentar reconectar
        reconnect_attempt = min(reconnect_attempt + 1, 10)
        backoff = min(60, (2 ** (reconnect_attempt if reconnect_attempt > 0 else 1)) * 0.5)
        jitter = random.uniform(0.5, 1.5)
        sleep_time = backoff * jitter
        log(f"[{symbol}] Reconectando em {sleep_time:.1f}s (attempt {reconnect_attempt})...", "info")
        await asyncio.sleep(sleep_time)

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Bot ativo — Lógica B (ajustada) — com força do sinal"

# ---------------- Execução ----------------
def run_flask():
    port = int(os.getenv("PORT", 10000))
    # Use Flask dev server here because você já roda assim no Render; em produção, trocar por gunicorn/uvicorn.
    app.run("0.0.0.0", port, debug=False, use_reloader=False)

async def main():
    # start flask in background thread
    threading.Thread(target=run_flask, daemon=True).start()
    # notify startup (if TG configurado)
    send_telegram("✅ Bot iniciado — Lógica B ajustada + Força do Sinal")

    # run all monitors concurrently
    await asyncio.gather(*(monitor_symbol(s) for s in SYMBOLS))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Encerrando...", "info")
```0
