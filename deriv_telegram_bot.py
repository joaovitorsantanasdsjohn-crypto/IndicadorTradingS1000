""" Deriv -> Telegram signal bot (skeleton)

Subscribes to Deriv websocket ticks

Aggregates 5-minute candles locally

Calculates EMA (short/medium/long), RSI, Bollinger

Sends formatted signals to Telegram when all 3 indicators agree


Requirements:

python 3.10+

pip install websockets pandas ta requests python-dotenv


Environment variables (recommended to set in Render / .env):

TELEGRAM_TOKEN  (your bot token)

TELEGRAM_CHAT_ID (your chat id)

DERIV_SYMBOL (default: frxEURUSD)

DERIV_APP_ID (optional)

EMA_SHORT (default: 8)

EMA_MED (default: 21)

EMA_LONG (default: 50)

RSI_PERIOD (default: 14)

BB_PERIOD (default: 20)

BB_STD (default: 2)

COOL_DOWN_MINUTES (default: 5)  # don't spam signals


Notes:

This file is a starting skeleton: robust logging, tests and production hardening are recommended before running with real money.

The bot uses ticks from Deriv and aggregates 5-minute candles locally to keep independence from Deriv candle endpoints and ensure exact control over candle close.


"""

import os 
import asyncio 
import json 
import time from collections 
import defaultdict, deque from datetime 
import datetime, timezone

import pandas as pd 
import requests 
import websockets from dotenv 
import load_dotenv

Optional: technical analysis helpers from ta package

from ta.trend 
import EMAIndicator from ta.momentum 
import RSIIndicator from ta.volatility 
import BollingerBands

load_dotenv()

=== Config ===

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") DERIV_SYMBOL = os.getenv("DERIV_SYMBOL", "frxEURUSD") DERIV_APP_ID = os.getenv("DERIV_APP_ID")  # optional

EMA_SHORT = int(os.getenv("EMA_SHORT", 8)) EMA_MED = int(os.getenv("EMA_MED", 21)) EMA_LONG = int(os.getenv("EMA_LONG", 50)) RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14)) BB_PERIOD = int(os.getenv("BB_PERIOD", 20)) BB_STD = float(os.getenv("BB_STD", 2)) COOLDOWN_MINUTES = int(os.getenv("COOL_DOWN_MINUTES", 5))

WEBSOCKET_URL = "wss://ws.derivws.com/websockets/v3" if DERIV_APP_ID: WEBSOCKET_URL += f"?app_id={DERIV_APP_ID}"

How many closed candles to keep for indicator history

CANDLE_HISTORY = 200

=== Globals / State ===

store partial candle aggregation: key = candle_start_epoch -> dict with o,h,l,c,count

partial_candles = {}

store closed candles in deque

closed_candles = deque(maxlen=CANDLE_HISTORY)

track last signal time to avoid duplicates

last_signal_time = None

=== Helper functions ===

def ts_to_epoch(ts_ms: int) -> int: return int(ts_ms // 1000)

def epoch_floor(epoch_s: int, period_s: int = 300) -> int: return (epoch_s // period_s) * period_s

def send_telegram_message(text: str): if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: print("Telegram token/chat_id not set. Skipping send.") return url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"} try: r = requests.post(url, json=payload, timeout=10) r.raise_for_status() except Exception as e: print("Failed to send telegram message:", e)

def format_signal_message(side: str, indicators: dict, candle_time_epoch: int) -> str: dt = datetime.fromtimestamp(candle_time_epoch, tz=timezone.utc) header = "🟢 <b>COMPRA</b>" if side == "buy" else "🔴 <b>VENDA</b>" lines = [header, f"⏱️ Candle (UTC): {dt.isoformat()}"] lines.append("") # EMA lines lines.append(f"EMA {EMA_SHORT}: {indicators['ema_short']:.5f}") lines.append(f"EMA {EMA_MED}: {indicators['ema_med']:.5f}") lines.append(f"EMA {EMA_LONG}: {indicators['ema_long']:.5f}") lines.append("") # RSI lines.append(f"RSI ({RSI_PERIOD}): {indicators['rsi']:.2f}") lines.append("") # Bollinger lines.append(f"Close: {indicators['close']:.5f}") lines.append(f"BB middle: {indicators['bb_mavg']:.5f}") lines.append(f"BB lower: {indicators['bb_low']:.5f}") lines.append(f"BB upper: {indicators['bb_high']:.5f}") lines.append("") lines.append("Fonte: Deriv · TIMEFRAME: 5m · Operação: FTT 5 minutos") return "\n".join(lines)

def evaluate_indicators(df: pd.DataFrame) -> tuple: """Returns (side, indicators_dict) where side is 'buy', 'sell' or None""" # need enough data if len(df) < max(EMA_LONG, BB_PERIOD, RSI_PERIOD) + 5: return None, {}

close = df['close']

# EMA
ema_short = EMAIndicator(close, window=EMA_SHORT).ema_indicator()  # pandas Series
ema_med = EMAIndicator(close, window=EMA_MED).ema_indicator()
ema_long = EMAIndicator(close, window=EMA_LONG).ema_indicator()

# RSI
rsi = RSIIndicator(close, window=RSI_PERIOD).rsi()

# Bollinger
bb = BollingerBands(close, window=BB_PERIOD, window_dev=BB_STD)
bb_mavg = bb.bollinger_mavg()
bb_high = bb.bollinger_hband()
bb_low = bb.bollinger_lband()

# take last (most recent closed candle)
idx = -1
last_close = float(close.iloc[idx])
last_ema_short = float(ema_short.iloc[idx])
last_ema_med = float(ema_med.iloc[idx])
last_ema_long = float(ema_long.iloc[idx])
last_rsi = float(rsi.iloc[idx])
last_bb_mavg = float(bb_mavg.iloc[idx])
last_bb_high = float(bb_high.iloc[idx])
last_bb_low = float(bb_low.iloc[idx])

# Define strict rules for agreement
buy_conditions = [
    last_ema_short > last_ema_med > last_ema_long,  # trend up
    last_rsi < 40,  # oversold-ish -> reversals (strict rule: below 40)
    last_close < last_bb_low  # price below lower band
]

sell_conditions = [
    last_ema_short < last_ema_med < last_ema_long,  # trend down
    last_rsi > 60,  # overbought-ish -> reversals (strict rule: above 60)
    last_close > last_bb_high  # price above upper band
]

if all(buy_conditions):
    indicators = {
        'ema_short': last_ema_short,
        'ema_med': last_ema_med,
        'ema_long': last_ema_long,
        'rsi': last_rsi,
        'bb_mavg': last_bb_mavg,
        'bb_high': last_bb_high,
        'bb_low': last_bb_low,
        'close': last_close,
    }
    return 'buy', indicators

if all(sell_conditions):
    indicators = {
        'ema_short': last_ema_short,
        'ema_med': last_ema_med,
        'ema_long': last_ema_long,
        'rsi': last_rsi,
        'bb_mavg': last_bb_mavg,
        'bb_high': last_bb_high,
        'bb_low': last_bb_low,
        'close': last_close,
    }
    return 'sell', indicators

return None, {}

=== Candle aggregation ===

def process_tick(price: float, epoch_s: int): """Aggregate incoming tick into 5-minute candles. When a candle closes, move it to closed_candles.""" global partial_candles, closed_candles candle_start = epoch_floor(epoch_s, 300) # update the candle c = partial_candles.get(candle_start) if c is None: # initialize with open=high=low=close=price partial_candles[candle_start] = {"open": price, "high": price, "low": price, "close": price, "count": 1} else: c['high'] = max(c['high'], price) c['low'] = min(c['low'], price) c['close'] = price c['count'] += 1

# check for closed candles: any candle with start < current candle_start is closed
closed = [s for s in partial_candles.keys() if s < candle_start]
for s in sorted(closed):
    closed_candles.append({
        'epoch': s,
        'open': partial_candles[s]['open'],
        'high': partial_candles[s]['high'],
        'low': partial_candles[s]['low'],
        'close': partial_candles[s]['close']
    })
    del partial_candles[s]

async def handle_deriv_ws(): global last_signal_time reconnect_delay = 1 while True: try: print(f"Connecting to Deriv WS: {WEBSOCKET_URL}") async with websockets.connect(WEBSOCKET_URL, ping_interval=20, ping_timeout=10) as ws: # subscribe to ticks for symbol req = {"ticks": DERIV_SYMBOL, "subscribe": 1} await ws.send(json.dumps(req)) print("Subscribed to ticks for", DERIV_SYMBOL)

# simple keepalive ping every 28s (Deriv recommends regular pings)
            async def keepalive():
                while True:
                    try:
                        await ws.send(json.dumps({"ping": 1}))
                    except Exception:
                        return
                    await asyncio.sleep(28)

            ka_task = asyncio.create_task(keepalive())

            async for msg in ws:
                data = json.loads(msg)
                # skip non-tick messages
                if 'error' in data:
                    print('Error from WS:', data)
                    continue
                # ticks return with key 'tick' and 'quote'
                tick = data.get('tick')
                if not tick:
                    continue
                # price is tick['quote'] and epoch is tick['epoch']
                price = float(tick['quote'])
                epoch_s = int(tick['epoch'])
                process_tick(price, epoch_s)

                # If a new closed candle just appended, evaluate
                if closed_candles:
                    # build DataFrame from closed candles
                    df = pd.DataFrame(list(closed_candles))
                    # ensure chronological order
                    df = df.sort_values('epoch')
                    df = df.reset_index(drop=True)
                    # we need a column 'close'
                    df.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low'}, inplace=True)

                    side, indicators = evaluate_indicators(df)
                    if side:
                        # get last closed candle epoch (most recent)
                        candle_epoch = int(df['epoch'].iloc[-1])
                        now = time.time()
                        if last_signal_time and (now - last_signal_time) < (COOLDOWN_MINUTES * 60):
                            # cooldown active
                            print('Signal found but in cooldown window. Skipping.')
                        else:
                            msg_text = format_signal_message(side, indicators, candle_epoch)
                            send_telegram_message(msg_text)
                            last_signal_time = time.time()

            ka_task.cancel()
    except Exception as e:
        print(f"Websocket error: {e}. Reconnecting in {reconnect_delay}s...")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)

def main(): print("Starting Deriv -> Telegram bot skeleton") print(f"Symbol: {DERIV_SYMBOL} · EMA: {EMA_SHORT}/{EMA_MED}/{EMA_LONG} · RSI: {RSI_PERIOD} · BB: {BB_PERIOD}@{BB_STD}") try: asyncio.run(handle_deriv_ws()) except KeyboardInterrupt: print("Shutting down")

if name == 'main': main()

