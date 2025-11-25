import time
import json
import pandas as pd
import numpy as np
import requests  # for tele alerts
import yfinance as yf
from datetime import datetime, timedelta, timezone
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
import sys
import builtins  # for flushing prints to the main controller
import math
from numba import njit
import threading
import os
import pickle

# alpaca imports, be careful with how its named in documentation.
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from dotenv import load_dotenv

load_dotenv()

print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)

# api
API_KEY_ID = os.getenv("APCA_API_KEY_ID")
API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

if not all([API_KEY_ID, API_SECRET_KEY, bot_token, chat_id]):
    print("missing API keys. check the environment variables")
    sys.exit(1)

paper_setting = True
adf_max = 0.2
hurst_max = 0.8
# these may seem random, but i assure you, they are empirically picked based on my WFA results.

if len(sys.argv) != 3:
    print("Usage: python trader.py <ASSET_A> <ASSET_B>")
    sys.exit(1)

ASSET_A = sys.argv[1]
ASSET_B = sys.argv[2]

# keep this here so i can paste it back everytime i remove above to debug through trader.py without goin thru the controller
# if len(sys.argv) != 3:
#   print("Usage: python trader.py <ASSET_A> <ASSET_B>")
#   sys.exit(1)

# ASSET_A = sys.argv[1]
# ASSET_B = sys.argv[2]

# FOR DEBUG
# ASSET_A = "CRL"
# ASSET_B = "IQV"

print(f"Trader started for pair: {ASSET_A}/{ASSET_B}")

# settings
trade_interval = 90
sleepSeconds = 180
tele_interval = 180
assigned_cptl = 100000  # TODO: check literature for recommended allocation for capital in relation to total capital

Z_ENTRY = None
Z_EXIT = None
Z_STOP_LOSS = None
LOOKBACK_WINDOW = None

last_heartbeat_time = datetime.min.replace(tzinfo=timezone.utc)
last_trade_time = datetime.min.replace(tzinfo=timezone.utc)
cooldown_time = 300

data_client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
trading_client = TradingClient(API_KEY_ID, API_SECRET_KEY, paper=paper_setting)

entry_price_a = None
entry_price_b = None
entry_pos_type = None

PERSISTENCE_DIR = '/tmp/trader_data'
BAR_CACHE_PATH = os.path.join(PERSISTENCE_DIR, f'cache_{ASSET_A}_{ASSET_B}.pkl')
STATE_FILE_PATH = os.path.join(PERSISTENCE_DIR, f'state_{ASSET_A}_{ASSET_B}.json')
GLOBAL_DATA_CACHE = None


def ensure_persistence_dir():
    os.makedirs(PERSISTENCE_DIR, exist_ok=True)

def save_state():
    global entry_price_a, entry_price_b, entry_pos_type
    try:
        ensure_persistence_dir()
        state = {
            "entry_price_a": entry_price_a,
            "entry_price_b": entry_price_b,
            "entry_pos_type": entry_pos_type,
            "last_save_time": datetime.now(timezone.utc).isoformat()
        }
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"!!! Warning: Failed to save state to {STATE_FILE_PATH}: {e}")

def load_state():
    global entry_price_a, entry_price_b, entry_pos_type
    if not os.path.exists(STATE_FILE_PATH):
        return

    try:
        with open(STATE_FILE_PATH, 'r') as f:
            state = json.load(f)

        if state.get("entry_price_a") is not None:
             entry_price_a = float(state["entry_price_a"])
        if state.get("entry_price_b") is not None:
             entry_price_b = float(state["entry_price_b"])
        entry_pos_type = state["entry_pos_type"]

        if entry_pos_type is not None:
             print(f"Loaded persistent state: {entry_pos_type} entered at A={entry_price_a}, B={entry_price_b}")

    except Exception as e:
        print(f"!!! Warning: Failed to load state from {STATE_FILE_PATH}: {e}")

def clear_state():
    global entry_price_a, entry_price_b, entry_pos_type
    entry_price_a = None
    entry_price_b = None
    entry_pos_type = None
    if os.path.exists(STATE_FILE_PATH):
        try:
            os.remove(STATE_FILE_PATH)
            print(f"Cleared persistent state file: {STATE_FILE_PATH}")
        except Exception as e:
            print(f"!!! Warning: Failed to delete state file: {e}")

def save_bar_cache():
    global GLOBAL_DATA_CACHE
    if GLOBAL_DATA_CACHE is None or GLOBAL_DATA_CACHE.empty:
        return
    try:
        ensure_persistence_dir()
        with open(BAR_CACHE_PATH, 'wb') as f:
            pickle.dump(GLOBAL_DATA_CACHE, f)
    except Exception as e:
        print(f"!!! Warning: Failed to save bar cache to {BAR_CACHE_PATH}: {e}")

def load_bar_cache():
    global GLOBAL_DATA_CACHE
    if not os.path.exists(BAR_CACHE_PATH):
        GLOBAL_DATA_CACHE = None
        return

    try:
        with open(BAR_CACHE_PATH, 'rb') as f:
            GLOBAL_DATA_CACHE = pickle.load(f)
            if not isinstance(GLOBAL_DATA_CACHE, pd.DataFrame):
                 GLOBAL_DATA_CACHE = None
                 print("Loaded cache was invalid, starting fresh.")
            else:
                print(f"Loaded cache for {ASSET_A}/{ASSET_B} with {len(GLOBAL_DATA_CACHE)} bars.")

    except Exception as e:
        print(f"!!! Warning: Failed to load bar cache: {e}. Starting fresh.")
        GLOBAL_DATA_CACHE = None

def send_tele(message, alert_type="INFO"):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    msgTypeMap = {
        "INFO": "",
        "SUCCESS": "",
        "ENTRY": "",
        "EXIT": "",
        "STOP": "",
        "ERROR": "",
        "WARNING": "",
        "MARKET": ""
    }

    full_message = f"{msgTypeMap.get(alert_type, '')} *[{alert_type}]*\n{message}"

    payload = {
        'chat_id': chat_id,
        'text': full_message,
    }

    try:
        response = requests.post(url, data=payload, timeout=10)

        if response.status_code != 200:
            print(f"Telegram API response: {response.status_code} {response.reason}")
            try:
                print(f"    Response body: {response.text}")
            except Exception:
                pass

        response.raise_for_status()
        print(f"Telegram Alert Sent ({alert_type}).")

    except requests.exceptions.HTTPError as http_err:
        print(f"!!! telegram failed to send msg due to HTTP error: ({http_err.response.status_code})")
        try:
            print(f"    Response: {http_err.response.text}")
        except Exception:
            pass
    except requests.exceptions.RequestException as e:
        print(f"telegram failed to send msg due to: {e}")


def load_param(file_path='optimized_params.json'):
    global Z_ENTRY, Z_EXIT, Z_STOP_LOSS, LOOKBACK_WINDOW

    try:
        with open(file_path, 'r') as f:
            params = json.load(f)

        LOOKBACK_WINDOW = params.get('metadata', {}).get('rolling_window_bars', 1000)

        key = f"{ASSET_A}/{ASSET_B}"

        if key in params:
            Z_ENTRY = params[key]["z_entry"]
            Z_EXIT = params[key]["z_exit"]
            Z_STOP_LOSS = params[key]["z_sl"]
            print(f"! loaded optimized parameters for {ASSET_A}/{ASSET_B}:")
            print(
                f"   Z_ENTRY={Z_ENTRY}, Z_EXIT={Z_EXIT}, Z_STOP_LOSS={Z_STOP_LOSS}, LOOKBACK_WINDOW={LOOKBACK_WINDOW}")

            return True
        else:
            print("no params found")

    except FileNotFoundError:
        print(f" json params file '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"failed to load parameters from JSON: {e}")
        return False


# read documentation for alpaca paper which APPARENTLY trades at the NBBO or wtv its called
# so technically i can get around bad alpaca IEX with like, using yfinance as the ... 'eyes'? and alpaca with the hands
# THis is a frankenstein setup which now relies on two systems to run happily but istg im tired of the bad iex data
def get_price(asset_a, asset_b):
    symbols = [asset_a, asset_b]
    price_map = {asset_a: np.nan, asset_b: np.nan}

    print(f"Fetching latest prices for {asset_a}/{asset_b} via Yahoo Finance...")

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            price = ticker.fast_info.get('last_price')

            if price is None or np.isnan(price):
                todays_data = ticker.history(period='1d', interval='1m')
                if not todays_data.empty:
                    price = todays_data['Close'].iloc[-1]

            if price is not None and price > 0:
                price_map[sym] = float(price)
                print(f"Price {sym}: ${price:.2f}")
            else:
                print(f"Could not find price for {sym}")

        except Exception as e:
            print(f"!!!Error fetching {sym}: {e}")

    return float(price_map[asset_a]), float(price_map[asset_b])


def get_history_from_yahoo(symbol, start_date, end_date, timeframe):
    if timeframe.unit == TimeFrameUnit.Minute:
        if timeframe.amount == 15:
            yf_interval = "15m"
        elif timeframe.amount == 5:
            yf_interval = "5m"
        else:
            yf_interval = "1m"
    else:
        yf_interval = "1d"

    max_days_limit = 59
    min_start_date = datetime.now() - timedelta(days=max_days_limit)

    if start_date < min_start_date and "m" in yf_interval:
        print(
            f"{symbol}: Request ({start_date.date()}) exceeds Yahoo 60d limit. Capping to {min_start_date.date()}.")
        start_date = min_start_date

    try:
        ticker = yf.Ticker(symbol)
        bars = ticker.history(start=start_date - timedelta(minutes=1), end=end_date, interval=yf_interval)

        if bars.empty:
            print(f"No history found for {symbol}")
            return None

        return bars['Close']

    except Exception as e:
        print(f"    [YF] History fetch failed for {symbol}: {e}")
        return None

def filters_data(asset_a, asset_b, lookback, timeframe=TimeFrame(1, TimeFrameUnit.Minute)):

    global GLOBAL_DATA_CACHE

    # load cache on first run if not already loaded (initialization phase)
    if GLOBAL_DATA_CACHE is None:
        load_bar_cache()

    end_date = datetime.now()
    max_history_days = 58


    if GLOBAL_DATA_CACHE is not None and not GLOBAL_DATA_CACHE.empty:
        last_ts = GLOBAL_DATA_CACHE.index[-1].to_pydatetime()
        start_date = last_ts + timedelta(minutes=1)
        print(f"CACHE HIT: Fetching new bars from {start_date.strftime('%Y-%m-%d %H:%M')}")
    else:
        start_date = end_date - timedelta(days=max_history_days)
        print(f"CACHE MISS: Performing initial {max_history_days}-day fetch from {start_date.strftime('%Y-%m-%d %H:%M')}")


    try:
        series_a = get_history_from_yahoo(asset_a, start_date, end_date, timeframe)
        series_b = get_history_from_yahoo(asset_b, start_date, end_date, timeframe)

        new_data = None
        if series_a is not None and series_b is not None:
            new_data = pd.concat([series_a, series_b], axis=1).dropna()
            new_data.columns = [f'Close_{asset_a}', f'Close_{asset_b}']

        if new_data is not None and not new_data.empty:
            if GLOBAL_DATA_CACHE is None:
                GLOBAL_DATA_CACHE = new_data
            else:
                GLOBAL_DATA_CACHE = pd.concat([GLOBAL_DATA_CACHE, new_data], axis=0).drop_duplicates(
                    keep='last')
                GLOBAL_DATA_CACHE.sort_index(inplace=True)

            save_bar_cache()


        if GLOBAL_DATA_CACHE is None or GLOBAL_DATA_CACHE.empty:
            print("Failed to retrieve sufficient historical data from Yahoo/Cache.")
            return None

        data = GLOBAL_DATA_CACHE.copy()
        data = data.ffill().dropna()

        if data.index.tz is None:
            data = data.tz_localize('US/Eastern')
        else:
            data.index = data.index.tz_convert('US/Eastern')

        rth_filter = ((data.index.hour > 9) | ((data.index.hour == 9) & (data.index.minute >= 30))) & (data.index.hour < 16)
        data = data[rth_filter]

        MAX_BARS_TO_KEEP = 60 * 24 * 60
        if len(data) > MAX_BARS_TO_KEEP:
            data = data.iloc[-MAX_BARS_TO_KEEP:]
            GLOBAL_DATA_CACHE = data.copy()

        data = data.iloc[-(lookback + 50):]

        if data.empty or len(data) < 200:
             print(f"!!! Warning: Only {len(data)} bars after filtering/truncating.")
             return None

        data['Log_A'] = np.log(data[f'Close_{asset_a}'])
        data['Log_B'] = np.log(data[f'Close_{asset_b}'])

        return data

    except Exception as e:
        print(f"filters_data() error: {e}")
        return None

@njit
def calc_kalman(y, x, delta=1e-4, ve=1e-3):
    n = len(y)
    beta = np.zeros(n)
    state_mean = 0.0
    state_cov = 0.1

    for t in range(n):
        state_cov = state_cov + delta
        obs_val = y[t]
        obs_mat = x[t]
        pred_obs = obs_mat * state_mean
        residual = obs_val - pred_obs
        variance = (obs_mat * state_cov * obs_mat) + ve
        kalman_gain = (state_cov * obs_mat) / variance
        state_mean = state_mean + kalman_gain * residual
        state_cov = (1 - kalman_gain * obs_mat) * state_cov
        beta[t] = state_mean

    return beta


@njit
def calc_hurst(series, window=100):
    n = len(series)
    hurst = np.full(n, 0.5)
    min_window = max(window, 30)

    for t in range(min_window, n):
        ts = series[t - window:t]
        mean_ts = np.mean(ts)
        centered = ts - mean_ts
        cumulative = np.cumsum(centered)
        R = np.max(cumulative) - np.min(cumulative)
        S = np.std(ts)
        if S == 0:
            hurst[t] = 0.5
        else:
            hurst[t] = np.log(R / S) / np.log(len(ts))

    return hurst


def calculateSignal(data):
    Y = data['Log_A'].values
    X = data['Log_B'].values
    beta_series = calc_kalman(Y, X)

    latest_beta = beta_series[-1]

    data['Kalman_Beta'] = beta_series
    data['Spread'] = data['Log_A'] - data['Kalman_Beta'] * data['Log_B']

    spread_values = data['Spread'].values
    hurst_val = calc_hurst(spread_values, window=100)[-1]

    # change this z_window with increasing it to make the trader more reactive to z-score. found it to be ok at 100? follows the json
    z_window = 100
    spread_tail = data['Spread'].iloc[-z_window:]

    mean = spread_tail.mean()
    std = spread_tail.std()
    latest_spread = spread_tail.iloc[-1]

    z_score = (latest_spread - mean) / std if std > 0 else np.nan

    # adf check (legacy) quite useless now tbh considering i have kalman which literally makes adf like close to 0 for everything... can consider removing to improve performance time as i heard adfuller is a pretty heavy fn
    try:
        adf_p = adfuller(data['Spread'].dropna().tail(100), autolag='AIC')[1]
    except:
        adf_p = 1.0

    return latest_beta, z_score, std, latest_spread, adf_p, hurst_val

def getCurrentPos(asset_a, asset_b):
    try:
        positions = trading_client.get_all_positions()

        pos_a = next((p for p in positions if p.symbol == asset_a), None)
        pos_b = next((p for p in positions if p.symbol == asset_b), None)

        qty_a, qty_b = 0.0, 0.0

        if pos_a is not None:
            qty_a = float(pos_a.qty_available)
        if pos_b is not None:
            qty_b = float(pos_b.qty_available)

        # check if pos open
        if (pos_a is not None or pos_b is not None) and (qty_a != 0 or qty_b != 0):

            if qty_a > 0 and qty_b < 0:
                return 1, qty_a, qty_b
            elif qty_a < 0 and qty_b > 0:
                return -1, qty_a, qty_b

        return 0, 0, 0

    except Exception as e:
        print(f"!!! Error checking position: {e}")
        return 0, 0, 0


def print_pnl_stats():
    equity = np.nan
    pnl_today = np.nan
    try:
        # get acc info
        account = trading_client.get_account()

        # get todays pnl and total equity
        equity = float(account.equity)
        try:
            pnl_today = float(account.equity) - float(account.last_equity)  # approximation
        except AttributeError:
            pnl_today = np.nan  # use nan if the attribute is missing

        print("\n**- TRADE CLOSED: PNL REPORT -**")
        print(f"current equity: ${equity:,.2f}")
        print(f"today's PnL: ${pnl_today:,.2f}")

        return equity, pnl_today

    except Exception as e:
        print(f"!!! Warning: Could not fetch PnL stats from Alpaca: {e}")
        return equity, pnl_today


def determine_sizing(asset_a_price, asset_b_price, beta, spread_volatility):
    """determines the posiiton size for a trade. rounds the shares to integers since i cant short fractional shares"""
    global assigned_cptl

    if np.isnan(beta) or spread_volatility <= 0 or asset_a_price <= 0 or asset_b_price <= 0:
        return 0.0, 0.0, 0, 0, 0.0, 0.0

    vol_benchmark = 0.15
    vol_scale = min(1.0, vol_benchmark / spread_volatility) if spread_volatility > 0 else 1.0
    V_Total = assigned_cptl * vol_scale

    # beta allocation
    b = abs(beta)
    V_A = V_Total * (b / (1.0 + b))
    V_B = V_Total - V_A

    raw_shares_a = V_A / asset_a_price
    raw_shares_b = V_B / asset_b_price

    # round it so I DONT SHORT fractional shares which alpaca does not like
    qty_a = int(math.floor(raw_shares_a))
    qty_b = int(math.floor(raw_shares_b))

    # avoid zero-qty tiny orders: if either qty is 0 skip trading. but unlikely to happen considering my capital per trade is 10k which is plenty..
    if qty_a == 0 or qty_b == 0:
        return raw_shares_a, raw_shares_b, qty_a, qty_b, V_A, V_B

    actual_V_A = qty_a * asset_a_price
    actual_V_B = qty_b * asset_b_price

    return raw_shares_a, raw_shares_b, qty_a, qty_b, actual_V_A, actual_V_B


def submit_order(symbol, qty, side, limit_price=None, order_type='LIMIT'):
    default_return = (False, 0.0, None, 0.0)

    max_wait_seconds = 30 if order_type == 'LIMIT' else 10
    retry_interval = 0.5  # Start at 0.5s
    max_retries = int(max_wait_seconds / retry_interval) * 2

    try:
        if qty <= 0:
            return default_return

        order = None
        if order_type == 'MARKET':
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.FOK
            )
            order = trading_client.submit_order(request)
            print(f"Placed MARKET {side.value} {qty} {symbol}")

        else:
            if limit_price is None:
                quote = data_client.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                ).get(symbol)

                bid = getattr(quote, "bid_price", None)
                ask = getattr(quote, "ask_price", None)

                if side == OrderSide.BUY:
                    limit_price = ask
                else:
                    limit_price = bid

            if limit_price is None or limit_price <= 0:
                print(f"Limit price could not be determined for {symbol}. Aborting order.")
                return default_return

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                limit_price=limit_price,
                time_in_force=TimeInForce.GTC
            )

            order = trading_client.submit_order(request)
            print(f"Placed LIMIT {side.value} {qty} {symbol} @ {limit_price:.2f}")

        filled_qty = 0.0
        filled_avg_price = 0.0

        for i in range(max_retries):
            wait_time = min(retry_interval * (2 ** i), 5.0)
            if i > 0:
                time.sleep(wait_time)

            order_status = trading_client.get_order_by_id(order.id)
            filled_qty = float(order_status.filled_qty)
            filled_avg_price = float(getattr(order_status, 'filled_avg_price', 0.0))

            if order_status.status in ["filled", "canceled", "rejected", "expired"]:
                break

            if filled_qty > 0 and filled_qty < qty:
                print(f"Partial Fill: {filled_qty}/{qty}")

            if i == max_retries - 1 and order_status.status == "working":
                print(f"Order {order.id} for {symbol} still working after timeout. Forcing cancel.")
                trading_client.cancel_order(order.id)
                time.sleep(1)
                order_status = trading_client.get_order_by_id(order.id)
                filled_qty = float(order_status.filled_qty)
                break

        order_status = trading_client.get_order_by_id(order.id)

        if filled_qty == 0.0:
            print(f"Order {order.id} for {symbol} failed with 0 fill. Status: {order_status.status}.")
            send_tele(f"Order {order.id} for {symbol} failed (0 fill). Status {order_status.status}.",
                      alert_type="ERROR")
            return default_return

        if filled_qty < qty and order_status.status == "working":
            # this should ideally be caught by the timeout logic above, but added for safety
            trading_client.cancel_order(order.id)
            print(f"Canceled remaining working order for {symbol}: {order.id}")
            time.sleep(1)  # Final wait
            order_status = trading_client.get_order_by_id(order.id)

        print(f"Final Fill: {filled_qty}/{qty} @ ${filled_avg_price:.2f} (ID: {order.id})")

        if filled_qty > 0:
            send_tele(
                f"{order_type} order {side.value} {symbol} executed: {filled_qty}/{qty} shares @ {filled_avg_price:.2f}",
                alert_type="INFO"
            )

        return True, filled_qty, order.id, filled_avg_price

    except Exception as e:
        print(f"CRITICAL: {order_type} order failed for {symbol}: {e}")
        send_tele(f"CRITICAL: {order_type} order failed for {symbol}: {e}", alert_type="ERROR")
        return default_return


def place_order_thread(result_dict, key, symbol, qty, side, order_type='LIMIT'):
    ok, filled, oid, avg = submit_order(symbol, qty, side, order_type=order_type)
    result_dict[key] = {"ok": ok, "filled": filled, "order_id": oid, "avg": avg, "qty": qty, "side": side}


def enter_pair_atomic(qty_a, qty_b, current_z, beta, hurst, spread_volatility, lastPriceA, lastPriceB,
                      entry_type, order_type='LIMIT'):
    global last_trade_time, entry_pos_type, entry_price_a, entry_price_b

    side_a = OrderSide.BUY if entry_type == "LONG_SPREAD_ENTRY" else OrderSide.SELL
    side_b = OrderSide.SELL if entry_type == "LONG_SPREAD_ENTRY" else OrderSide.BUY

    results = {}
    t1 = threading.Thread(target=place_order_thread,
                          args=(results, 'A', ASSET_A, qty_a, side_a, order_type))
    t2 = threading.Thread(target=place_order_thread,
                          args=(results, 'B', ASSET_B, qty_b, side_b, order_type))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    res_a = results.get('A', {"filled": 0.0, "qty": qty_a, "side": side_a})
    res_b = results.get('B', {"filled": 0.0, "qty": qty_b, "side": side_b})

    filled_a, filled_b = res_a['filled'], res_b['filled']

    if filled_a == qty_a and filled_b == qty_b:
        print("Atomic Entry SUCCESS: Both legs fully filled.")
        entry_price_a = lastPriceA
        entry_price_b = lastPriceB
        entry_pos_type = entry_type
        save_state()

        alert_msg = (
            f"New Pair Trade Entered: {entry_type.replace('_', ' ')}.\n"
            f"Z-Score: {current_z:.4f} | Beta: {beta:.4f} | Hurst: {hurst:.4f}\n"
            f"Orders Filled: {ASSET_A} ({filled_a:.0f} shares), {ASSET_B} ({filled_b:.0f} shares)"
        )
        send_tele(alert_msg, alert_type="ENTRY")
        last_trade_time = datetime.now(timezone.utc)
        return True

    if filled_a > 0 or filled_b > 0:
        print("Atomic Entry FAILURE: Asymmetric/Partial fill detected. IMMEDIATELY neutralizing residual positions.")

        qty_to_close_a = filled_a
        qty_to_close_b = filled_b

        if qty_to_close_a > 0:
            opposite_a = OrderSide.SELL if res_a['side'] == OrderSide.BUY else OrderSide.BUY
            print(f"ROLLBACK: Closing {qty_to_close_a:.0f} of {ASSET_A} via MARKET.")
            submit_order(ASSET_A, int(abs(qty_to_close_a)), opposite_a, order_type='MARKET')

        if qty_to_close_b > 0:
            opposite_b = OrderSide.SELL if res_b['side'] == OrderSide.BUY else OrderSide.BUY
            print(f"ROLLBACK: Closing {qty_to_close_b:.0f} of {ASSET_B} via MARKET.")
            submit_order(ASSET_B, int(abs(qty_to_close_b)), opposite_b, order_type='MARKET')

        send_tele(
            f"CRITICAL: Failed to enter {ASSET_A}/{ASSET_B} atomically. Rollback attempted.\n"
            f"Requested: A={qty_a}, B={qty_b} | Filled: A={filled_a:.0f}, B={filled_b:.0f}",
            alert_type="ERROR"
        )

    else:
        print("Atomic Entry FAILURE: Zero fill on both legs. No residual positions to clean up.")

    return False


# USE MARKET ORDER for liquidate becuse i dont wanna be caught holding only one side of the spread if my limit order fails
def liquidate(reason="No reason provided."):
    global entry_price_a, entry_price_b, entry_pos_type

    print(f"[LIQUIDATE] entry_price_a={entry_price_a}, entry_price_b={entry_price_b}, entry_pos_type={entry_pos_type}")

    lastPriceA, lastPriceB = get_price(ASSET_A, ASSET_B)
    position, qty_a_open, qty_b_open = getCurrentPos(ASSET_A, ASSET_B)

    if position == 0:
        print("No open position to liquidate.")
        send_tele(
            f"Liquidate called for {ASSET_A}/{ASSET_B} but no open pair position was found.\nReason: {reason}",
            alert_type="EXIT"
        )
        clear_state()
        return

    equity_pre, pnl_today_pre = print_pnl_stats()

    qty_a_initial, qty_b_initial = qty_a_open, qty_b_open

    def close_legs(qty_a, qty_b):
        filled_a, filled_b = 0, 0
        if qty_a != 0:
            side_a = OrderSide.SELL if qty_a > 0 else OrderSide.BUY
            _, filled_a, _, _ = submit_order(ASSET_A, int(abs(qty_a)), side_a, order_type='MARKET')

        if qty_b != 0:
            side_b = OrderSide.SELL if qty_b > 0 else OrderSide.BUY
            _, filled_b, _, _ = submit_order(ASSET_B, int(abs(qty_b)), side_b, order_type='MARKET')
        return filled_a, filled_b

    close_legs(qty_a_initial, qty_b_initial)
    time.sleep(5)

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        position_post, qty_a_final, qty_b_final = getCurrentPos(ASSET_A, ASSET_B)

        if position_post == 0:
            print(f"Liquidation successful on attempt {attempt}.")
            break

        print(f"!!! Warning: Position still open after attempt {attempt}. Retrying MARKET close for residuals.")
        print(f"Remaining: {ASSET_A}: {qty_a_final:.0f}, {ASSET_B}: {qty_b_final:.0f}")

        close_legs(qty_a_final, qty_b_final)

        if attempt < MAX_RETRIES:
            time.sleep(5)

    try:
        position_post, qty_a_final, qty_b_final = getCurrentPos(ASSET_A, ASSET_B)

        if position_post != 0:
            raise Exception("POSITION_STILL_OPEN")

        print(f"LIQUIDATED: Closed all positions for {ASSET_A} and {ASSET_B}.")
        equity_post, pnl_today_post = print_pnl_stats()

        pnl_pair = None
        if entry_price_a is not None and entry_price_b is not None and entry_pos_type is not None:
            if entry_pos_type == "LONG_SPREAD_ENTRY":
                pnl_a = (lastPriceA - entry_price_a) * abs(qty_a_initial)
                pnl_b = (entry_price_b - lastPriceB) * abs(qty_b_initial)
            elif entry_pos_type == "SHORT_SPREAD_ENTRY":
                pnl_a = (entry_price_a - lastPriceA) * abs(qty_a_initial)
                pnl_b = (lastPriceB - entry_price_b) * abs(qty_b_initial)
            else:
                pnl_a = pnl_b = 0.0

            pnl_pair = pnl_a + pnl_b

            print(f"Position Closed: {ASSET_A}/{ASSET_B}")
            print(f"    {ASSET_A}: Entry ${entry_price_a:.2f}, Exit ${lastPriceA:.2f}, Qty {abs(qty_a_initial)}")
            print(f"    {ASSET_B}: Entry ${entry_price_b:.2f}, Exit ${lastPriceB:.2f}, Qty {abs(qty_b_initial)}")
            print(f"PnL = ${pnl_pair:,.2f}")

            send_tele(
                f"Pair {ASSET_A}/{ASSET_B} closed.\n"
                f"PnL for trade = ${pnl_pair:,.2f}\n"
                f"Reason: {reason}",
                alert_type="EXIT"
            )
        else:
            send_tele(
                f"Pair {ASSET_A}/{ASSET_B} closed, but entry details missing (bot restart / not tracked). "
                f"Exit prices: {ASSET_A} ${lastPriceA:.2f}, {ASSET_B} ${lastPriceB:.2f}\nReason: {reason}",
                alert_type="EXIT"
            )

        clear_state()


    except Exception as e:

        if str(e) == "POSITION_STILL_OPEN":

            position_final, qty_a_final, qty_b_final = getCurrentPos(ASSET_A, ASSET_B)

            print(f"!!! CRITICAL FAILURE: Liquidation failed after {MAX_RETRIES} attempts.")
            print(f"Remaining Position: {ASSET_A}: {qty_a_final:.0f}, {ASSET_B}: {qty_b_final:.0f}")

            send_tele(
                f"!!! CRITICAL WARNING: Liquidation FAILED after {MAX_RETRIES} retries for {ASSET_A}/{ASSET_B}.\n"
                f"ONE OR BOTH LEGS ARE STILL OPEN.\n"
                f"Remaining Position: {ASSET_A}: {qty_a_final:.0f}, {ASSET_B}: {qty_b_final:.0f}",
                alert_type="ERROR"
            )

        else:
            print(f"!!! Liquidation error: {e}")
            send_tele(f"Liquidation failed for {ASSET_A}/{ASSET_B}: {e}", alert_type="ERROR")


def log_status(current_z, beta, p_value, hurst, price_a, price_b, position, pos_a, pos_b, acc_eqty,
               loop_start_time, is_rth):
    global last_heartbeat_time

    rth_status = "OPEN (RTH)" if is_rth else "CLOSED"

    status_msg = (
        f"\n{loop_start_time.strftime('%Y-%m-%d %H:%M:%S')} | MARKET: {rth_status} | Z: {current_z:.4f} | Beta: {beta:.4f} | ADF P: {p_value:.4f} | Hurst: {hurst:.4f}\n"
        f"Price {ASSET_A}: ${price_a:.2f} | Price {ASSET_B}: ${price_b:.2f}\n"
        f"Position: {position} ({ASSET_A}: {pos_a:.0f} shares, {ASSET_B}: {pos_b:.0f} shares) | Equity: ${acc_eqty:,.2f}"
    )
    print(status_msg)

    current_time_utc = datetime.now(timezone.utc)

    if is_rth and (current_time_utc - last_heartbeat_time).total_seconds() >= tele_interval:
        alert_msg = (
            f"| Trading Status Market: {rth_status} |\n"
            f"Z-Score: {current_z:.4f} | Hurst: {hurst:.4f} | ADF P: {p_value:.4f}\n"
            f"Prices: {ASSET_A} ${price_a:.2f}, {ASSET_B} ${price_b:.2f}\n"
            f"Current Position: {position} ({ASSET_A}: {pos_a:.0f} shares, {ASSET_B}: {pos_b:.0f} shares)\n"
            f"Equity: ${acc_eqty:,.2f}"
        )
        send_tele(alert_msg, alert_type="INFO")
        last_heartbeat_time = current_time_utc

# TODO: really gotta cut this down into more functions and less words. i literally have a stroke everytime i try to edit this
def liveLoop():
    global last_trade_time, entry_pos_type, entry_price_a, entry_price_b
    if not load_param():
        return

    ensure_persistence_dir()
    load_state()

    # TODO: consider removing this, trader is pretty robust in terms of its connection to telegram in the latest version
    send_tele(f"{ASSET_A}/{ASSET_B} BOT STARTED: Initial connectivity check and parameters loaded.", alert_type="INFO")

    bad_regime_counter = 0

    print(
        f"Running Pairs Trader. Check interval: {trade_interval}s. ADF Filter Threshold: P-Value <= {adf_max}. Hurst < {hurst_max}")

    start_pos, start_qty_a, start_qty_b = getCurrentPos(ASSET_A, ASSET_B)

    if start_pos != 0 and entry_pos_type is None:
        print("Open position save state missing, deducing its state")
        if start_qty_a > 0:
            entry_pos_type = "LONG_SPREAD_ENTRY"
        else:
            entry_pos_type = "SHORT_SPREAD_ENTRY"
        entry_price_a = 0.0
        entry_price_b = 0.0
        save_state()

    while True:
        if (datetime.now(timezone.utc) - last_trade_time).total_seconds() < cooldown_time:
            print("Cooldown active: waiting for 5 min after last trade.")
            time.sleep(trade_interval)
            continue

        loop_start_time = datetime.now()

        current_z = np.nan
        lastPriceA = np.nan
        lastPriceB = np.nan
        acc_eqty = np.nan
        beta = np.nan
        p_value = np.nan
        hurst = np.nan
        available_cash = np.nan
        position = 0
        pos_a = 0
        pos_b = 0
        current_pos = 0
        is_market_open_rth = False

        # accoutns status display
        try:
            account = trading_client.get_account()
            acc_eqty = float(account.equity)
            available_cash = float(account.cash)
            positions = trading_client.get_all_positions()
            current_pos = len(positions)

            position, pos_a, pos_b = getCurrentPos(ASSET_A, ASSET_B)

            print(f"\n CURRENT ACCOUNT STATUS: ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"Current Equity: ${acc_eqty:,.2f}")
            print(f"Available Cash: ${available_cash:,.2f}")
            print(f"Active Positions: {current_pos}")
            print(f"Pairs Position Status: {position} ({ASSET_A}: {pos_a:.0f}, {ASSET_B}: {pos_b:.0f})")

        except Exception as e:
            print(f"!!! Warning: Could not fetch account statistics. Error: {e}")

        try:
            clock = trading_client.get_clock()
            is_market_open_rth = clock.is_open
            market_time_str = clock.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
            next_open_str = clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')

        except Exception as e:
            print(f"!!! Error getting market clock: {e}. Assuming market is closed and waiting.")
            is_market_open_rth = False
            market_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            next_open_str = "N/A"

        if not is_market_open_rth:
            print(f"\nMarkt Closed: {market_time_str}. Next open: {next_open_str}. Current position: {position}")

            current_time_utc = datetime.now(timezone.utc)
            global last_heartbeat_time
            if (current_time_utc - last_heartbeat_time).total_seconds() >= sleepSeconds:
                alert_msg = (
                    f"Market Closed. No trading will occur.\n"
                    f"Next open: {next_open_str}.\n"
                    f"Current Equity: ${acc_eqty:,.2f}\n"
                    f"Trader: {ASSET_A}/{ASSET_B}\n"
                )
                send_tele(alert_msg, alert_type="MARKET")
                last_heartbeat_time = current_time_utc

            time.sleep(sleepSeconds)
            continue

        try:
            latest_data = filters_data(ASSET_A, ASSET_B, LOOKBACK_WINDOW)
            if latest_data is None or latest_data.empty:
                print("No RTH data available. Waiting...")
                time.sleep(trade_interval)
                continue

            beta, z_score, spread_volatility, latest_spread, p_value, hurst = calculateSignal(latest_data)

            if np.isnan(z_score) or np.isnan(beta):
                print("Signal calculation failed (NaN). Waiting...")
                time.sleep(trade_interval)
                continue

            current_z = z_score

            lastPriceA, lastPriceB = get_price(ASSET_A, ASSET_B)

            if np.isnan(lastPriceA) or np.isnan(lastPriceB):
                print("Could not fetch real-time prices. Waiting...")
                time.sleep(trade_interval)
                continue

            log_status(current_z, beta, p_value, hurst, lastPriceA, lastPriceB, position, pos_a,
                       pos_b, acc_eqty, loop_start_time, is_market_open_rth)

            if np.isnan(p_value) or p_value > adf_max:

                bad_regime_counter += 1
                print(f"SUSPICIOUS ADF DETECTED: P-Value {p_value:.4f}. Strike {bad_regime_counter}/3")

                if bad_regime_counter >= 3:
                    if position != 0:
                        print(
                            f"REGIME SHIFT CONFIRMED: P-Value {p_value:.4f} > {adf_max} for 3 checks. Liquidating.")
                        liquidate(reason=f"Regime shift detected: P-Value {p_value:.4f} > Threshold {adf_max}")

                    print(f"REGIME FILTER ACTIVE: P-Value {p_value:.4f}. Trading temporarily suspended.")
                else:
                    print(f"Waiting for confirmation of regime shift...")

            else:
                bad_regime_counter = 0

                if position != 0:
                    is_mean_reversion_exit = abs(current_z) <= Z_EXIT
                    is_stop_loss_exit = abs(current_z) >= Z_STOP_LOSS

                    if is_mean_reversion_exit or is_stop_loss_exit:
                        if is_mean_reversion_exit:
                            print(f"EXIT SIGNAL: Z-Score {current_z:.2f} is within |ZEXIT|={Z_EXIT}. Closing position.")
                            liquidate(reason=f"Mean-reversion exit: Z-Score {current_z:.2f} within |ZEXIT|={Z_EXIT}")
                            last_trade_time = datetime.now(timezone.utc)
                        elif is_stop_loss_exit:
                            print(
                                f"STOP LOSS: Z-Score {current_z:.2f} is beyond |ZSTOPLOSS|={Z_STOP_LOSS}. Liquidating NOW.")
                            liquidate(
                                reason=f"Stop-loss triggered: Z-Score {current_z:.2f} beyond |ZSTOPLOSS|={Z_STOP_LOSS}")
                            last_trade_time = datetime.now(timezone.utc)

                elif position == 0:
                    if abs(current_z) >= Z_ENTRY and Z_STOP_LOSS >= abs(current_z) and hurst < hurst_max:

                        shares_a, shares_b, qty_a, qty_b, V_A, V_B = determine_sizing(
                            lastPriceA, lastPriceB, beta, spread_volatility)

                        if qty_a == 0 or qty_b == 0:
                            print("Sizing resulted in zero quantity for one or both legs. Skipping entry.")
                            continue

                        entry_type = None
                        if current_z < -Z_ENTRY:
                            entry_type = "LONG_SPREAD_ENTRY"
                            print(
                                f"ENTRY LONG SPREAD: Z-Score {current_z:.2f} < -{Z_ENTRY}. Submitting orders atomically...")
                        else:
                            entry_type = "SHORT_SPREAD_ENTRY"
                            print(
                                f"ENTRY SHORT SPREAD: Z-Score {current_z:.2f} > +{Z_ENTRY}. Submitting orders atomically...")

                        entry_success = enter_pair_atomic(
                            qty_a, qty_b, current_z, beta, hurst, spread_volatility,
                            lastPriceA, lastPriceB, entry_type, order_type='LIMIT'
                        )

                        if entry_success:
                            pass
                        else:
                            pass

        except Exception as e:
            error_msg = f"!!! An unexpected error occurred in the live loop: {e}. Waiting 5 minutes."
            print(f"{error_msg}")

            alert_msg = f"{ASSET_A}/{ASSET_B} TRADER FACED UNEXPECTED ERROR: {e}\nLast Z-Score: {current_z:.4f} | Equity: ${acc_eqty:,.2f}"
            send_tele(alert_msg, alert_type="ERROR")

            time.sleep(300)

        print(f"Sleeping {trade_interval} seconds...")
        time.sleep(trade_interval)


# for me to debug if the code will send fractional shorts
def debug_submit_orders():
    test_price_a = 50.0
    test_price_b = 30.0
    test_beta = 1.2
    test_spread_vol = 0.08

    raw_a, raw_b, qty_a, qty_b, V_A, V_B = determine_sizing(
        test_price_a, test_price_b, test_beta, test_spread_vol
    )

    print("=== DEBUG SUBMIT ORDERS ===")
    print(f"Raw shares A: {raw_a} | Rounded qty A: {qty_a}")
    print(f"Raw shares B: {raw_b} | Rounded qty B: {qty_b}")
    print(f"Position Value A: {V_A:.2f} | Position Value B: {V_B:.2f}")

    if qty_a > 0 and qty_b > 0:
        print(f"Submitting BUY {qty_a} shares of {ASSET_A} and SELL {qty_b} shares of {ASSET_B}")
        success_a, filled_a, id_a, avg_a = submit_order(ASSET_A, qty_a, OrderSide.BUY)
        success_b, filled_b, id_b, avg_b = submit_order(ASSET_B, qty_b, OrderSide.SELL)

        print(f"Order Results -> {ASSET_A}: success={success_a}, filled={filled_a}, ID={id_a}, AvgP={avg_a:.2f}")
        print(f"Order Results -> {ASSET_B}: success={success_b}, filled={filled_b}, ID={id_b}, AvgP={avg_b:.2f}")
    else:
        print("Quantity too small, skipping order to avoid fractional shares.")


def debug_model():
    print("\n===== DEBUG MODEL / SPREAD / Z-SCORE =====")

    print("Loading parameters...")
    if not load_param():
        print("Could not load parameters. Exiting debug.")
        return

    global LOOKBACK_WINDOW
    if LOOKBACK_WINDOW is None:
        print("⚠ LOOKBACK_WINDOW was None. Using fallback = 5000.")
        LOOKBACK_WINDOW = 5000

    print("\nFetching lookback data...")
    # This now uses the I/O optimized path!
    data = filters_data(ASSET_A, ASSET_B, LOOKBACK_WINDOW)
    if data is None or len(data) == 0:
        print("No data loaded. Exiting debug.")
        return

    print(f"Loaded {len(data)} rows of RTH minute data.")

    print("\n>>> Testing spread direction detection...")
    # NOTE: choose_best_spread and MODEL_CACHE are missing from the provided code block, skipping this section
    print("Skipping spread detection test: choose_best_spread/MODEL_CACHE functions are not defined.")

    print("\n>>> Testing calculateSignal() ...")
    beta2, z2, std2, latest_spread2, pv2, hurst2 = calculateSignal(data)
    print(f"    Z-score = {z2:.6f}")
    print(f"    spread std = {std2:.6f}")
    print(f"    beta used = {beta2}")
    print(f"    p-value = {pv2}")
    print(f"    Hurst = {hurst2:.4f}")

    print("\n>>> Z-score movement over last 150 bars:")
    tail = data.iloc[-150:]
    zs = []
    for i in range(50, 150):
        temp = tail.iloc[:i]
        _, ztemp, _, _, _, _ = calculateSignal(temp)
        zs.append(ztemp)

    for i, z in enumerate(zs[-20:], 1):
        print(f"   step {i:02d}: Z={z:.4f}")

    print("\n===== DEBUG DONE =====")


if __name__ == "__main__":
    liveLoop()
    # debug_model()
    # debug_submit_orders() #comment out when not debugging