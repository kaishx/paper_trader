import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests  # for tele alerts
import sys
import builtins  # for flushing prints to the main controller
import math
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
import threading
import os
import pickle
import traceback
from numba import njit

# alpaca imports, gotta be careful with how its named in documentation
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

paper_setting = True

# relaxed constraints for IEX data quality
adf_env = os.getenv("adf_max")
if adf_env:
    adf_max = float(adf_env)
else:
    adf_max = 0.40  # relaxed from 0.2

hurst_env = os.getenv("hurstMax")
if hurst_env:
    hurst_max = float(hurst_env)
else:
    hurst_max = 0.80  # relaxed from 0.75

if not all([API_KEY_ID, API_SECRET_KEY, bot_token, chat_id]):
    print("missing api keys. check the environment variables, make sure in same directory")
    sys.exit(1)

if len(sys.argv) != 3:
    ASSET_A = "JPM"
    ASSET_B = "SAP"
    print(f"Inputs not found, use case: python #trader.py <ASSET_A> <ASSET_B>. Defaulting to {ASSET_A}/{ASSET_B}")
    sys.exit(1)

ASSET_A = sys.argv[1]
ASSET_B = sys.argv[2]

print(f"Trader started for pair: {ASSET_A}/{ASSET_B}")

trade_interval = 90
sleepSeconds = 180
tele_interval = 180
assigned_cptl = 100000  # probably need to check literature for recommended allocation for capital in relation to total capital

Z_ENTRY = 2.3
Z_EXIT = 0.1
Z_STOP_LOSS = 4.5
LOOKBACK_WINDOW = 390

last_heartbeat_time = datetime.min.replace(tzinfo=timezone.utc)
last_trade_time = datetime.min.replace(tzinfo=timezone.utc)
cooldown_time = 300 # this is deprecated

data_client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
trading_client = TradingClient(API_KEY_ID, API_SECRET_KEY, paper=paper_setting)

entry_price_a = None
entry_price_b = None
entry_pos_type = None

PERSISTENCE_DIR = 'trader_data'

bar_cache_path = os.path.join(PERSISTENCE_DIR, f'cache_{ASSET_A}_{ASSET_B}.pkl')
state_path = os.path.join(PERSISTENCE_DIR, f'state_{ASSET_A}_{ASSET_B}.json')
datacache = None


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
        with open(state_path, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"!!! failed to save state to {state_path}: {e} !!!")


def load_state():
    # added so i could save entry data across runs for pnl calculation
    # but the pnl calculation is still broken regardless of this. arghhhhh
    global entry_price_a, entry_price_b, entry_pos_type
    if not os.path.exists(state_path):
        return

    try:
        with open(state_path, 'r') as f:
            state = json.load(f)

        if state.get("entry_price_a") is not None:
            entry_price_a = float(state["entry_price_a"])
        if state.get("entry_price_b") is not None:
            entry_price_b = float(state["entry_price_b"])
        entry_pos_type = state["entry_pos_type"]

        if entry_pos_type is not None:
            print(f"loaded persistent state: {entry_pos_type} entered at A={entry_price_a}, B={entry_price_b}")

    except Exception as e:
        print(f"!!! failed to load state from {state_path}: {e} !!!")


def clear_state():
    global entry_price_a, entry_price_b, entry_pos_type
    entry_price_a = None
    entry_price_b = None
    entry_pos_type = None
    if os.path.exists(state_path):
        os.remove(state_path)


def save_cache():
    global datacache
    if datacache is None or datacache.empty:
        return
    ensure_persistence_dir()
    with open(bar_cache_path, 'wb') as f:
        pickle.dump(datacache, f)


def load_cache():
    global datacache
    if not os.path.exists(bar_cache_path):
        datacache = None
        return

    with open(bar_cache_path, 'rb') as f:
        datacache = pickle.load(f)
        if not isinstance(datacache, pd.DataFrame):
            datacache = None


def send_tele(message, alert_type="INFO"):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': f"[{alert_type}] {message}"}
    try:
        requests.post(url, data=payload, timeout=10)
    except:
        pass


def load_param(file_path='optimized_params.json'):
    global Z_ENTRY, Z_EXIT, Z_STOP_LOSS, LOOKBACK_WINDOW

    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"!!! Error loading/parsing parameters file: {e} !!!")
        return False

    LOOKBACK_WINDOW = params.get('metadata', {}).get('rolling_window_bars', 390)
    key = f"{ASSET_A}/{ASSET_B}"

    if key in params:
        Z_ENTRY = params[key]["z_entry"]
        Z_EXIT = params[key]["z_exit"]
        Z_STOP_LOSS = params[key]["z_sl"]
        print(f"! loaded optimized parameters: Z={Z_ENTRY}, Exit={Z_EXIT}, SL={Z_STOP_LOSS}")
        return True
    else:
        return False


def get_price(asset_a, asset_b):
    symbols = [asset_a, asset_b]
    price_map = {asset_a: np.nan, asset_b: np.nan}

    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        res = data_client.get_stock_latest_quote(req)

        now = datetime.now(timezone.utc)

        for sym in symbols:
            if sym in res:
                q = res[sym]

                # check for stale data (older than 15s is sussy on IEX)
                if (now - q.timestamp).total_seconds() > 15:
                    print(f"Quote stale for {sym} ({(now - q.timestamp).total_seconds()}s)")
                    return np.nan, np.nan

                if q.ask_price > 0 and q.bid_price > 0:
                    price = (q.ask_price + q.bid_price) / 2
                elif q.ask_price > 0:
                    price = q.ask_price
                elif q.bid_price > 0:
                    price = q.bid_price
                else:
                    price = 0.0

                if price > 0:
                    price_map[sym] = float(price)
            else:
                print(f"Note: no quote data returned for {sym}")

    except Exception as e:
        print(f"!!! error fetching live prices for {asset_a}/{asset_b}: {e} !!!")

    return float(price_map[asset_a]), float(price_map[asset_b])


def get_hist(symbol, start_date, end_date, timeframe):
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX,
            limit=10000
        )
        bars = data_client.get_stock_bars(req)

        if not bars.data or symbol not in bars.data:
            return None

        return bars.df.loc[symbol]['close']

    except Exception as e:
        return None


def dataFilter(asset_a, asset_b, lookback, timeframe=TimeFrame(15, TimeFrameUnit.Minute)):
    global datacache

    if datacache is None:
        load_cache()
        
    end_date = datetime.now(timezone.utc)

    if datacache is not None and not datacache.empty:
        if datacache.index.tz is None:
            datacache.index = datacache.index.tz_localize('UTC')
        else:
            datacache.index = datacache.index.tz_convert('UTC')

        last_ts = datacache.index[-1]
        start_date = last_ts + timedelta(minutes=1)
        print(f"cache found, getting data from {start_date}")

    try:
        series_a = get_hist(asset_a, start_date, end_date, timeframe)
        series_b = get_hist(asset_b, start_date, end_date, timeframe)

        new_data = None
        if series_a is not None and series_b is not None:
            new_data = pd.concat([series_a, series_b], axis=1).dropna()
            new_data.columns = [f'Close_{asset_a}', f'Close_{asset_b}']

            if new_data.index.tz is None:
                new_data.index = new_data.index.tz_localize('UTC')
            else:
                new_data.index = new_data.index.tz_convert('UTC')

        if new_data is not None and not new_data.empty:
            if datacache is None:
                datacache = new_data
            else:
                datacache = pd.concat([datacache, new_data], axis=0).drop_duplicates(keep='last')
                datacache.sort_index(inplace=True)

            save_cache()

        if datacache is None or datacache.empty:
            print("!!! failed to retrieve sufficient historical data. !!!")
            return None

        data = datacache.copy()
        data_est = data.tz_convert('US/Eastern')

        # RTH Filter
        rth_filter = ((data_est.index.hour > 9) | ((data_est.index.hour == 9) & (data_est.index.minute >= 30))) & (
                    data_est.index.hour < 16)
        data = data[rth_filter]

        MAX_BARS_TO_KEEP = 60 * 24 * 60
        if len(data) > MAX_BARS_TO_KEEP:
            data = data.iloc[-MAX_BARS_TO_KEEP:]
            datacache = data.copy()

        data = data.iloc[-(lookback + 50):].copy()

        if data.empty or len(data) < lookback:
            print(f"!!! Warning! only {len(data)} bars after filtering. Need {lookback}")
            return None

        data['Log_A'] = np.log(data[f'Close_{asset_a}'])
        data['Log_B'] = np.log(data[f'Close_{asset_b}'])

        return data

    except Exception as e:
        print(f"dataFilter() error: {e}")   
        return None


@njit
def calc_kalman(y, x, delta=1e-4, ve=1e-3):
    # not exactly 100% on top of the math, just tried to follow this https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/ and other similar ones
    n = len(y)
    beta = np.zeros(n)
    state_mean = 0.0
    state_cov = 0.1

    for t in range(n):
        state_cov += delta
        obs_val = y[t]
        obs_mat = x[t]
        pred_obs = obs_mat * state_mean
        residual = obs_val - pred_obs
        var = (obs_mat * state_cov * obs_mat) + ve
        gain = (state_cov * obs_mat) / var
        state_mean += gain * residual
        state_cov = (1 - gain * obs_mat) * state_cov
        beta[t] = state_mean

    return beta


@njit
def calc_hurst(series, window=100):
    n = len(series)
    hurst = np.full(n, 0.5)
    min_window = max(window, 30)

    for t in range(min_window, n):
        start = max(0, t - window)
        ts = series[start:t]
        if len(ts) < 10: continue

        m = np.mean(ts)
        cum_dev = np.cumsum(ts - m)
        r = np.max(cum_dev) - np.min(cum_dev)
        s = np.std(ts)

        if s == 0 or r == 0 or len(ts) < 2:
            hurst[t] = 0.5
        else:
            hurst[t] = np.log(r / s) / np.log(len(ts))

    return hurst


def calc_signal(data):
    z_window = 390

    Y = data['Log_A'].values
    X = data['Log_B'].values
    beta_series = calc_kalman(Y, X)

    latest_beta = beta_series[-1]
    data['Kalman_Beta'] = beta_series

    rolling_mean_Y = data['Log_A'].rolling(window=z_window).mean()
    rolling_mean_X = data['Log_B'].rolling(window=z_window).mean()

    data['Alpha'] = rolling_mean_Y - data['Kalman_Beta'] * rolling_mean_X
    data['Spread'] = data['Log_A'] - (data['Alpha'] + data['Kalman_Beta'] * data['Log_B'])

    spread_tail = data['Spread'].iloc[-z_window:].dropna()

    if len(spread_tail) < 30: return latest_beta, 0, 0, 0, 1.0, 0.5

    mean = spread_tail.mean()
    std = spread_tail.std()
    latest_spread = spread_tail.iloc[-1]
    hurst_val = calc_hurst(data['Spread'].values)[-1]

    z_score = (latest_spread - mean) / (std + 1e-9)

    try:
        adf_p = adfuller(spread_tail, autolag='AIC')[1]
    except:
        adf_p = 1.0

    return latest_beta, z_score, std, latest_spread, adf_p, hurst_val


def determine_sizing(asset_a_price, asset_b_price, beta, spread_volatility):
    global assigned_cptl

    if np.isnan(beta) or spread_volatility <= 0 or asset_a_price <= 0 or asset_b_price <= 0:
        return 0, 0, 0, 0

    vol_benchmark = 0.15
    vol_scale = min(1.0, vol_benchmark / spread_volatility) if spread_volatility > 0 else 1.0
    V_Total = assigned_cptl * vol_scale

    b = abs(beta)
    if b == 0: b = 1.0 # PREVENT div by 0 here

    V_A = V_Total * (b / (1.0 + b))
    V_B = V_Total - V_A

    qty_a = int(math.floor(V_A / asset_a_price))
    qty_b = int(math.floor(V_B / asset_b_price))

    return qty_a, qty_b, V_A, V_B


def submit_order(symbol, qty, side, limit_price=None, order_type='LIMIT'):
    try:
        if qty <= 0: return False, 0, None, 0

        req = None
        if order_type == 'MARKET':
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
            print(f"placed market order: {side} {qty} {symbol}")
        else:
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=side, limit_price=limit_price,
                                    time_in_force=TimeInForce.GTC)
            print(f"placed limit order: {side} {qty} {symbol} @ {limit_price:.2f}")

        order = trading_client.submit_order(req)

        for _ in range(30):
            time.sleep(1)
            o = trading_client.get_order_by_id(order.id)
            if o.status == 'filled':
                return True, float(o.filled_qty), o.id, float(o.filled_avg_price or 0)
            if o.status in ['canceled', 'rejected', 'expired']:
                return False, float(o.filled_qty), o.id, 0

        if order_type == 'LIMIT':
            print(f"!!! order {order.id} timed out. Canceling. !!!")
            trading_client.cancel_order(order.id)

        return False, 0, order.id, 0

    except Exception as e:
        print(f"!!! order failed: {e}!!!")
        return False, 0, None, 0


def enter_pair_atomic(qty_a, qty_b, current_z, beta, hurst, spread_volatility, lastPriceA, lastPriceB, entry_type):
    # changed this logic to be hybrid limit-market to enusre that it gets filled most of the time
    global last_trade_time, entry_pos_type, entry_price_a, entry_price_b

    side_a = OrderSide.BUY if entry_type == "in_long" else OrderSide.SELL
    side_b = OrderSide.SELL if entry_type == "in_long" else OrderSide.BUY

    print("Attempting entry")

    success_a, filled_a, _, avg_a = submit_order(ASSET_A, qty_a, side_a, limit_price=lastPriceA, order_type='LIMIT')

    if not success_a or filled_a == 0:
        print("!!! leg A failed/timed out. aborting. !!!")
        return False

    print(f"Leg A filled ({filled_a}). Firing leg B Market...")
    success_b, filled_b, _, avg_b = submit_order(ASSET_B, qty_b, side_b, order_type='MARKET')

    if not success_b or filled_b == 0:
        print("!!! Leg B Market Order Failed. Rolling back Leg A. !!!")
        send_tele(f"order fail: {ASSET_A} filled, {ASSET_B} failed. Rolling back.", "ERROR")
        rb_side = OrderSide.SELL if side_a == OrderSide.BUY else OrderSide.BUY
        submit_order(ASSET_A, int(filled_a), rb_side, order_type='MARKET')
        return False

    print("entry done")
    entry_price_a = avg_a
    entry_price_b = avg_b
    entry_pos_type = entry_type
    save_state()

    msg = f"entered {entry_type}\n{ASSET_A}: {filled_a} @ {avg_a:.2f}\n{ASSET_B}: {filled_b} @ {avg_b:.2f}\nZ: {current_z:.2f}"
    send_tele(msg, "ENTRY")
    last_trade_time = datetime.now(timezone.utc)
    return True


def liquidate(reason="No reason provided"):
    global entry_price_a, entry_price_b, entry_pos_type

    print(f"[LIQUIDATE] {reason}")

    _, qty_a, qty_b = getCurrentPos(ASSET_A, ASSET_B)

    if qty_a != 0:
        side = OrderSide.SELL if qty_a > 0 else OrderSide.BUY
        submit_order(ASSET_A, int(abs(qty_a)), side, order_type='MARKET')

    if qty_b != 0:
        side = OrderSide.SELL if qty_b > 0 else OrderSide.BUY
        submit_order(ASSET_B, int(abs(qty_b)), side, order_type='MARKET')

    send_tele(f"Liquidated {ASSET_A}/{ASSET_B} due to {reason}", "EXIT")
    clear_state()


def statusupdate(current_z, beta, p_value, hurst, price_a, price_b, position, pos_a, pos_b, acc_eqty, loop_start_time,
               is_rth):
    global last_heartbeat_time
    print(
        f"{loop_start_time.strftime('%H:%M')} | Z:{current_z:.2f} | ADF:{p_value:.2f} | H:{hurst:.2f} | {ASSET_A}:${price_a:.2f} {ASSET_B}:${price_b:.2f}")

    if is_rth and (datetime.now(timezone.utc) - last_heartbeat_time).total_seconds() >= tele_interval:
        msg = f"Z: {current_z:.2f} | ADF: {p_value:.2f}\nPos: {pos_a}/{pos_b}"
        send_tele(msg, "INFO")
        last_heartbeat_time = datetime.now(timezone.utc)


def liveLoop():
    global last_trade_time, entry_pos_type, entry_price_a, entry_price_b
    if not load_param(): return
    ensure_persistence_dir()
    load_state()

    print(f"started running. ADF<{adf_max}, Hurst<{hurst_max}")
    bad_regime_counter = 0

    while True:
        try:
            if (datetime.now(timezone.utc) - last_trade_time).total_seconds() < cooldown_time:
                time.sleep(60);
                continue

            try:
                clock = trading_client.get_clock()
                if not clock.is_open:
                    print("Market is closed");
                    time.sleep(sleepSeconds);
                    continue
            except:
                pass

            latest_data = dataFilter(ASSET_A, ASSET_B, LOOKBACK_WINDOW)
            if latest_data is None:
                time.sleep(trade_interval);
                continue

            beta, z_score, spread_vol, _, p_value, hurst = calc_signal(latest_data)

            pa, pb = get_price(ASSET_A, ASSET_B)
            if np.isnan(pa):
                time.sleep(10);
                continue

            position, pos_a, pos_b = getCurrentPos(ASSET_A, ASSET_B)
            if position == 999:
                print("!!! orphan detected. stopping trader for 10m. Please manually fix. !!!");
                send_tele("ORPHAN", "ERROR");
                time.sleep(600);
                continue

            acct = trading_client.get_account()
            statusupdate(z_score, beta, p_value, hurst, pa, pb, position, pos_a, pos_b, float(acct.equity),
                       datetime.now(), True)

            if p_value > adf_max:
                bad_regime_counter += 1
                if bad_regime_counter >= 5 and position != 0:
                    liquidate("Regime Shift")
            else:
                bad_regime_counter = 0
                if position != 0:
                    if abs(z_score) <= Z_EXIT or abs(z_score) >= Z_STOP_LOSS:
                        liquidate(f"Exit Z={z_score:.2f}")
                elif position == 0 and hurst < hurst_max:
                    if abs(z_score) > Z_ENTRY:
                        qa, qb, _, _ = determine_sizing(pa, pb, beta, spread_vol)
                        if qa > 0 and qb > 0:
                            etype = "in_long" if z_score < 0 else "in_short"
                            enter_pair_atomic(qa, qb, z_score, beta, hurst, spread_vol, pa, pb, etype)

            time.sleep(trade_interval)

        except Exception as e:
            print(f"!!! Crash: {e}!!! ")
            traceback.print_exc()
            send_tele(f"Crash: {e}", "ERROR")
            time.sleep(60)

if __name__ == "__main__":
    liveLoop()