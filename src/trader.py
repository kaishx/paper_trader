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

# be careful with how its named in documentation, CHECK THE API DOCUMENTATION AGAIN
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

adf_env = os.getenv("adf_max")
if adf_env:
    adf_max = float(adf_env)
else:
    adf_max = 0.20

hurst_env = os.getenv("hurstMax")
if hurst_env:
    hurst_max = float(hurst_env)
else:
    hurst_max = 0.80

if not all([API_KEY_ID, API_SECRET_KEY, bot_token, chat_id]):
    print("missing api keys. check the environment variables, make sure in same directory")
    sys.exit(1)

if len(sys.argv) != 3:
    ASSET_A = "JPM"
    ASSET_B = "SAP"
    print(f"Inputs not found, use case: python #trader.py <ASSET_A> <ASSET_B>. Defaulting to {ASSET_A}/{ASSET_B}")

if len(sys.argv) == 3:
    ASSET_A = sys.argv[1]
    ASSET_B = sys.argv[2]

print(f"Trader started for pair: {ASSET_A}/{ASSET_B}")

trade_interval = 90
sleepSeconds = 180
tele_interval = 180
assigned_cptl = 100000  # probably need to check literature for recommended allocation for capital in relation to total capital
ABS_DOLLAR_STOP = 2500.0

Z_ENTRY = 2.3
Z_EXIT = 0.1
Z_STOP_LOSS = 4.5
LOAD_LOOKBACK = 4000
CALC_WINDOW = 390

last_heartbeat_time = datetime.min.replace(tzinfo=timezone.utc)
last_trade_time = datetime.min.replace(tzinfo=timezone.utc)
cooldown_time = 300  # i think this is deprecated

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
    global Z_ENTRY, Z_EXIT, Z_STOP_LOSS, LOAD_LOOKBACK, OPTIMIZER_ADF_P, OPTIMIZER_TRADABLE

    OPTIMIZER_ADF_P = None
    OPTIMIZER_TRADABLE = True

    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"!!! Error loading/parsing parameters file: {e} !!!")
        return False

    LOAD_LOOKBACK = params.get('metadata', {}).get('rolling_window_bars', LOAD_LOOKBACK)
    key = f"{ASSET_A}/{ASSET_B}"

    if key in params:
        Z_ENTRY = params[key]["z_entry"]
        Z_EXIT = params[key]["z_exit"]
        Z_STOP_LOSS = params[key]["z_sl"]
        OPTIMIZER_ADF_P = params[key].get("adf_p_value", None)
        if OPTIMIZER_ADF_P is not None:
            OPTIMIZER_TRADABLE = (OPTIMIZER_ADF_P <= adf_max)
        else:
            OPTIMIZER_TRADABLE = True

        print(f"! loaded optimized parameters: Z={Z_ENTRY}, Exit={Z_EXIT}, SL={Z_STOP_LOSS} | opt_adf={OPTIMIZER_ADF_P}")
        return True
    else:
        print(f"params missing for {key} in {file_path}")
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

                # make sure data is not too stale. data can get sussy baka on IEX)
                if (now - q.timestamp).total_seconds() > 120:
                    print(f"Quote too stale for {sym} ({(now - q.timestamp).total_seconds()}s)")
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

def get_live_price():
    # just a wrapper
    return get_price(ASSET_A, ASSET_B)

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
    else:
        print("no cache found, performing cold fetch")
        start_date = end_date - timedelta(days=120)

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
                datacache = pd.concat([datacache, new_data], axis=0)
                datacache = datacache[~datacache.index.duplicated(keep='last')]

            save_cache()

        if datacache is None or datacache.empty:
            print("!!! failed to retrieve sufficient historical data. !!!")
            return None

        data = datacache.copy()

        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')

        data_est = data.tz_convert('US/Eastern')

        rth_filter = ((data_est.index.hour > 9) | ((data_est.index.hour == 9) & (data_est.index.minute >= 30))) & (
                data_est.index.hour < 16)
        data = data[rth_filter]

        if len(data) > lookback:
            data = data.iloc[-lookback:]
            datacache = data.copy()

        if data.empty or len(data) < lookback:
            print(f"!!! Warning! only {len(data)} bars. Desired Warmup: {lookback}")

        data['Log_A'] = np.log(data[f'Close_{asset_a}'])
        data['Log_B'] = np.log(data[f'Close_{asset_b}'])

        return data

    except Exception as e:
        print(f"dataFilter() error: {e}")
        return None



@njit
def calc_kalman(y, x, delta=1e-4, ve=1e-3):
    # nottt exactly 100% on top of the math, just tried to follow this https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/ and other similar ones
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
    data['Log_A'] = np.log(data[f'Close_{ASSET_A}'])
    data['Log_B'] = np.log(data[f'Close_{ASSET_B}'])

    Y = data['Log_A'].values
    X = data['Log_B'].values

    data['Beta'] = calc_kalman(Y, X)

    data['Alpha'] = data['Log_A'].rolling(CALC_WINDOW).mean() - data['Beta'] * data['Log_B'].rolling(CALC_WINDOW).mean()
    data['Spread'] = data['Log_A'] - (data['Alpha'] + data['Beta'] * data['Log_B'])

    spread_mean = data['Spread'].shift(1).rolling(CALC_WINDOW).mean()
    spread_std = data['Spread'].shift(1).rolling(CALC_WINDOW).std()

    data['Z_Score'] = (data['Spread'] - spread_mean) / (spread_std + 1e-9)
    data['Hurst'] = calc_hurst(data['Spread'].values)

    try:
        import statsmodels.api as sm
        Y = data['Log_A'].astype(float)
        X = data['Log_B'].astype(float)
        Xc = sm.add_constant(X)
        model = sm.OLS(Y, Xc).fit()
        resid = model.resid
        adf_p = float(adfuller(resid)[1])
    except Exception as e:
        print(f"warning: ADF calc failed: {e}")
        adf_p = 1.0

    last = data.iloc[-1]
    return last['Beta'], last['Z_Score'], spread_std.iloc[-1], adf_p, last['Hurst']


def determine_sizing(asset_a_price, asset_b_price, beta, spread_volatility):
    global assigned_cptl

    if np.isnan(beta) or spread_volatility <= 0 or asset_a_price <= 0 or asset_b_price <= 0:
        return 0, 0, 0, 0

    vol_benchmark = 0.15
    vol_scale = min(1.0, vol_benchmark / spread_volatility) if spread_volatility > 0 else 1.0
    V_Total = assigned_cptl * vol_scale

    b = abs(beta)
    if b == 0: b = 1.0  # PREVENT div by 0 here

    V_A = V_Total * (1.0 / (1.0 + b))
    V_B = V_Total - V_A

    qty_a = int(math.floor(V_A / asset_a_price))
    qty_b = int(math.floor(V_B / asset_b_price))

    return qty_a, qty_b, V_A, V_B


def getCurrentPos(asset_a, asset_b):
    try:
        positions = trading_client.get_all_positions()
        pos_a = 0.0
        pos_b = 0.0
        entry_a = 0.0
        entry_b = 0.0

        for p in positions:
            if p.symbol == asset_a:
                pos_a = float(p.qty)
                entry_a = float(p.avg_entry_price) if p.avg_entry_price else 0.0
            elif p.symbol == asset_b:
                pos_b = float(p.qty)
                entry_b = float(p.avg_entry_price) if p.avg_entry_price else 0.0

        return pos_a, pos_b, entry_a, entry_b

    except Exception as e:
        print(f"!!! getCurrentPos failed: {e} !!!")
        return 0.0, 0.0, 0.0, 0.0


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

def enter_pair(qty_a, qty_b, current_z, beta, hurst, spread_volatility, lastPriceA, lastPriceB, entry_type):
    global last_trade_time, entry_pos_type, entry_price_a, entry_price_b

    side_a = OrderSide.BUY if entry_type == "in_long" else OrderSide.SELL
    side_b = OrderSide.SELL if entry_type == "in_long" else OrderSide.BUY

    print(f"Attempting entry: Target A={qty_a}, B={qty_b}")

    success_a, filled_a, id_a, avg_a = submit_order(ASSET_A, qty_a, side_a, limit_price=lastPriceA, order_type='LIMIT')

    if filled_a == 0:
        print("!!! buying of leg A failed completely (0 filled). Aborting. !!!")
        return False

    # calculate B if A somehow only partially fills. but shouldnt if what i read on alpaca api docs is correct
    fill_ratio = filled_a / qty_a
    qty_b_adjusted = int(math.floor(qty_b * fill_ratio))

    if qty_b_adjusted == 0 and filled_a > 0:
        qty_b_adjusted = 1

    print(f"Leg A filled ({filled_a}). Adjusting Leg B to {qty_b_adjusted} (Ratio: {fill_ratio:.2f})...")

    success_b, filled_b, _, avg_b = submit_order(ASSET_B, qty_b_adjusted, side_b, order_type='MARKET')

    if not success_b or filled_b == 0:
        print("!!! Leg B Market Order Failed. Rolling back Leg A. !!!")
        send_tele(f"order fail: {ASSET_A} filled {filled_a}, {ASSET_B} failed. Rolling back.", "ERROR")
        rb_side = OrderSide.SELL if side_a == OrderSide.BUY else OrderSide.BUY
        submit_order(ASSET_A, int(filled_a), rb_side, order_type='MARKET')
        return False

    print(f"Entry Done for {ASSET_A}/{ASSET_B}")
    entry_price_a = avg_a
    entry_price_b = avg_b
    entry_pos_type = entry_type
    save_state()

    msg = f"Entered {entry_type}\n{ASSET_A}: {filled_a} @ {avg_a:.2f}\n{ASSET_B}: {filled_b} @ {avg_b:.2f}\nZ: {current_z:.2f}"
    send_tele(msg, "ENTRY")
    last_trade_time = datetime.now(timezone.utc)
    return True


def liquidate(reason="No reason provided"):
    print(f"[LIQUIDATE] {reason}")

    qty_a, qty_b, entry_a, entry_b = getCurrentPos(ASSET_A, ASSET_B)

    avg_exit_a = 0.0
    avg_exit_b = 0.0

    if qty_a != 0:
        side = OrderSide.SELL if qty_a > 0 else OrderSide.BUY
        _, _, _, avg_exit_a = submit_order(ASSET_A, int(abs(qty_a)), side, order_type='MARKET')

    if qty_b != 0:
        side = OrderSide.SELL if qty_b > 0 else OrderSide.BUY
        _, _, _, avg_exit_b = submit_order(ASSET_B, int(abs(qty_b)), side, order_type='MARKET')

    pnl = 0.0

    if entry_a > 0 and avg_exit_a > 0:
        pnl += (avg_exit_a - entry_a) * qty_a

    if entry_b > 0 and avg_exit_b > 0:
        pnl += (avg_exit_b - entry_b) * qty_b

    send_tele(f"Liquidated {ASSET_A}/{ASSET_B} due to {reason}\nRealized PnL: ${pnl:.2f}", "EXIT")

    clear_state()


def statusupdate(current_z, beta, p_value, hurst, price_a, price_b, position, pos_a, pos_b, acc_eqty, loop_start_time,
                 is_rth):
    global last_heartbeat_time
    print(
        f"{loop_start_time.strftime('%H:%M')} | Z:{current_z:.3f} | ADF:{p_value:.3f} | H:{hurst:.3f} | {ASSET_A}:${price_a:.2f} {ASSET_B}:${price_b:.2f} | Eqty: ${acc_eqty:,.2f}")

    if is_rth and (datetime.now(timezone.utc) - last_heartbeat_time).total_seconds() >= tele_interval:
        status_text = "OPEN (RTH)" if is_rth else "CLOSED"

        msg = (
            f"| Trading Status Market: {status_text} |\n"
            f"Z-Score: {current_z:.4f} | Hurst: {hurst:.4f} | ADF P: {p_value:.4f}\n"
            f"Prices: {ASSET_A} ${price_a:.2f}, {ASSET_B} ${price_b:.2f}\n"
            f"Current Position: {position} ({ASSET_A}: {pos_a} shares, {ASSET_B}: {pos_b} shares)\n"
            f"Equity: ${acc_eqty:,.2f}"
        )

        send_tele(msg, "INFO")

        last_heartbeat_time = datetime.now(timezone.utc)


def isMarketOpen():
    clock = trading_client.get_clock()

    if clock.is_open:
        return True

    now = datetime.now(timezone.utc)
    next_open = clock.next_open
    sleeptime = (next_open - now).total_seconds()

    if sleeptime > 0:
        msg = f"{ASSET_A}/{ASSET_B}: Market is Closed. Next Open: {next_open.isoformat()}. Sleeping for {sleeptime / 3600:.2f} hours..."
        print(msg)
        send_tele(msg, "STATUS")
        time.sleep(sleeptime + 60)
        return False

    return True

def live_loop():
    ensure_persistence_dir()
    load_state()

    print(f"Trader starting... awaiting params for {ASSET_A}/{ASSET_B}")

    while True:
        try:
            if not isMarketOpen():
                continue

            if not load_param():
                print("Parameters not found. Sleeping...")
                time.sleep(300)
                continue

            if not OPTIMIZER_TRADABLE:
                print(f"Pair {ASSET_A}/{ASSET_B} non-stationary (p={OPTIMIZER_ADF_P}). Sleeping 1 hour.")
                time.sleep(3600)
                continue

            df = dataFilter(ASSET_A, ASSET_B, LOAD_LOOKBACK)
            if df is None or len(df) < 500:
                print("Waiting for data buffer...")
                time.sleep(60)
                continue

            beta, z_score, vol, adf_p, hurst = calc_signal(df.copy())

            pa, pb = get_live_price()

            if pa <= 0 or pb <= 0 or np.isnan(pa) or np.isnan(pb):
                print("Invalid price data (0 or NaN). Retrying...")
                time.sleep(10)
                continue

            qa, qb, entry_a, entry_b = getCurrentPos(ASSET_A, ASSET_B)

            try:
                acc_eqty = float(trading_client.get_account().equity)
            except:
                acc_eqty = 0.0

            if abs(qa) > 0 and entry_a == 0:
                if entry_price_a is not None and entry_price_a > 0:
                    entry_a = entry_price_a
                    print(f"alpaca returned 0 entry for {ASSET_A}, using cached state: {entry_a}")

            if abs(qb) > 0 and entry_b == 0:
                if entry_price_b is not None and entry_price_b > 0:
                    entry_b = entry_price_b
                    print(f"alpaca returned 0 entry for {ASSET_B}, using cached state: {entry_b}")

            unrealized_pnl = 0.0
            valid_pnl = True

            if abs(qa) > 0:
                if entry_a > 0:
                    pnl_a = (pa - entry_a) * qa
                else:
                    print(f"!!! missing cost basis for {ASSET_A} !!!")
                    valid_pnl = False
                    pnl_a = 0

            if abs(qb) > 0:
                if entry_b > 0:
                    pnl_b = (pb - entry_b) * qb
                else:
                    print(f"!!! missing cost basis for {ASSET_B} !!!")
                    valid_pnl = False
                    pnl_b = 0

            if valid_pnl and (abs(qa) > 0 or abs(qb) > 0):
                unrealized_pnl = pnl_a + pnl_b

                # ONLY check stops if PnL is valid
                if unrealized_pnl < -ABS_DOLLAR_STOP:
                    print(f"!!! DOLLAR STOP TRIGGERED (PnL: {unrealized_pnl:.2f}) !!!")
                    liquidate("DOLLAR STOP")

            status_msg = f"Z:{z_score:.2f} | PnL:${unrealized_pnl:.1f} | ADF(Monitor):{adf_p:.2f}"
            print(status_msg)

            pos_str = "LONG" if qa > 0 else "SHORT" if qa < 0 else "FLAT"

            statusupdate(z_score, beta, adf_p, hurst, pa, pb, pos_str, qa, qb, acc_eqty, datetime.now(timezone.utc),
                         True)

            if abs(qa) > 0 or abs(qb) > 0:
                if unrealized_pnl < -ABS_DOLLAR_STOP:
                    print(f"!!! DOLLAR STOP TRIGGERED (PnL: {unrealized_pnl:.2f}) !!!")
                    liquidate("DOLLAR STOP")

                elif abs(z_score) > Z_STOP_LOSS:
                    print("!!! Z STOP TRIGGERED !!!")
                    liquidate("Z-SCORE STOP")

                elif abs(z_score) < Z_EXIT:
                    print("!!! TARGET EXIT !!!")
                    liquidate("TARGET EXIT")

            else:
                if hurst < hurst_max and abs(z_score) > Z_ENTRY:
                    capital = assigned_cptl
                    vol_scale = min(1.0, 0.15 / vol) if vol > 0 else 1.0
                    alloc = capital * vol_scale

                    va = alloc / (1.0 + abs(beta))
                    vb = alloc - va
                    tqa = int(va / pa)
                    tqb = int(vb / pb)

                    e_type = "in_long" if z_score < -Z_ENTRY else "in_short"

                    enter_pair(tqa, tqb, z_score, beta, hurst, vol, pa, pb, e_type)

            time.sleep(60)

        except Exception as e:
            print(f"!!! error in main loop: {e} !!!")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    live_loop()
