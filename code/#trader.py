import time
import json
import pandas as pd
import numpy as np
import requests  # for tele alerts
import yfinance as yf # old and archaic, keeping it around.
from datetime import datetime, timedelta, timezone
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
import sys
import builtins  # for flushing prints to the main controller
import math
from numba import njit

# alpaca imports, be careful with how its named in documentation. ai always give wrong imports, have to read documentation for this
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)

# api
API_KEY_ID = "PKCHRDERPHH52D5RJN3UNEYKBU"  # HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB
API_SECRET_KEY = "EFWXvL7vkmnSzSLVUpfqDB9tBrgfNms7PWdjrwn7rQ3c"  # HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB

bot_token = "8575237777:AAHDxDYRu-m_bpb9vb2lfCpIDETXpBEdUtU"  #HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB
chat_id = "-1003289508299"  # HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB

paper_setting = True
adf_max = 0.2  # UP: more looser, DOWN: more tight. noticed from WFA that 0.05 is kinda tight after the kalman update
hurst_max = 0.8  # supposed to be 0.5 according to its defintiion, but that is too tight

if len(sys.argv) != 3:
    print("Usage: python trader.py <ASSET_A> <ASSET_B>")
    sys.exit(1)

ASSET_A = sys.argv[1]
ASSET_B = sys.argv[2]

# keep this here so i can paste it back everytime i remove above to debug through trader.py without goin thru the controller
# if len(sys.argv) != 3:
# print("Usage: python trader.py <ASSET_A> <ASSET_B>")
# sys.exit(1)

# ASSET_A = sys.argv[1]
# ASSET_B = sys.argv[2]

# FOR DEBUG
# ASSET_A = "GOOG"
# ASSET_B = "GOOGL"

print(f"Trader started for pair: {ASSET_A}/{ASSET_B}")

# settings
trade_interval = 60
sleepSeconds = 300
tele_interval = 180
assigned_cptl = 10000 # todo: check literature for recommended allocation for capital in relation to total capital

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

    full_message = f"{msgTypeMap.get(alert_type, '💬')} *[{alert_type}]*\n{message}"

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
        print(f"telegram failed to send msg due to HTTP error: ({http_err.response.status_code})")
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

        LOOKBACK_WINDOW = params.get('metadata', {}).get('rolling_window_bars', 5000)

        found = False
        for result in params.get('optimization_results', []):
            if result['asset_a'] == ASSET_A and result['asset_b'] == ASSET_B:
                Z_ENTRY = result['optimal_z_entry']
                Z_EXIT = result['optimal_z_exit']
                Z_STOP_LOSS = result['optimal_z_stop_loss']
                found = True
                print(f"✅ Loaded optimized parameters for {ASSET_A}/{ASSET_B}:")
                print(
                    f"   Z_ENTRY={Z_ENTRY}, Z_EXIT={Z_EXIT}, Z_STOP_LOSS={Z_STOP_LOSS}, LOOKBACK_WINDOW={LOOKBACK_WINDOW}")
                break

        if not found:
            print(f"no params found for {ASSET_A}/{ASSET_B} in '{file_path}'.")
            return False

        return True

    except FileNotFoundError:
        print(f" json params file '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"failed to load parameters from JSON: {e}")
        return False


def get_raw(symbol, start, end, timeframe):
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=start.isoformat(),
        end=end.isoformat(),
        feed=DataFeed.IEX,
        adjustment="all"
    )

    # alpaca first
    for attempt in range(3):
        try:
            bars = data_client.get_stock_bars(request_params).df
            if not bars.empty:
                close_prices = bars.loc[(symbol, slice(None)), 'close'].rename(f'Close_{symbol}')
                return close_prices.droplevel('symbol')
        except Exception as e:
            print(f"alpaca attempt {attempt + 1} failed for {symbol}: {e}")
            time.sleep(2 ** attempt)

    print(f"❌ Failed to fetch data for {symbol} after 3 attempts.")
    return None


def filters_data(asset_a, asset_b, lookback, timeframe=TimeFrame(15, TimeFrameUnit.Minute)):
    try:
        end_date = datetime.now()

        days_to_fetch = int((lookback / 26) * 2.0) + 10
        start_date = end_date - timedelta(days=days_to_fetch)

        data_a = get_raw(asset_a, start_date, end_date, timeframe)
        data_b = get_raw(asset_b, start_date, end_date, timeframe)

        if data_a is None or data_b is None:
            raise Exception("failed to retrieve latest data.")

        data = pd.concat([data_a, data_b], axis=1).dropna()

        # rth filter
        if data.index.tz is None:
            data = data.tz_localize('UTC')
        local_data = data.tz_convert('US/Eastern')
        rth_filter = (
                (local_data.index.hour >= 9) &
                ((local_data.index.hour != 9) | (local_data.index.minute >= 30)) &
                (local_data.index.hour < 16)
        )
        data = data[rth_filter]

        data = data.iloc[-(lookback + 50):]

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

    # adf check (legacy)
    try:
        adf_p = adfuller(data['Spread'].dropna().tail(100), autolag='AIC')[1]
    except:
        adf_p = 1.0

    return latest_beta, z_score, std, latest_spread, adf_p, hurst_val


def get_price(asset_a, asset_b):
    symbols = [asset_a, asset_b]
    price_map = {asset_a: np.nan, asset_b: np.nan}

    print("Fetching latest price via Alpaca REST API...")

    try:
        quotes = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbols))

        for sym in symbols:
            quote = quotes.get(sym)
            if quote is None:
                print(f"Alpaca: no quote object for {sym}")
                continue
            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)

            if bid and ask and bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2.0
                price_map[sym] = float(mid_price)
                print(f"Price {sym} (Alpaca Mid-Quote): ${mid_price:.2f}")
            else:
                print(f"Alpaca: quote for {sym} missing bid/ask (bid={bid}, ask={ask}). Will fallback per-symbol.")
    except Exception as e:
        print(f"Alpaca request failed with exception: {type(e).__name__}: {e}")

    return float(price_map[asset_a]), float(price_map[asset_b])


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

            # long A short B
            if qty_a > 0 and qty_b < 0:
                return 1, qty_a, qty_b
            # short A long B
            elif qty_a < 0 and qty_b > 0:
                return -1, qty_a, qty_b

        return 0, 0, 0

    except Exception as e:
        print(f"Error checking position: {e}")
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

        print("\n📈📊 **--- TRADE CLOSED: PNL REPORT ---** 📊📈")
        print(f"💰 Account Equity: ${equity:,.2f}")
        print(f"🔥 PnL for Today (Approximation): ${pnl_today:,.2f}")
        print("------------------------------------------")

        return equity, pnl_today

    except Exception as e:
        print(f"⚠️ Warning: Could not fetch PnL stats from Alpaca: {e}")
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


def submit_order(symbol, qty, side, limit_price=None, wait_interval=1, max_wait_seconds=20):
    try:
        if qty <= 0:
            return False, 0.0

        quote = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        ).get(symbol)

        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)

        if limit_price is None:
            if side == OrderSide.BUY:
                limit_price = ask  # aggressive but safer fill
            else:
                limit_price = bid

        if limit_price is None or limit_price <= 0:
            # fallback if quote fails, use last trade or something safe, or just abort
            print(f"⚠️ Limit price could not be determined for {symbol} (bid={bid}, ask={ask}). Aborting order.")
            return False, 0.0

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC
        )

        order = trading_client.submit_order(request)
        print(f"Placed LIMIT {side.value} {qty} {symbol} @ {limit_price:.2f}")

        total_waited = 0
        filled_qty = 0.0

        while total_waited < max_wait_seconds:
            order_status = trading_client.get_order_by_id(order.id)
            filled_qty = float(order_status.filled_qty)

            if 0 < filled_qty < qty:
                print(f"Partial Fill: {filled_qty}/{qty}")

            if order_status.status in ["filled", "canceled", "rejected"]:
                break

            time.sleep(wait_interval)
            total_waited += wait_interval

        print(f"Filled: {filled_qty}/{qty}")

        send_tele(
            f"Limit order {side.value} {symbol} executed: {filled_qty}/{qty} shares @ {limit_price:.2f}",
            alert_type="INFO"
        )

        return filled_qty > 0, filled_qty

    except Exception as e:
        print(f"LIMIT order failed: {e}")
        send_tele(f"LIMIT order failed for {symbol}: {e}", alert_type="ERROR")
        return False, 0.0


def liquidate(reason="No reason provided."):
    global entry_price_a, entry_price_b, entry_pos_type

    print(f"[LIQUIDATE] entry_price_a={entry_price_a}, entry_price_b={entry_price_b}, entry_pos_type={entry_pos_type}")

    lastPriceA, lastPriceB = get_price(ASSET_A, ASSET_B)
    position, qty_a, qty_b = getCurrentPos(ASSET_A, ASSET_B)

    if position == 0:
        print("No open position to liquidate.")
        send_tele(
            f"Liquidate called for {ASSET_A}/{ASSET_B} but no open pair position was found.\nReason: {reason}",
            alert_type="EXIT"
        )
        return

    equity_pre, pnl_today_pre = print_pnl_stats()

    try:
        if qty_a != 0:
            side = OrderSide.SELL if qty_a > 0 else OrderSide.BUY
            submit_order(ASSET_A, abs(qty_a), side)

        if qty_b != 0:
            side = OrderSide.SELL if qty_b > 0 else OrderSide.BUY
            submit_order(ASSET_B, abs(qty_b), side)

        print(f"LIQUIDATED: Closed all positions for {ASSET_A} and {ASSET_B}.")
        time.sleep(5)

        equity_post, pnl_today_post = print_pnl_stats()

        pnl_pair = None
        if entry_price_a is not None and entry_price_b is not None and entry_pos_type is not None:
            if entry_pos_type == "LONG_SPREAD_ENTRY":
                # long A short B
                pnl_a = (lastPriceA - entry_price_a) * abs(qty_a)
                pnl_b = (entry_price_b - lastPriceB) * abs(qty_b)
            elif entry_pos_type == "SHORT_SPREAD_ENTRY":
                # short A long B
                pnl_a = (entry_price_a - lastPriceA) * abs(qty_a)
                pnl_b = (lastPriceB - entry_price_b) * abs(qty_b)
            else:
                pnl_a = pnl_b = 0.0

            pnl_pair = pnl_a + pnl_b

            print(f"Position Sold: {ASSET_A}/{ASSET_B}")
            print(f"    {ASSET_A}: Entry ${entry_price_a:.2f}, Exit ${lastPriceA:.2f}, Qty {qty_a}")
            print(f"    {ASSET_B}: Entry ${entry_price_b:.2f}, Exit ${lastPriceB:.2f}, Qty {qty_b}")
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

        entry_price_a = entry_price_b = entry_pos_type = None

    except Exception as e:
        print(f"Liquidation error: {e}")
        send_tele(f"Liquidation failed for {ASSET_A}/{ASSET_B}: {e}", alert_type="ERROR")


def log_status(current_z, beta, p_value, hurst, price_a, price_b, position, pos_a, pos_b, acc_eqty,
               loop_start_time, is_rth):
    global last_heartbeat_time

    rth_status = "OPEN (RTH)" if is_rth else "CLOSED"

    status_msg = (
        f"\n--- {loop_start_time.strftime('%Y-%m-%d %H:%M:%S')} | MARKET: {rth_status} --- Z: {current_z:.4f} | Beta: {beta:.4f} | ADF P: {p_value:.4f} | Hurst: {hurst:.4f}\n"
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


def liveLoop():
    global last_trade_time, entry_pos_type, entry_price_a, entry_price_b
    if not load_param():
        return

    # TODO: consider removing this, trader is pretty robust in terms of its connection to telegram in the latest version
    send_tele(f"{ASSET_A}/{ASSET_B} BOT STARTED: Initial connectivity check and parameters loaded.", alert_type="INFO")

    bad_regime_counter = 0

    print(
        f"Running Pairs Trader. Check interval: {trade_interval}s. ADF Filter Threshold: P-Value <= {adf_max}. Hurst < {hurst_max}")

    start_pos, start_qty_a, start_qty_b = getCurrentPos(ASSET_A, ASSET_B)
    if start_pos != 0:
        print("⚠️ Detected existing open position on startup.")
        if start_qty_a > 0:
            entry_pos_type = "LONG_SPREAD_ENTRY"
        else:
            entry_pos_type = "SHORT_SPREAD_ENTRY"
        entry_price_a = 0.0
        entry_price_b = 0.0

    while True:
        if (datetime.now(timezone.utc) - last_trade_time).total_seconds() < cooldown_time:
            print("Cooldown active: waiting for 5 min after last trade.")
            time.sleep(trade_interval)
            continue

        loop_start_time = datetime.now()

        # initialize variables
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

            # get current position status
            position, pos_a, pos_b = getCurrentPos(ASSET_A, ASSET_B)

            print(f"\n--- ACCOUNT STATUS ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            print(f"Current Equity: ${acc_eqty:,.2f}")
            print(f"Available Cash: ${available_cash:,.2f}")
            print(f"Active Positions: {current_pos}")
            print(f"Pairs Position Status: {position} ({ASSET_A}: {pos_a:.0f}, {ASSET_B}: {pos_b:.0f})")
            print("-----------------------------------")


        except Exception as e:
            print(f"Warning: Could not fetch account statistics. Error: {e}")

        try:
            clock = trading_client.get_clock()
            is_market_open_rth = clock.is_open
            market_time_str = clock.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
            next_open_str = clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')

        except Exception as e:
            # failsafe: assume closed if clock api fails
            print(f"Error getting market clock: {e}. Assuming market is closed and waiting.")
            is_market_open_rth = False
            market_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            next_open_str = "N/A"

        if not is_market_open_rth:
            print(f"\nMarkt Closed: {market_time_str}. Next open: {next_open_str}. Current position: {position}")

            # tele alert
            current_time_utc = datetime.now(timezone.utc)
            global last_heartbeat_time  # use heartbeat timer for market close alerts too
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
            if latest_data.empty:
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
                    # non-stationary regime detected PERSISTENTLY
                    if position != 0:
                        print(
                            f"REGIME SHIFT CONFIRMED: P-Value {p_value:.4f} > {adf_max} for 3 checks. Liquidating.")
                        liquidate(reason=f"Regime shift detected: P-Value {p_value:.4f} > Threshold {adf_max}")

                    print(f"REGIME FILTER ACTIVE: P-Value {p_value:.4f}. Trading temporarily suspended.")
                else:
                    print(f"Waiting for confirmation of regime shift...")

            else:
                # reset counter if data is good
                bad_regime_counter = 0

                # exit/SL logic
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

                # entry logic
                elif position == 0:
                    if abs(current_z) >= Z_ENTRY and Z_STOP_LOSS >= abs(current_z) and hurst < hurst_max:

                        shares_a, shares_b, qty_a, qty_b, V_A, V_B = determine_sizing(
                            lastPriceA, lastPriceB, beta, spread_volatility)

                        entry_type = None

                        if qty_a > 0 and qty_b > 0:
                            if current_z < -Z_ENTRY:
                                # long A short B
                                print(f"ENTRY LONG SPREAD: Z-Score {current_z:.2f} < -{Z_ENTRY}")
                                success_a, done_qty_a = submit_order(ASSET_A, qty_a, OrderSide.BUY)
                                success_b, done_qty_b = submit_order(ASSET_B, qty_b, OrderSide.SELL)
                                if success_a and success_b: entry_type = "LONG_SPREAD_ENTRY"

                            else:
                                # short A long B
                                print(f"ENTRY SHORT SPREAD: Z-Score {current_z:.2f} > +{Z_ENTRY}")
                                success_a, done_qty_a = submit_order(ASSET_A, qty_a, OrderSide.SELL)
                                success_b, done_qty_b = submit_order(ASSET_B, qty_b, OrderSide.BUY)
                                if success_a and success_b: entry_type = "SHORT_SPREAD_ENTRY"

                            if entry_type:
                                entry_price_a = lastPriceA
                                entry_price_b = lastPriceB
                                entry_pos_type = entry_type

                                alert_msg = (
                                    f"New Pair Trade Entered: {entry_type.replace('_', ' ')}.\n"
                                    f"Z-Score: {current_z:.4f} | Beta: {beta:.4f} | Hurst: {hurst:.4f}\n"
                                    f"Orders Sent: {ASSET_A} ({done_qty_a:.0f} shares), {ASSET_B} ({done_qty_b:.0f} shares)\n"
                                    f"Current Position Status: {ASSET_A}: {qty_a:.0f}, {ASSET_B}: {qty_b:.0f}"
                                )
                                send_tele(alert_msg, alert_type="ENTRY")

                                last_trade_time = datetime.now(timezone.utc)


        except Exception as e:
            error_msg = f"An unexpected error occurred in the live loop: {e}. Waiting 5 minutes."
            print(f"{error_msg}")

            alert_msg = f"{ASSET_A}/{ASSET_B} TRADER FACED UNEXPECTED ERROR: {e}\nLast Z-Score: {current_z:.4f} | Equity: ${acc_eqty:,.2f}"
            send_tele(alert_msg, alert_type="ERROR")

            time.sleep(300)

        now = datetime.now()
        minutes = now.minute
        minutes_to_next = 15 - (now.minute % 15)
        seconds_to_wait = (minutes_to_next * 60) - now.second + 5

        if seconds_to_wait < 0:
            seconds_to_wait += 900

        print(f"⏳ Syncing with 15m candle... Sleeping for {seconds_to_wait:.0f} seconds.")
        time.sleep(seconds_to_wait)


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
        success_a, filled_a = submit_order(ASSET_A, qty_a, OrderSide.BUY)
        success_b, filled_b = submit_order(ASSET_B, qty_b, OrderSide.SELL)

        print(f"Order Results -> {ASSET_A}: success={success_a}, filled={filled_a}")
        print(f"Order Results -> {ASSET_B}: success={success_b}, filled={filled_b}")
    else:
        print("Quantity too small, skipping order to avoid fractional shares.")


def debug_model():
    print("\n===== DEBUG MODEL / SPREAD / Z-SCORE =====")

    print("Loading parameters...")
    if not load_param():
        print("❌ Could not load parameters. Exiting debug.")
        return

    global LOOKBACK_WINDOW
    if LOOKBACK_WINDOW is None:
        print("⚠ LOOKBACK_WINDOW was None. Using fallback = 5000.")
        LOOKBACK_WINDOW = 5000

    print("\nFetching lookback data...")
    data = filters_data(ASSET_A, ASSET_B, LOOKBACK_WINDOW)
    if data is None or len(data) == 0:
        print("❌ No data loaded. Exiting debug.")
        return

    print(f"Loaded {len(data)} rows of RTH minute data.")

    print("\n>>> Testing spread direction detection...")
    best = choose_best_spread(data)
    if best is None:
        print("❌ Spread test failed.")
        return

    model_type, beta, intercept, spread_series, adf_p, variance = best
    print(f"✓ Best Model: {model_type}")
    print(f"    Beta: {beta:.6f}")
    print(f"    Intercept: {intercept:.6f}")
    print(f"    ADF p-value: {adf_p:.6f}")
    print(f"    Spread Variance: {variance:.6e}")

    print("\n>>> Testing static model cache update...")
    compute_static_model(data)
    print("MODEL_CACHE CONTENTS:")
    for k, v in MODEL_CACHE.items():
        print(f"   {k}: {v}")

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