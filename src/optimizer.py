# if you wish to mimic how the wfa works, you can use optimizer on every 1st and 3rd sundays of the month.

import os
import json
import time
import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from numba import jit, njit, float64
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from dotenv import load_dotenv

load_dotenv()

API_KEY_ID = os.getenv("APCA_API_KEY_ID")
API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY_ID or not API_SECRET_KEY:
    print("!!! api keys missing in environment !!!")

PAIRS_CONFIG_FILE = "pairs.json"
pairs = []
if os.path.exists(PAIRS_CONFIG_FILE):
    with open(PAIRS_CONFIG_FILE, "r") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs from config.")
else:
    print("!!! pairs.json not found. !!!")

cptl = 100000.0
ABS_DOLLAR_STOP = 2500.0
txfee = 0.0001
slippagefee = 0.01

hurst_env = os.getenv("hurstMax")
hurstMax = float(hurst_env) if hurst_env else 0.80

minTrades = 5

Z_ENTRY_GRID = [1.7, 1.9, 2.1, 2.3, 2.5]
Z_EXIT_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
Z_STOP_LOSS_GRID = [3.0, 3.3, 3.6, 3.9, 4.2, 4.5]

LOOKBACK_DAYS = 100
START_DATE = datetime.now() - timedelta(days=LOOKBACK_DAYS)
END_DATE = datetime.now() - timedelta(days=1)

PARAMS_FILE = "optimized_params.json"

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


def calc_spread_z(data, lookback=390):
    Y = data['Log_A'].values
    X = data['Log_B'].values

    data['Beta'] = calc_kalman(Y, X)

    rolling_mean_Y = data['Log_A'].rolling(window=lookback).mean()
    rolling_mean_X = data['Log_B'].rolling(window=lookback).mean()
    data['Alpha'] = rolling_mean_Y - data['Beta'] * rolling_mean_X

    data['Spread'] = data['Log_A'] - (data['Alpha'] + data['Beta'] * data['Log_B'])

    rolling_mean_spread = data['Spread'].shift(1).rolling(window=lookback).mean()
    rolling_std_spread = data['Spread'].shift(1).rolling(window=lookback).std()
    
    data['Spread_Std'] = rolling_std_spread

    data['Z_Score'] = (data['Spread'] - rolling_mean_spread) / (rolling_std_spread + 1e-9)

    hurst_win = 100
    data['Hurst'] = calc_hurst(data['Spread'].values, window=hurst_win)

    data.dropna(subset=['Beta', 'Alpha', 'Z_Score', 'Hurst', 'Spread_Std'], inplace=True)
    return data


def checkADF(data):
    if data.empty: return 1.0
    Y = data['Log_A']
    X = data['Log_B']
    X = sm.add_constant(X)
    try:
        model = sm.OLS(Y, X).fit()
        residuals = model.resid
        adf_result = adfuller(residuals)
        return adf_result[1]
    except:
        return 1.0


@njit
def backtest(close_a, close_b, z_score, hurst, beta, spread_std,
             z_entry, z_exit, z_stop_loss, cptl, tx_fee, slippage,
             hurst_thresh, abs_stop):

    # consider splitting this in the future to make it more readable

    n_bars = len(close_a)
    pnl_array = np.zeros(n_bars, dtype=np.float64)
    position = 0.0
    entry_price_a = 0.0
    entry_price_b = 0.0
    n_A = 0.0
    n_B = 0.0

    for i in range(n_bars - 1):
        current_z = z_score[i]
        current_hurst = hurst[i]
        current_beta = beta[i]
        current_vol = spread_std[i]

        prev_position = position
        current_price_a = close_a[i]
        current_price_b = close_b[i]
        next_price_a = close_a[i + 1]
        next_price_b = close_b[i + 1]
        is_final_bar = (i == n_bars - 2)

        is_dollar_stop = False
        if prev_position != 0:
            pnl_check_a = 0.0
            pnl_check_b = 0.0
            if prev_position == 1:
                pnl_check_a = (current_price_a - entry_price_a) * n_A
                pnl_check_b = (entry_price_b - current_price_b) * n_B
            else:
                pnl_check_a = (entry_price_a - current_price_a) * n_A
                pnl_check_b = (current_price_b - entry_price_b) * n_B

            if (pnl_check_a + pnl_check_b) <= -abs_stop:
                is_dollar_stop = True

        if prev_position != 0:
            is_mean_reversion = np.abs(current_z) <= z_exit
            is_z_stop = np.abs(current_z) >= z_stop_loss

            if is_mean_reversion or is_z_stop or is_dollar_stop or is_final_bar:
                if prev_position == 1:
                    pnl_a = (next_price_a - entry_price_a) * n_A
                    pnl_b = (entry_price_b - next_price_b) * n_B
                else:
                    pnl_a = (entry_price_a - next_price_a) * n_A
                    pnl_b = (next_price_b - entry_price_b) * n_B

                gross_pnl = pnl_a + pnl_b
                entry_val = entry_price_a * n_A + entry_price_b * n_B
                exit_val = next_price_a * n_A + next_price_b * n_B
                cost = tx_fee * (entry_val + exit_val) + (slippage * (n_A + n_B) * 2.0)

                pnl_array[i + 1] = gross_pnl - cost
                position = 0.0
                n_A = 0.0
                n_B = 0.0

        elif prev_position == 0 and not is_final_bar:
            if np.abs(current_z) > z_entry and current_hurst < hurst_thresh:
                vol_scale = 1.0
                if current_vol > 0:
                    vol_scale = 0.15 / current_vol
                    if vol_scale > 1.0: vol_scale = 1.0

                adj_capital = cptl * vol_scale
                b_abs = np.abs(current_beta)
                if b_abs == 0: b_abs = 1.0

                val_a = adj_capital / (1.0 + b_abs)
                val_b = adj_capital - val_a

                n_A = np.floor(val_a / next_price_a)
                n_B = np.floor(val_b / next_price_b)

                entry_price_a = next_price_a
                entry_price_b = next_price_b

                if current_z < -z_entry:
                    position = 1.0
                else:
                    position = -1.0

    return pnl_array


def prepData(client, symbol_a, symbol_b):
    print(f"getting data from ({START_DATE.strftime('%Y-%m-%d')}) for {symbol_a}/{symbol_b}...")
    req = StockBarsRequest(
        symbol_or_symbols=[symbol_a, symbol_b],
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=START_DATE,
        end=END_DATE,
        feed=DataFeed.IEX
    )
    try:
        bars = client.get_stock_bars(req).df
        df_a = bars.loc[symbol_a]['close'].rename(f'Close_{symbol_a}')
        df_b = bars.loc[symbol_b]['close'].rename(f'Close_{symbol_b}')

        df = pd.concat([df_a, df_b], axis=1, join='inner')
        df['Log_A'] = np.log(df[f'Close_{symbol_a}'])
        df['Log_B'] = np.log(df[f'Close_{symbol_b}'])

        return df
    except Exception as e:
        print(f"!!! error fetching data: {e} !!!")
        return pd.DataFrame()


def getBestParams(symbol_a, symbol_b, df, cptl, tx_fee, slippage):
    bars_per_day = 26
    required_bars = 60 * bars_per_day

    if len(df) < required_bars + 100:
        print(f"!!! Not enough data: Have {len(df)}, Need {required_bars} !!!")
        return None

    opt_df = df.iloc[-required_bars:].copy()
    df_calc = calc_spread_z(df.copy(), lookback=390)
    opt_df_ready = df_calc.iloc[-required_bars:].copy()

    # CHECK ADF ON OLS (Critical Fix)
    adf_p_value = checkADF(opt_df_ready)
    print(f"ADF (OLS) p-value: {adf_p_value:.4f}")

    if adf_p_value > 0.20:  # Match cpp_wfa threshold
        print(f"Skipping {symbol_a}/{symbol_b}: Non-stationary (p={adf_p_value:.3f})")
        return None

    close_a = opt_df_ready[f'Close_{symbol_a}'].values
    close_b = opt_df_ready[f'Close_{symbol_b}'].values
    z_score = opt_df_ready['Z_Score'].values
    hurst = opt_df_ready['Hurst'].values
    beta = opt_df_ready['Beta'].values
    spread_std = opt_df_ready['Spread_Std'].values

    best_sharpe = -999.0
    best_params = None

    param_combi = list(itertools.product(Z_ENTRY_GRID, Z_EXIT_GRID, Z_STOP_LOSS_GRID))

    for z_entry, z_exit, z_sl in param_combi:
        if not (z_exit < z_entry < z_sl): continue

        pnl = backtest(close_a, close_b, z_score, hurst, beta, spread_std,
                       z_entry, z_exit, z_sl,
                       cptl, tx_fee, slippage, hurstMax, ABS_DOLLAR_STOP)

        trade_count = np.count_nonzero(pnl)
        if trade_count < minTrades: continue

        total_pnl = np.sum(pnl)
        if total_pnl == 0:
            sharpe = 0.0
        else:
            returns = pnl[pnl != 0] / cptl
            if len(returns) > 1:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(len(returns))
            else:
                sharpe = 0.0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (z_entry, z_exit, z_sl)

    if best_params is None: return None

    return {
        "z_entry": best_params[0],
        "z_exit": best_params[1],
        "z_sl": best_params[2],
        "sharpe": best_sharpe,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "adf_p_value": adf_p_value
    }

if __name__ == "__main__":
    client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)

    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            try:
                final_params = json.load(f)
            except:
                final_params = {}
    else:
        final_params = {}

    for pair in pairs:
        sym_a, sym_b = pair
        pair_key = f"{sym_a}/{sym_b}"

        print(f"\noptimizing {pair_key}...")

        df = prepData(client, sym_a, sym_b)
        if df.empty: continue

        result = getBestParams(sym_a, sym_b, df, cptl, txfee, slippagefee)

        if result:
            final_params[pair_key] = result
            print(f"New params: Entry {result['z_entry']} | Exit {result['z_exit']} | SL {result['z_sl']}")
            print(f"stats: Sharpe {result['sharpe']:.2f} | adf {result['adf_p_value']:.3f}")
        else:
            print("!!! optimization failed. no profitable settings found !!!")

    with open(PARAMS_FILE, 'w') as f:
        json.dump(final_params, f, indent=4)

    print(f"\nsaved optimized parameters to {PARAMS_FILE}")