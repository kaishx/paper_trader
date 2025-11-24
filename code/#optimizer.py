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

API_KEY_ID = "PKCHRDERPHH52D5RJN3UNEYKBU"  # HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB
API_SECRET_KEY = "EFWXvL7vkmnSzSLVUpfqDB9tBrgfNms7PWdjrwn7rQ3c"  # HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB

# pair ; RMBER the format here is different from the wfa side ["XXX", "YYY"]
PAIRS_TO_OPTIMIZE = [
    ["ICLR", "IQV"],
    ["PZZA", "DPZ"],
]

cptl = 10000.0
txfee = 0.0001 #THIS IS NOT IN PERCENT BTW. SO THIS NUMBER * 100 = percentage
slippagefee = 0.01

# not using ADF here as i wnt this to be an optimizer regardless of regime during this "IS". letting the trader decide if the pair is good enough to trade in real time.
hurstMax = 0.80

Z_ENTRY_GRID = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1]
Z_EXIT_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
Z_STOP_LOSS_GRID = [3.3, 3.6, 3.9, 4.2, 4.5, 4.8]

LOOKBACK_DAYS = 365
START_DATE = datetime.now() - timedelta(days=LOOKBACK_DAYS)
END_DATE = datetime.now() - timedelta(days=1)

PARAMS_FILE = "optimized_params.json"

@njit
def calculate_kalman_beta(y, x, delta=1e-4, ve=1e-3):
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
def calculate_rolling_hurst(series, window=100):
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


def calculate_spread_and_zscore(data, lookback=78):
    Y = data['Log_A'].values
    X = data['Log_B'].values

    data['Beta'] = calculate_kalman_beta(Y, X)

    rolling_mean_Y = data['Log_A'].rolling(window=lookback).mean()
    rolling_mean_X = data['Log_B'].rolling(window=lookback).mean()
    data['Alpha'] = rolling_mean_Y - data['Beta'] * rolling_mean_X

    data['Spread'] = data['Log_A'] - (data['Alpha'] + data['Beta'] * data['Log_B'])

    rolling_mean_spread = data['Spread'].rolling(window=lookback).mean()
    rolling_std_spread = data['Spread'].rolling(window=lookback).std()
    data['Z_Score'] = (data['Spread'] - rolling_mean_spread) / (rolling_std_spread + 1e-9)

    hurst_win = 100
    data['Hurst'] = calculate_rolling_hurst(data['Spread'].values, window=hurst_win)

    data.dropna(subset=['Beta', 'Alpha', 'Z_Score', 'Hurst'], inplace=True)
    return data


def check_cointegration_adf(data):
    # KEEPING this for legacy, due to my revised nature of howim gonna use ADF
    spread = data['Spread'].values
    spread = spread[~np.isnan(spread)]

    if len(spread) < 30:
        return 1.0

    adf_result = adfuller(spread)
    p_value = adf_result[1]
    return p_value

@njit
def numba_backtest_core(close_a, close_b, z_score, hurst, z_entry, z_exit, z_stop_loss, cptl, tx_fee, slippage,
                        hurst_thresh):
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
        prev_position = position
        next_price_a = close_a[i + 1]
        next_price_b = close_b[i + 1]

        is_final_bar = (i == n_bars - 2)

        if prev_position != 0:
            is_mean_reversion = np.abs(current_z) <= z_exit
            is_stop_loss = np.abs(current_z) >= z_stop_loss

            if is_mean_reversion or is_stop_loss or is_final_bar:
                if prev_position == 1: 
                    pnl_a = (next_price_a - entry_price_a) * n_A
                    pnl_b = (entry_price_b - next_price_b) * n_B
                else:
                    pnl_a = (entry_price_a - next_price_a) * n_A
                    pnl_b = (next_price_b - entry_price_b) * n_B

                gross_pnl = pnl_a + pnl_b

                entry_val = entry_price_a * n_A + entry_price_b * n_B
                exit_val = next_price_a * n_A + next_price_b * n_B
                cost_fee = tx_fee * (entry_val + exit_val)
                cost_slip = slippage * (n_A + n_B) * 2.0

                pnl_array[i + 1] = gross_pnl - (cost_fee + cost_slip)
                position = 0.0

        elif prev_position == 0 and not is_final_bar:
            if np.abs(current_z) >= z_entry and current_hurst < hurst_thresh:
                # Simple 50/50 cptl split for optimization speed
                # traer uses beta-weighted, but this is close enough for param finding
                n_A = (cptl / 2) / next_price_a
                n_B = (cptl / 2) / next_price_b

                entry_price_a = next_price_a
                entry_price_b = next_price_b

                if current_z < -z_entry:
                    position = 1.0
                else:
                    position = -1.0

    return pnl_array

def prepare_data(client, symbol_a, symbol_b):
    print(f"Fetching 1 Year Data ({START_DATE.strftime('%Y-%m-%d')}) for {symbol_a}/{symbol_b}...")
    req = StockBarsRequest(
        symbol_or_symbols=[symbol_a, symbol_b],
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=START_DATE,
        end=END_DATE,
        feed=DataFeed.SIP
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
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def find_best_params(symbol_a, symbol_b, df, cptl, tx_fee, slippage):
    df = calculate_spread_and_zscore(df)

    current_p_value = check_cointegration_adf(df)
    print(f"Current ADF p-value: {current_p_value:.4f} (Recorded, not filtered)")

    opt_start_idx = max(0, len(df) - int(120 * 26))
    opt_df = df.iloc[opt_start_idx:].copy()

    if len(opt_df) < 100:
        print("Not enough recent data to optimize.")
        return None

    close_a = opt_df[f'Close_{symbol_a}'].values
    close_b = opt_df[f'Close_{symbol_b}'].values
    z_score = opt_df['Z_Score'].values
    hurst = opt_df['Hurst'].values

    best_sharpe = -999.0
    best_params = (2.1, 0.1, 3.3)  # default backup

    param_combinations = list(itertools.product(Z_ENTRY_GRID, Z_EXIT_GRID, Z_STOP_LOSS_GRID))

    for z_entry, z_exit, z_sl in param_combinations:
        if not (z_exit < z_entry < z_sl):
            continue

        pnl = numba_backtest_core(close_a, close_b, z_score, hurst, z_entry, z_exit, z_sl,
                                  cptl, tx_fee, slippage, hurstMax)

        total_pnl = np.sum(pnl)
        if total_pnl == 0:
            sharpe = 0.0
        else:
            returns = pnl[pnl != 0] / cptl
            if len(returns) > 1:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(len(returns))  # rough estimate
            else:
                sharpe = 0.0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (z_entry, z_exit, z_sl)

    return {
        "z_entry": best_params[0],
        "z_exit": best_params[1],
        "z_sl": best_params[2],
        "sharpe": best_sharpe,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "adf_p_value": current_p_value
    }

if __name__ == "__main__":
    client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)

    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            final_params = json.load(f)
    else:
        final_params = {}

    for pair in PAIRS_TO_OPTIMIZE:
        sym_a, sym_b = pair
        pair_key = f"{sym_a}/{sym_b}"

        print(f"\nOptimizing {pair_key}...")

        df = prepare_data(client, sym_a, sym_b)
        if df.empty: continue

        result = find_best_params(sym_a, sym_b, df, cptl, txfee, slippagefee)

        if result:
            final_params[pair_key] = result
            print(f"      New Params: Entry {result['z_entry']} | Exit {result['z_exit']} | SL {result['z_sl']}")
            print(f"      Stats: Sharpe {result['sharpe']:.2f} | ADF {result['adf_p_value']:.3f}")
        else:
            print("     Optimization failed.")
#
    with open(PARAMS_FILE, 'w') as f:
        json.dump(final_params, f, indent=4)

    print(f"\nSaved optimized parameters to {PARAMS_FILE}")