import datetime
import pytz
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from alpaca.data.requests import StockBarsRequest, TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data import TimeFrameUnit
from numba import njit
from typing import Dict, Any

API_KEY_ID = "XXX"  # HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB
API_SECRET_KEY = "XXX"  # HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB

# params
timeInterval = "15m"
cptl = 10000
tx_fee = 0.0005
slippage = 0.01
optimizeFor = "sharpe"

# data fetching config
lookback_days = 250
rolling_window = 100
hurst_max = 0.6

PAIRS_CONFIG = [
    {"A": "XXX", "B": "YYY"},
]


def get_time(timeInterval: str) -> TimeFrame:
    unit_map = {"m": TimeFrameUnit.Minute, "h": TimeFrameUnit.Hour, "d": TimeFrameUnit.Day}
    unit = unit_map.get(timeInterval[-1].lower())
    if not unit:
        raise ValueError("invald time unit")
    return TimeFrame(int(timeInterval[:-1]), unit)


def get_data(asset_a: str, asset_b: str, start_date: datetime.datetime, end_date: datetime.datetime,
             tf: TimeFrame) -> pd.DataFrame:
    client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
    print(f"grabbing data for {asset_a}/{asset_b}...")

    request = StockBarsRequest(symbol_or_symbols=[asset_a, asset_b], timeframe=tf, start=start_date, end=end_date)
    try:
        bars = client.get_stock_bars(request).df
        if bars.empty:
            return pd.DataFrame()

        data_a = bars.loc[asset_a]['close'].rename('Close_A')
        data_b = bars.loc[asset_b]['close'].rename('Close_B')
        return pd.DataFrame({"Close_A": data_a, "Close_B": data_b}).dropna()
    except Exception as e:
        print(f"alpaca api choked: {e}")
        return pd.DataFrame()


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-10
    data['Log_A'] = np.log(data['Close_A'].clip(lower=epsilon))
    data['Log_B'] = np.log(data['Close_B'].clip(lower=epsilon))
    return data

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


def main_calc(data: pd.DataFrame, window: int) -> pd.DataFrame:
    Y = data['Log_A'].values
    X = data['Log_B'].values

    data['Beta'] = calculate_kalman_beta(Y, X)

    data['Spread'] = data['Log_A'] - data['Beta'] * data['Log_B']

    data['Hurst'] = calculate_rolling_hurst(data['Spread'].values, window=100)

    spread_mean = data['Spread'].rolling(window=window).mean()
    spread_std = data['Spread'].rolling(window=window).std()

    data['Z_Score'] = (data['Spread'] - spread_mean) / spread_std

    return data.dropna()


@njit(cache=True)
def numba_strat(z_scores, betas, close_a, close_b, hurst, z_entry, z_exit, z_stop_loss, hurst_max,
                cptl, tx_fee, slippage):
    cash = cptl
    shares_a = 0.0
    shares_b = 0.0
    in_trade = False

    equity_curve = np.zeros(len(z_scores))
    equity_curve[0] = cptl

    # 0 = none, 1 = long spread, -1 = short spread
    trade_dir = 0

    for i in range(1, len(z_scores)):
        z = z_scores[i]
        h = hurst[i]
        beta = betas[i]
        price_a = close_a[i]
        price_b = close_b[i]

        current_equity = cash + (shares_a * price_a) + (shares_b * price_b)

        if in_trade:
            should_exit = False

            if z >= z_stop_loss or z <= -z_stop_loss:
                should_exit = True

            if trade_dir == 1 and z >= -z_exit:
                should_exit = True
            elif trade_dir == -1 and z <= z_exit:
                should_exit = True

            if should_exit:
                proceeds_a = -(shares_a * price_a)
                cost_a = abs(shares_a * price_a) * tx_fee

                proceeds_b = -(shares_b * price_b)
                cost_b = abs(shares_b * price_b) * tx_fee

                cash += (proceeds_a - cost_a) + (proceeds_b - cost_b)

                shares_a = 0.0
                shares_b = 0.0
                in_trade = False
                trade_dir = 0

        if not in_trade:
            if abs(z) >= z_entry and h < hurst_max:

                cptl_per_leg = current_equity / 2.0

                if z <= -z_entry:
                    qty_a = int(cptl_per_leg / price_a)
                    qty_b = int((cptl_per_leg / price_b) / beta)

                    shares_a = qty_a
                    cash -= (qty_a * price_a) * (1 + tx_fee)

                    shares_b = -qty_b
                    cash += (qty_b * price_b) * (1 - tx_fee)

                    in_trade = True
                    trade_dir = 1

                elif z >= z_entry:
                    # Short Spread logic
                    qty_a = int(cptl_per_leg / price_a)
                    qty_b = int((cptl_per_leg / price_b) / beta)

                    shares_a = -qty_a
                    cash += (qty_a * price_a) * (1 - tx_fee)

                    shares_b = qty_b
                    cash -= (qty_b * price_b) * (1 + tx_fee)

                    in_trade = True
                    trade_dir = -1

        equity_curve[i] = cash + (shares_a * price_a) + (shares_b * price_b)

    return equity_curve


def backtest(data: pd.DataFrame, z_entry, z_exit, z_stop_loss):
    z_scores = data['Z_Score'].values
    betas = data['Beta'].values
    close_a = data['Close_A'].values
    close_b = data['Close_B'].values
    hurst = data['Hurst'].values

    equity_curve = numba_strat(
        z_scores, betas, close_a, close_b, hurst,
        z_entry, z_exit, z_stop_loss, hurst_max,
        cptl, tx_fee, slippage
    )

    returns = pd.Series(equity_curve).pct_change().fillna(0)

    if returns.std() == 0:
        sharpe = -np.inf
    else:
        # 15m bars -> 26 bars per day * 252 days
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 26)

    total_return = equity_curve[-1] / equity_curve[0] - 1
    max_drawdown = 1 - np.min(equity_curve) / np.max(np.maximum.accumulate(equity_curve))
    calmar = total_return / max(1e-6, max_drawdown)

    return sharpe, calmar


def run_opt(data: pd.DataFrame, asset_a: str, asset_b: str) -> Dict[str, Any]:
    Z_entry_range = np.arange(1.0, 3.0, 0.25)
    Z_exit_range = np.arange(0.0, 0.8, 0.2)
    Z_stop_loss_range = np.arange(3.0, 5.0, 0.5)

    best_metric = -np.inf
    z_entry_opt = z_exit_opt = z_stop_loss_opt = 0.0

    for z_entry in Z_entry_range:
        for z_exit in Z_exit_range:
            if z_exit >= z_entry: continue

            for z_stop_loss in Z_stop_loss_range:
                if z_stop_loss <= z_entry: continue

                sharpe, calmar = backtest(data, z_entry, z_exit, z_stop_loss)
                metric = calmar if optimizeFor == "calmar" else sharpe

                if metric > best_metric:
                    best_metric = metric
                    z_entry_opt, z_exit_opt, z_stop_loss_opt = z_entry, z_exit, z_stop_loss

    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "optimal_z_entry": round(z_entry_opt, 2),
        "optimal_z_exit": round(z_exit_opt, 2),
        "optimal_z_stop_loss": round(z_stop_loss_opt, 2),
        f"in_sample_{optimizeFor}": round(best_metric, 4),
        "metadata": {"rolling_window": rolling_window},
        "status": "OPTIMIZED"
    }


def main():
    tf = get_time(timeInterval)
    end_date = datetime.datetime.now(pytz.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)

    all_results = []

    print(f"Starting Opt (Lookback: {lookback_days} days, Window: {rolling_window} bars, TF: {timeInterval})...")

    for pair in PAIRS_CONFIG:
        asset_a, asset_b = pair["A"], pair["B"]

        raw_data = get_data(asset_a, asset_b, start_date, end_date, tf)

        if raw_data.empty or len(raw_data) < rolling_window + 50:
            print(f"Skipping {asset_a}/{asset_b}: Not enough rows.")
            all_results.append({"asset_a": asset_a, "asset_b": asset_b, "status": "INSUFFICIENT_DATA"})
            continue

        processed = prep_data(raw_data)
        processed = main_calc(processed, rolling_window)

        if processed.empty:
            print(f"Skipping {asset_a}/{asset_b}: Data evaporated after rolling window.")
            continue

        result = run_opt(processed, asset_a, asset_b)
        print(
            f" -> {asset_a}/{asset_b}: Z-Entry {result['optimal_z_entry']} | Sharpe: {result.get(f'in_sample_{optimizeFor}', 0)}")
        all_results.append(result)

    final_output = {
        "metadata": {
            "timestamp": datetime.datetime.now(pytz.utc).isoformat(),
            "timeframe": timeInterval,
            "optimization_lookback_days": lookback_days,
            "rolling_window_bars": rolling_window
        },
        "optimization_results": all_results
    }

    with open("optimized_params.json", "w") as f:
        json.dump(final_output, f, indent=4)

    print("\nDone. params saved to optimized_params.json")


if __name__ == "__main__":
    main()
