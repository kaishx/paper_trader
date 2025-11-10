# this is being run on task scheduler to run this at 9.30pm GMT+8 every day, to update the optimized_params.json.

import os
import sys
import datetime
import numpy as np
import pandas as pd
import pytz
import json
from datetime import timedelta
from alpaca.data.requests import StockBarsRequest,TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data import TimeFrameUnit
from numba import njit
from typing import Tuple, List, Dict, Any

#api
API_KEY_ID = "XXX" #HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB
API_SECRET_KEY = "XXX" #HARDCODED BUT REMOVE BEFORE PUTTING ON GITHUB

# config
TIME_CONFIG = "1m" # inputs: "1m" "15m" "1h" and "1d", but dont use 1d / should be run on 1m
ROLLING_WINDOW_BARS = 2000
CAPITAL = 10000
TX_FEE_PERCENT = 0.0005 # make it a lil realistic but it doesnt really matter
SLIPPAGE_PER_SHARE = 0.01
OPTIMIZE_FOR = "sharpe"  # "sharpe" or "calmar" / DO NOT USE FOR CALMAR, it returns a value in the billions...


# utility fns

def parse_time_config(time_config: str) -> TimeFrame:
    """ turns the config above into timeframe so alpaca can read it """
    try:
        if time_config.lower().endswith('m'):
            unit = TimeFrameUnit.Minute
            value = int(time_config[:-1])
        elif time_config.lower().endswith('h'):
            unit = TimeFrameUnit.Hour
            value = int(time_config[:-1])
        elif time_config.lower().endswith('d'):
            unit = TimeFrameUnit.Day
            value = int(time_config[:-1])
        else:
            raise ValueError("Time configuration must end with 'm', 'h', or 'd'.")

        return TimeFrame(value, unit)
    except Exception as e:
        print(f"Error parsing TIME_CONFIG '{time_config}': {e}")
        sys.exit(1)

def get_data(asset_a: str, asset_b: str, start_date: datetime.datetime, end_date: datetime.datetime,
                    tf: TimeFrame) -> pd.DataFrame:
    """ gets alpaca data and merges it / fixed 031125, merging was breaking"""
    client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
    symbols = [asset_a, asset_b]

    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start_date,
        end=end_date
    )

    bars = client.get_stock_bars(request_params).df

    # check if anythign missing
    if bars.empty:
        print(f"  ❌ No data returned for {asset_a} and {asset_b}.")
        return pd.DataFrame()

    if asset_a not in bars.index.get_level_values(0) or asset_b not in bars.index.get_level_values(0):
        print(f"  ❌ Missing data for one or both assets ({asset_a}, {asset_b}).")
        return pd.DataFrame()

    data_a = bars.loc[asset_a].rename(columns={'close': 'Close_A'})['Close_A']
    data_b = bars.loc[asset_b].rename(columns={'close': 'Close_B'})['Close_B']

    merged_data = pd.DataFrame({'Close_A': data_a, 'Close_B': data_b}).dropna()
    return merged_data


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    """cleans up the data, just removes the naninfs"""
    epsilon = 1e-10

    # Clip prices to ensure they are positive before taking the log
    data['Log_A'] = np.log(data['Close_A'].clip(lower=epsilon))
    data['Log_B'] = np.log(data['Close_B'].clip(lower=epsilon))

    # clean up nan/infs introduced during log transform just in case
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    return data



def zscore_spread_calc(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """calculates spread zscore and beta for the window but i dun rly understand the math but it works so :/"""
    if len(data) < window:
        return data

    data['Beta'] = np.zeros(len(data))
    data['Spread'] = np.zeros(len(data))
    data['Spread_Mean'] = np.zeros(len(data))
    data['Spread_Std'] = np.zeros(len(data))

    log_a = data['Log_A'].values
    log_b = data['Log_B'].values

    for i in range(window, len(data)):
        window_log_a = log_a[i - window:i]
        window_log_b = log_b[i - window:i]

        # ols beta calc
        cov_matrix = np.cov(window_log_a, window_log_b)
        beta = cov_matrix[0, 1] / cov_matrix[0, 0] if cov_matrix[0, 0] != 0 else 1.0

        data.loc[data.index[i], 'Beta'] = beta

        # spread calculation
        spread_series = window_log_b - beta * window_log_a
        data.loc[data.index[i], 'Spread'] = log_b[i] - beta * log_a[i]
        data.loc[data.index[i], 'Spread_Mean'] = spread_series.mean()
        data.loc[data.index[i], 'Spread_Std'] = spread_series.std()

    # zscore calculation
    data['Z_Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_Std']

    return data.dropna(subset=['Z_Score', 'Beta']).copy()


@njit(cache=True)
def numba_strat(
        z_scores: np.ndarray,
        betas: np.ndarray,
        close_a: np.ndarray,
        close_b: np.ndarray,
        z_entry: float,
        z_exit: float,
        z_stop_loss: float,
        capital: float,
        tx_fee_percent: float,
        slippage_per_share: float
) -> Tuple[float, float, List[Tuple[float, float, float, float, float, float]]]:
    """backtesting logic. #TODO if i rmber: implement numba cuz this shit takes FOREVER with normal python loops bruh"""

    cash = capital
    shares_a = 0.0
    shares_b = 0.0
    in_trade = False

    equity_curve = [capital]
    trade_log = []

    for i in range(len(z_scores)):
        z = z_scores[i]
        beta = betas[i]
        price_a = close_a[i]
        price_b = close_b[i]

        current_equity = cash + shares_a * price_a + shares_b * price_b

        # check if hit stoploss/exit
        if in_trade:
            # SL check
            if z >= z_stop_loss or z <= -z_stop_loss:
                # Close position
                cash += shares_a * price_a * (1 - tx_fee_percent)
                cash += shares_b * price_b * (1 - tx_fee_percent)

                pnl = (cash + shares_a * price_a + shares_b * price_b) - equity_curve[-1]
                trade_log.append((i, pnl, z, z_stop_loss, price_a, price_b))

                shares_a, shares_b = 0.0, 0.0
                in_trade = False

            # exit check long spread
            elif z <= z_exit and shares_b > 0:
                # close positions by shorting A and longing B
                cash += shares_a * price_a * (1 - tx_fee_percent)
                cash += shares_b * price_b * (1 - tx_fee_percent)

                pnl = (cash + shares_a * price_a + shares_b * price_b) - equity_curve[-1]
                trade_log.append((i, pnl, z, z_exit, price_a, price_b))

                shares_a, shares_b = 0.0, 0.0
                in_trade = False

            # exit check short spread
            elif z >= -z_exit and shares_a > 0:
                # close positions by longing A and shorting B
                cash += shares_a * price_a * (1 - tx_fee_percent)
                cash += shares_b * price_b * (1 - tx_fee_percent)

                pnl = (cash + shares_a * price_a + shares_b * price_b) - equity_curve[-1]
                trade_log.append((i, pnl, z, -z_exit, price_a, price_b))

                shares_a, shares_b = 0.0, 0.0
                in_trade = False

        # entry logic
        if not in_trade:

            # z > zentry check
            if z >= z_entry:

                capital_per_leg = current_equity / 2.0

                # long A
                shares_a = int(capital_per_leg / price_a)

                # short B scaled by beta
                shares_b = int((capital_per_leg / price_b) / beta)

                cost_a = shares_a * price_a * (1 + tx_fee_percent)
                cash -= cost_a

                revenue_b = shares_b * price_b * (1 - tx_fee_percent)
                cash += revenue_b

                in_trade = True

            # z < zentry check
            elif z <= -z_entry:

                capital_per_leg = current_equity / 2.0

                # short A
                shares_a = int(capital_per_leg / price_a)

                # long B scaled by beta
                shares_b = int((capital_per_leg / price_b) / beta)

                revenue_a = shares_a * price_a * (1 - tx_fee_percent)
                cash += revenue_a

                cost_b = shares_b * price_b * (1 + tx_fee_percent)
                cash -= cost_b

                in_trade = True

        # updater the eq cve
        equity_curve.append(cash + shares_a * price_a + shares_b * price_b)

    if in_trade:
        final_pnl = (cash + shares_a * price_a + shares_b * price_b) - equity_curve[-1]
        trade_log.append((len(z_scores) - 1, final_pnl, z, 0.0, price_a, price_b))

    return equity_curve[1:], len(trade_log), trade_log


def backtest(data: pd.DataFrame, z_entry: float, z_exit: float, z_stop_loss: float, capital: float,
                      tx_fee_percent: float, slippage_per_share: float, asset_a: str, asset_b: str) -> Tuple[
    float, float, float, float, int, pd.DataFrame, pd.DataFrame]:
    """calculate metrics, and is js a wrap for numba_strat function ltr"""

    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Z_Score', 'Beta', 'Close_A', 'Close_B']).copy()

    z_scores = data_clean['Z_Score'].values
    betas = data_clean['Beta'].values
    close_a = data_clean['Close_A'].values
    close_b = data_clean['Close_B'].values

    if len(z_scores) == 0:
        return 0.0, -float('inf'), -float('inf'), 0.0, 0, pd.DataFrame(), pd.DataFrame()

    equity_pnl, trade_count, trade_log_list = numba_strat(
        z_scores, betas, close_a, close_b,
        z_entry, z_exit, z_stop_loss,
        capital, tx_fee_percent, slippage_per_share
    )

    # metrics calculator
    pnl_df = pd.DataFrame(index=data_clean.index,
                          data={'Equity': [capital] + equity_pnl[:-1], 'PnL': np.array(equity_pnl) - capital})

    pnl_df['Returns'] = pnl_df['Equity'].diff().fillna(0)

    # annualization factor
    bars_per_day = 252 / len(data_clean.index.normalize().unique()) if len(data_clean) > 0 else 1
    ann_factor = np.sqrt(252 * bars_per_day)

    # aharpe ratio calculation requires returns based on capital, let's use percent return
    pnl_df['Percent_Returns'] = pnl_df['Equity'].pct_change().fillna(0)
    sharpe = pnl_df['Percent_Returns'].mean() / pnl_df['Percent_Returns'].std() * ann_factor if pnl_df['Percent_Returns'].std() != 0 else -float('inf')

    pnl_df['Cumulative_Equity'] = pnl_df['Equity']
    pnl_df['Peak'] = pnl_df['Cumulative_Equity'].cummax()
    pnl_df['Drawdown'] = (pnl_df['Peak'] - pnl_df['Cumulative_Equity']) / pnl_df['Peak']
    max_drawdown = pnl_df['Drawdown'].max()

    total_pnl = pnl_df['Equity'].iloc[-1] - capital

    # bro this calmar calculations is sooo fucked. chatgpt keeps tweaking when i ask a fix for this so im gonna leave it as is and i will just never optimize for calmar
    total_return_pct = (pnl_df['Equity'].iloc[-1] / pnl_df['Equity'].iloc[0]) - 1
    max_drawdown_pct = max(pnl_df['Drawdown'].max(), 1e-6)
    calmar = total_return_pct / max_drawdown_pct

    trade_df = pd.DataFrame(trade_log_list, columns=['Index_Bar', 'PnL', 'Z_Score', 'Exit_Z', 'Price_A', 'Price_B'])

    return total_pnl, sharpe, calmar, max_drawdown, trade_count, trade_df, pnl_df


def run_opt(data: pd.DataFrame, capital: float, tx_fee_percent: float, slippage_per_share: float,
                            asset_a: str, asset_b: str) -> Dict[str, Any]:
    """performs a single grid search optimization for the data"""

    # grid, super arbitrary
    Z_entry_range = np.arange(1.8, 3.8, 0.2)
    Z_exit_range = np.arange(0.2, 1.2, 0.1)
    Z_stop_loss_range = np.arange(3.0, 5.0, 0.3)

    #  OLD GRID RANGE (too coarse, replaced with finer increments)
    #  Z_entry_range = [2.0, 3.0]
    #  Z_exit_range = [0.5, 1.0]
    #  Z_stop_loss_range = [3.0, 4.0]

    best_metric = -float('inf')
    z_entry_opt, z_exit_opt, z_stop_loss_opt = 0.0, 0.0, 0.0

    print(f"  Starting IS optimization ({OPTIMIZE_FOR.upper()}) over {len(data)} bars...")

    for z_entry in Z_entry_range:
        for z_exit in Z_exit_range:
            if z_exit >= z_entry:
                continue
            for z_stop_loss in Z_stop_loss_range:

                _, sharpe, calmar, _, _, _, _ = backtest(
                    data, z_entry, z_exit, z_stop_loss,
                    capital, tx_fee_percent, slippage_per_share, asset_a, asset_b
                )

                if OPTIMIZE_FOR == "calmar":
                    metric_value = calmar
                else:
                    metric_value = sharpe

                if metric_value > best_metric:
                    best_metric = metric_value
                    z_entry_opt, z_exit_opt, z_stop_loss_opt = z_entry, z_exit, z_stop_loss

    print(f"  Optimization complete. Best {OPTIMIZE_FOR.title()}: {best_metric:.4f}")

    # return results as a dict
    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "optimal_z_entry": round(z_entry_opt, 2),
        "optimal_z_exit": round(z_exit_opt, 2),
        "optimal_z_stop_loss": round(z_stop_loss_opt, 2),
        f"in_sample_{OPTIMIZE_FOR}": round(best_metric, 4),
        "status": "OPTIMIZED",
        "metric_used": OPTIMIZE_FOR
    }



def main():
    """main"""

    #change the pairs to config for here. #TODO; change it so i can use the same format as the walkforward analysis side so i dont have to keep asking AI to return it in this format whenever i change this list
    PAIRS_CONFIG = [
        {"A": "XXX", "B": "YYY"},
    ]

    tf = parse_time_config(TIME_CONFIG)

    print(f"--- Multi Z-Score Optimizer ({TIME_CONFIG}) ---")

    # lookback window - 60 day period
    end_date = datetime.datetime.now(pytz.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=60)

    print(f"Optimization lookback: {start_date.date()} to {end_date.date()}")
    print("-" * 50)

    all_results = []

    # check each pair
    for pair in PAIRS_CONFIG:
        asset_a = pair['A']
        asset_b = pair['B']

        print(f"\nProcessing pair: {asset_a} vs {asset_b}...")

        try:
            raw_data = get_data(asset_a, asset_b, start_date, end_date, tf)

            if not raw_data.empty:
                print(f"Raw bars count - {asset_a}: {len(raw_data['Close_A'])}, {asset_b}: {len(raw_data['Close_B'])}")
                print(f"Merged bars (after dropna): {len(raw_data.dropna())}")
            
        except Exception as e:
            print(f"  ❌ Error fetching Alpaca data: {e}")
            all_results.append({"asset_a": asset_a, "asset_b": asset_b, "status": "FETCH_ERROR"})
            continue

        if raw_data.empty:
            all_results.append({"asset_a": asset_a, "asset_b": asset_b, "status": "NO_DATA"})
            continue

        raw_data = prep_data(raw_data)

        # calculate metrics
        processed_data = zscore_spread_calc(raw_data, ROLLING_WINDOW_BARS)

        if processed_data.empty:
            print(f"  ❌ Error: Not enough data ({len(raw_data)} bars) to calculate {ROLLING_WINDOW_BARS}-bar Z-Scores.")
            all_results.append({"asset_a": asset_a, "asset_b": asset_b, "status": "SHORT_DATA"})
            continue

        # run the optimization here
        pair_result = run_opt(
            processed_data,
            CAPITAL, TX_FEE_PERCENT, SLIPPAGE_PER_SHARE, asset_a, asset_b
        )
        all_results.append(pair_result)

    # output JSOn for the trader.py to see
    final_output = {
        "metadata": {
            "timestamp": datetime.datetime.now(pytz.utc).isoformat(),
            "timeframe": TIME_CONFIG,
            "rolling_window_bars": ROLLING_WINDOW_BARS,
            "optimization_lookback_days": 60
        },
        "optimization_results": all_results
    }

    print("\n" + "=" * 70)
    print("      ✨ ALL OPTIMAL Z-SCORES (Combined JSON Output) ✨")
    print("=" * 70)

    json_output_str = json.dumps(final_output, indent=4)

    print(json_output_str)

    filename = f"optimized_params.json"

    # save it in the same directory for easy reference
    try:
        with open(filename, 'w') as f:
            f.write(json_output_str)
        print(f"\n✅ Successfully saved optimization parameters to {filename}")
    except IOError as e:
        print(f"\n❌ Error saving file {filename}: {e}")


if __name__ == "__main__":
    main()

    # DEBUG ONLY: Run one pair manually to test optimizer timings
    # test_pair = {"A": "MET", "B": "PRU"}
    # df = get_data(test_pair["A"], test_pair["B"], start_date, end_date, tf)
    # print(run_opt(prep_data(zscore_spread_calc(df, 2000)), CAPITAL, TX_FEE_PERCENT, SLIPPAGE_PER_SHARE, test_pair["A"], test_pair["B"]))
