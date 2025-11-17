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
from typing import Tuple, List, Dict, Any

# ===== CONFIG =====
API_KEY_ID = "xxx"
API_SECRET_KEY = "xxx"

timeInterval = "1m"
CAPITAL = 10000
tx_fee = 0.0005
slippage = 0.01
optimizeFor = "sharpe"  # "sharpe" or "calmar"
lookback_days = 50 
PAIRS_CONFIG = [
    {"A": "xxx", "B": "yyy"},
]

def get_time(timeInterval: str) -> TimeFrame:
    unit_map = {"m": TimeFrameUnit.Minute, "h": TimeFrameUnit.Hour, "d": TimeFrameUnit.Day}
    unit = unit_map.get(timeInterval[-1].lower())
    if not unit:
        raise ValueError("Time config must end with 'm', 'h', or 'd'.")
    return TimeFrame(int(timeInterval[:-1]), unit)


def get_data(asset_a: str, asset_b: str, start_date: datetime.datetime, end_date: datetime.datetime, tf: TimeFrame) -> pd.DataFrame:
    client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
    request = StockBarsRequest(symbol_or_symbols=[asset_a, asset_b], timeframe=tf, start=start_date, end=end_date)
    bars = client.get_stock_bars(request).df

    if bars.empty:
        return pd.DataFrame()

    data_a = bars.loc[asset_a]['close'].rename('Close_A')
    data_b = bars.loc[asset_b]['close'].rename('Close_B')
    return pd.DataFrame({"Close_A": data_a, "Close_B": data_b}).dropna()


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-10
    data['Log_A'] = np.log(data['Close_A'].clip(lower=epsilon))
    data['Log_B'] = np.log(data['Close_B'].clip(lower=epsilon))
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def main_calc(data: pd.DataFrame) -> pd.DataFrame:
    log_a = data['Log_A'].values
    log_b = data['Log_B'].values

    cov_matrix = np.cov(log_a, log_b)
    beta = cov_matrix[0, 1] / cov_matrix[0, 0] if cov_matrix[0, 0] != 0 else 1.0

    spread = log_b - beta * log_a
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std

    data['Beta'] = beta
    data['Spread'] = spread
    data['Spread_Mean'] = spread_mean
    data['Spread_Std'] = spread_std
    data['Z_Score'] = z_score

    return data.copy()

# uses numba to make things go faster, but i dont exactly know how
@njit(cache=True)
def numba_strat(z_scores, betas, close_a, close_b, z_entry, z_exit, z_stop_loss,
                capital, tx_fee, slippage):
    cash = capital
    shares_a = 0.0
    shares_b = 0.0
    in_trade = False
    equity_curve = [capital]

    for i in range(len(z_scores)):
        z = z_scores[i]
        beta = betas[i]
        price_a = close_a[i]
        price_b = close_b[i]

        current_equity = cash + shares_a * price_a + shares_b * price_b

        # exit lgc
        if in_trade:
            if z >= z_stop_loss or z <= -z_stop_loss:
                cash += shares_a * price_a * (1 - tx_fee) + shares_b * price_b * (1 - tx_fee)
                shares_a, shares_b = 0.0, 0.0
                in_trade = False
            elif (z <= z_exit and shares_b > 0) or (z >= -z_exit and shares_a > 0):
                cash += shares_a * price_a * (1 - tx_fee) + shares_b * price_b * (1 - tx_fee)
                shares_a, shares_b = 0.0, 0.0
                in_trade = False

        # entry lgc
        if not in_trade:
            capital_per_leg = current_equity / 2.0
            if z >= z_entry:
                shares_a = int(capital_per_leg / price_a)
                shares_b = int((capital_per_leg / price_b) / beta)
                cash -= shares_a * price_a * (1 + tx_fee)
                cash += shares_b * price_b * (1 - tx_fee)
                in_trade = True
            elif z <= -z_entry:
                shares_a = int(capital_per_leg / price_a)
                shares_b = int((capital_per_leg / price_b) / beta)
                cash += shares_a * price_a * (1 - tx_fee)
                cash -= shares_b * price_b * (1 + tx_fee)
                in_trade = True

        equity_curve.append(cash + shares_a * price_a + shares_b * price_b)

    return equity_curve[1:]



def backtest(data: pd.DataFrame, z_entry, z_exit, z_stop_loss):
    z_scores = data['Z_Score'].values
    betas = data['Beta'].values
    close_a = data['Close_A'].values
    close_b = data['Close_B'].values

    equity_curve = numba_strat(z_scores, betas, close_a, close_b, z_entry, z_exit, z_stop_loss,
                               CAPITAL, tx_fee, slippage)
    returns = pd.Series(equity_curve).pct_change().fillna(0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else -np.inf
    calmar = (equity_curve[-1]/equity_curve[0] - 1) / max(1e-6, 1 - np.min(equity_curve)/equity_curve[0])
    return sharpe, calmar


def run_opt(data: pd.DataFrame, asset_a: str, asset_b: str) -> Dict[str, Any]:
    Z_entry_range = np.arange(1.8, 3.8, 0.2)
    Z_exit_range = np.arange(0.2, 1.2, 0.1)
    Z_stop_loss_range = np.arange(3.0, 5.0, 0.3)

    best_metric = -np.inf
    z_entry_opt = z_exit_opt = z_stop_loss_opt = 0.0

    for z_entry in Z_entry_range:
        for z_exit in Z_exit_range:
            if z_exit >= z_entry:
                continue
            for z_stop_loss in Z_stop_loss_range:
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
        "status": "OPTIMIZED",
        "metric_used": optimizeFor
    }


def main():
    tf = get_time(timeInterval)
    end_date = end_date = datetime.datetime.now(pytz.utc) - timedelta(days=1) #i cant use the most recent data due to alpaca SIP rules so..
    start_date = end_date - timedelta(days=lookback_days)
    all_results = []

    for pair in PAIRS_CONFIG:
        asset_a, asset_b = pair["A"], pair["B"]
        raw_data = get_data(asset_a, asset_b, start_date, end_date, tf)
        if raw_data.empty:
            all_results.append({"asset_a": asset_a, "asset_b": asset_b, "status": "NO_DATA"})
            continue
        processed = prep_data(raw_data)
        processed = main_calc(processed)
        all_results.append(run_opt(processed, asset_a, asset_b))

    final_output = {
        "metadata": {
            "timestamp": datetime.datetime.now(pytz.utc).isoformat(),
            "timeframe": timeInterval,
            "optimization_lookback_days": lookback_days
        },
        "optimization_results": all_results
    }

    with open("optimized_params.json", "w") as f:
        json.dump(final_output, f, indent=4)

    print("✅ Optimization complete and saved to optimized_params.json")


if __name__ == "__main__":
    main()
