# Pairs Trading Paper Trading Algorithm

---

This is a pairs trading algorithm utilizing Kalman Filters and Hurst Exponents to trade mean-reverting pairs. The system is engineered to operate autonomously on the Alpaca Paper Trading API. It is specifically designed to be robust against the data sparsity and latency constraints of the Alpaca Free Tier (IEX Feed).

Below is the architecture of the whole system:

![System Architecture](assets/designnew.png)

*Figure 1: System Architecture demonstrating the decoupling of the Optimization Engine (Historical Calibration) from the Live Execution Engine. The Controller manages multiprocessing to ensure adherence to API rate limits, while the Trader operates on a decoupled asynchronous loop.*

---

## This repository contains the code for:


**Controller**: Manages the paper-traders concurrently (seeking to run about 20-25 traders)

**Trader**: Interacts with the Alpaca API to simulate the trading while calculating the kalman beta and other metrics.

**Optimizer**: Backtests recent data to calibrate optimal Z-score entry/exit thresholds for the current trading session.

---

**The system monitors equity and positions in real time. Results can be viewed on my Telegram channel:**

[Telegram Channel (equity & positions updates every few minutes)](https://t.me/+12M82bTPLAtjMzZl)

---

> **Note:** All mathematical logic has been manually reviewed. This project is for educational purposes only and is a proof of concept. It DEFINITELY CONTAINS bugs, mistakes and heuristic parameter choices.

