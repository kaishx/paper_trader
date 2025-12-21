import subprocess
import sys
import time
import threading
import json
import os

PAIRS_CONFIG_FILE = "pairs.json"
PARAMS_FILE = "optimized_params.json"

if not os.path.exists(PAIRS_CONFIG_FILE):
    raise FileNotFoundError(f"!!! {PAIRS_CONFIG_FILE} not found. please create/import it !!!")

with open(PAIRS_CONFIG_FILE, "r") as f:
    all_pairs = json.load(f)

if not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"!!! {PARAMS_FILE} not found. No pairs can be filtered !!!")

with open(PARAMS_FILE, "r") as f:
    optimized_params = json.load(f)

filtered_pairs = []
for pair in all_pairs:
    asset_a, asset_b = pair[0], pair[1]
    pair_key = f"{asset_a}/{asset_b}"

    if pair_key in optimized_params:
        filtered_pairs.append(pair)
    else:
        print(f"Skipping {pair_key}: Not found in {PARAMS_FILE}")

print(f"Loaded {len(filtered_pairs)} active pairs from config.")

traderScript = "trader.py"
delay = 5  # seconds between launching each trader


def monitor(proc, asset_a, asset_b):
    for line in iter(proc.stdout.readline, ''):
        print(f"[{asset_a}/{asset_b}] {line.strip()}")
    print(f"[{asset_a}/{asset_b}] Process terminated.")


def start(pairs_list):
    processes = []
    for pair in pairs_list:
        asset_a = pair[0]
        asset_b = pair[1]
        proc = subprocess.Popen(
            [sys.executable, "-u", traderScript, asset_a, asset_b],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        processes.append((proc, asset_a, asset_b))
        threading.Thread(target=monitor, args=(proc, asset_a, asset_b), daemon=True).start()
        time.sleep(delay)
    return processes


if __name__ == "__main__":
    procs = start(filtered_pairs)
    print(f"{len(procs)} traders started.")

    try:
        while any(proc.poll() is None for proc, _, _ in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("keyboardinterrupt detected")
        for proc, _, _ in procs:
            proc.terminate()
