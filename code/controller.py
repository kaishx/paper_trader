import subprocess
import sys
import time
import threading
import json
import os

PAIRS_CONFIG_FILE = "pairs.json"

if not os.path.exists(PAIRS_CONFIG_FILE):
    raise FileNotFoundError(f"!!! {PAIRS_CONFIG_FILE} not found. please create/import it !!!")

with open(PAIRS_CONFIG_FILE, "r") as f:
    pairs = json.load(f)

print(f"loaded {len(pairs)} pairs from config.")

traderScript = "trader.py"  # make sure the name is correct and debugs are not activated
delay = 10   # seconds between launching each trader

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
            encoding="utf-8",  # explicitly force utf8
            errors="replace"  # prevnt crashes on some characters so the thing keeps running properly
        )
        processes.append((proc, asset_a, asset_b))

        threading.Thread(target=monitor, args=(proc, asset_a, asset_b), daemon=True).start()

        time.sleep(delay)

    return processes


if __name__ == "__main__":
    procs = start(pairs)
    print(f"{len(procs)} traders started.")

    try:
        while any(proc.poll() is None for proc, _, _ in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("keyboardinterrupt detected")
        for proc, _, _ in procs:
            proc.terminate()

