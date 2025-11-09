import subprocess
import sys
import time
import threading

# pairs down here. template for how to write it: ("XXX", "YYY")
TRADER_PAIRS = [
    ("UDR", "AVB"),
    ("GOOG", "GOOGL"),
    ("ORCL", "IBM"),
    ("SLB", "HAL"),
    ("SPHY", "USHY")
    ("AOS", "MAS")
    ]

TRADER_SCRIPT = "trader.py"  # maincontroller points to trader
LOG_INTERVAL = 20  # seconds between printing logs
LAUNCH_DELAY = 10   # seconds between launching each trader


def monitor(proc, asset_a, asset_b):
    for line in iter(proc.stdout.readline, ''):
        print(f"[{asset_a}/{asset_b}] {line.strip()}")
    print(f"[{asset_a}/{asset_b}] Process terminated.")

def start(pairs_list):
    processes = []

    for asset_a, asset_b in pairs_list:
        print(f"Starting trader for pair: {asset_a}/{asset_b}")
        proc = subprocess.Popen(
            [sys.executable, "-u", TRADER_SCRIPT, asset_a, asset_b],  # <- "-u" forces unbuffered stdout/stderr
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            encoding="utf-8",  # explicitly force utf8
            errors="replace"  # prevnt crashes on some characters so the thing keeps running properly
        )
        processes.append((proc, asset_a, asset_b))

        # this starts a monitor
        threading.Thread(target=monitor, args=(proc, asset_a, asset_b), daemon=True).start()

        time.sleep(LAUNCH_DELAY)

    return processes


if __name__ == "__main__":
    procs = start(TRADER_PAIRS)
    print(f"{len(procs)} traders started.")

    try:
        # keep the main alive while the subs are running
        while any(proc.poll() is None for proc, _, _ in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Terminating all traders...")
        for proc, _, _ in procs:
            proc.terminate()
