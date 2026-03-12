
#!/usr/local/bin/python3.11
import os
import requests
import random
import datetime
import csv
import numpy as np

# ==== Jouw analyse constants ====
UP = 1
DOWN = 2

# ==== Analyse functie (jouw code) ====
def read_csv(input_path):
    bins = {}
    bins_up = {}
    bins_down = {}
    price_moves = {}
    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None

    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            price = float(row[2])
            if prev_price is None:
                prev_price = price
                continue
            if price > prev_price:
                direction = UP
            elif price < prev_price:
                direction = DOWN
            else:
                prev_price = price
                continue
            if counter == 0:
                streak_start_price = prev_price
            counter += 1
            if prev_direction is not None and direction != prev_direction:
                streak_len = counter
                delta = abs(prev_price - streak_start_price)
                bins[streak_len] = bins.get(streak_len, 0) + 1
                price_moves.setdefault(streak_len, []).append(delta)
                if prev_direction == UP:
                    bins_up[streak_len] = bins_up.get(streak_len, 0) + 1
                else:
                    bins_down[streak_len] = bins_down.get(streak_len, 0) + 1
                counter = 1
                streak_start_price = prev_price
            prev_direction = direction
            prev_price = price

    # Stats printout
    print("\n=== Hit streak stats ===\n")
    total = sum(bins.values())
    print(f"Total hits: {total}\n")
    for s in sorted(bins.keys()):
        arr = np.array(price_moves.get(s, []))
        if len(arr) == 0:
            continue
        print(f"Streak {s-1}: {bins[s]} hits ({bins[s]/total*100:.2f}%)")
        print(f"  min Δ: {arr.min():.8f}")
        print(f"  max Δ: {arr.max():.8f}")
        print(f"  mean Δ: {arr.mean():.8f}")
        print(f"  median Δ: {np.median(arr):.8f}")
        print(f"  p10: {np.percentile(arr, 10):.8f}")
        print(f"  p25: {np.percentile(arr, 25):.8f}")
        print(f"  p50: {np.percentile(arr, 50):.8f}")
        print(f"  p75: {np.percentile(arr, 75):.8f}")
        print(f"  p90: {np.percentile(arr, 90):.8f}")
        print()

# ==== Binance download helper ====
def download_binance_trades(symbol: str, date: datetime.date, save_dir="tick_data"):
    """
    Download CSV tick data for a given symbol and date from Binance.
    symbol: e.g., "BTCUSDT"
    date: datetime.date object
    """
    base_url = "https://data.binance.vision/data/spot/daily/trades/"
    pair_folder = symbol.upper() + "/"
    filename = f"{symbol.upper()}-{date.strftime('%Y-%m-%d')}.csv"
    url = f"{base_url}{pair_folder}{filename}"

    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, filename)

    if os.path.exists(local_path):
        print(f"[INFO] File already exists: {local_path}")
        return local_path

    print(f"[INFO] Downloading {url} ...")
    r = requests.get(url)
    if r.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"[INFO] Saved to {local_path}")
        return local_path
    else:
        print(f"[WARN] Failed to download {url} (status {r.status_code})")
        return None

# ==== Random day generator ====
def random_dates(start_date, end_date, n):
    delta = end_date - start_date
    dates = []
    for _ in range(n):
        random_days = random.randint(0, delta.days)
        dates.append(start_date + datetime.timedelta(days=random_days))
    return dates

# ==== Main workflow ====
if __name__ == "__main__":
    symbol = "BTCUSDT"       # crypto pair
    num_days = 3             # aantal random dagen
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2025, 12, 15)

    for day in random_dates(start, end, num_days):
        csv_file = download_binance_trades(symbol, day)
        if csv_file:
            read_csv(csv_file)
