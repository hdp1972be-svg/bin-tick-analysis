
#!/usr/bin/env python3
import csv
import numpy as np
import os
from datetime import datetime

UP = 1
DOWN = 2

def analyze_csv(file_path):
    bins = {}
    price_moves = {}
    first_ticks = []
    streaks_data = []  # for longest 5% streaks: (delta, start_datetime)

    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None
    streak_start_time = None

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find column indices
        ts_idx = header.index('timestamp') if 'timestamp' in header else 0
        price_idx = header.index('price') if 'price' in header else 2

        for row in reader:
            price = float(row[price_idx])
            ts = row[ts_idx]

            dt = datetime.fromisoformat(ts)

            if prev_price is None:
                prev_price = price
                prev_direction = None
                continue

            # direction
            if price > prev_price:
                direction = UP
            elif price < prev_price:
                direction = DOWN
            else:
                prev_price = price
                continue

            if counter == 0:
                streak_start_price = prev_price
                streak_start_time = dt

            counter += 1

            if prev_direction is not None and direction != prev_direction:
                delta = abs(prev_price - streak_start_price)
                streak_len = counter

                bins[streak_len] = bins.get(streak_len, 0) + 1
                price_moves.setdefault(streak_len, []).append(delta)

                first_ticks.append(delta)
                streaks_data.append((streak_len, delta, streak_start_time))

                # reset streak
                counter = 1
                streak_start_price = prev_price
                streak_start_time = dt

            prev_direction = direction
            prev_price = price

    # Statistics
    first_ticks_arr = np.array(first_ticks)
    print(f"\n=== Eerste ticks van alle streaks in {os.path.basename(file_path)} ===")
    print(f"Count: {len(first_ticks_arr)}, Min: {first_ticks_arr.min():.2f}, Max: {first_ticks_arr.max():.2f}, "
          f"Mean: {first_ticks_arr.mean():.4f}, Median: {np.median(first_ticks_arr):.4f}, "
          f"P90: {np.percentile(first_ticks_arr, 90):.2f}")

    # Longest 5% streaks
    streaks_data_sorted = sorted(streaks_data, key=lambda x: x[0], reverse=True)
    top_5pct_count = max(1, int(0.05 * len(streaks_data_sorted)))
    top_streaks = streaks_data_sorted[:top_5pct_count]

    print(f"\n=== Top 5% langste streaks in {os.path.basename(file_path)} ===")
    for s_len, delta, dt in top_streaks:
        weekday = dt.strftime("%A")
        print(f"Len: {s_len}, Delta: {delta:.2f}, Start: {dt}, Weekday: {weekday}")

def analyze_dir(directory):
    for f in os.listdir(directory):
        if f.lower().endswith(".csv"):
            analyze_csv(os.path.join(directory, f))

if __name__ == "__main__":
    demo_dir = "./testdata"
    analyze_dir(demo_dir)
