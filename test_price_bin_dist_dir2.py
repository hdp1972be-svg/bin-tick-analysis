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
    streaks_data = []  # (streak_len, delta, start_datetime)

    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None
    streak_start_time = None

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Column indices
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

    # Statistics per CSV
    first_ticks_arr = np.array(first_ticks)
    print(f"\n=== Eerste ticks van alle streaks in {os.path.basename(file_path)} ===")
    print(f"Count: {len(first_ticks_arr)}, Min: {first_ticks_arr.min():.2f}, Max: {first_ticks_arr.max():.2f}, "
          f"Mean: {first_ticks_arr.mean():.4f}, Median: {np.median(first_ticks_arr):.4f}, "
          f"P90: {np.percentile(first_ticks_arr, 90):.2f}")

    # Top 5% streaks
    streaks_data_sorted = sorted(streaks_data, key=lambda x: x[0], reverse=True)
    top_5pct_count = max(1, int(0.05 * len(streaks_data_sorted)))
    top_streaks = streaks_data_sorted[:top_5pct_count]

    print(f"\n=== Top 5% langste streaks in {os.path.basename(file_path)} ===")
    for s_len, delta, dt in top_streaks:
        weekday = dt.strftime("%A")
        print(f"Len: {s_len}, Delta: {delta:.2f}, Start: {dt}, Weekday: {weekday}")

    return streaks_data  # return for combined analysis

def analyze_dir_combined(directory):
    combined_streaks = []

    for f in os.listdir(directory):
        if f.lower().endswith(".csv"):
            file_path = os.path.join(directory, f)
            streaks_data = analyze_csv(file_path)
            combined_streaks.extend(streaks_data)

    # Combined top 5% streaks
    combined_sorted = sorted(combined_streaks, key=lambda x: x[0], reverse=True)
    top_5pct_count = max(1, int(0.05 * len(combined_sorted)))
    top_combined = combined_sorted[:top_5pct_count]

    print("\n=== Top 5% langste streaks COMBINED van alle CSV's ===")
    for s_len, delta, dt in top_combined:
        weekday = dt.strftime("%A")
        print(f"Len: {s_len}, Delta: {delta:.2f}, Start: {dt}, Weekday: {weekday}")

if __name__ == "__main__":
    demo_dir = "trading_cloud/tick-data/BTCUSD/csv"
    analyze_dir_combined(demo_dir)
