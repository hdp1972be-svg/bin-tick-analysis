
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
    streaks_data = []  # (streak_len, delta_first_tick, delta_total, start_datetime)

    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None
    streak_start_time = None

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

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

            # streak break
            if prev_direction is not None and direction != prev_direction:
                delta_total = abs(prev_price - streak_start_price)  # totale delta van de streak
                delta_first_tick = abs(streak_start_price - (streak_start_price if counter == 1 else streak_start_price))  # optie: kan 0 of eerste tick delta zijn
                streak_len = counter

                bins[streak_len] = bins.get(streak_len, 0) + 1
                price_moves.setdefault(streak_len, []).append(delta_total)

                first_ticks.append(delta_first_tick)
                streaks_data.append((streak_len, delta_first_tick, delta_total, streak_start_time))

                # reset streak
                counter = 1
                streak_start_price = prev_price
                streak_start_time = dt

            prev_direction = direction
            prev_price = price

    total_streaks = sum(bins.values())

    first_ticks_arr = np.array(first_ticks)
    print(f"\n=== Eerste ticks van alle streaks in {os.path.basename(file_path)} ===")
    print(f"Count: {len(first_ticks_arr)}, Min: {first_ticks_arr.min():.2f}, Max: {first_ticks_arr.max():.2f}, "
          f"Mean: {first_ticks_arr.mean():.4f}, Median: {np.median(first_ticks_arr):.4f}, "
          f"P90: {np.percentile(first_ticks_arr, 90):.2f}")

    return streaks_data, bins, total_streaks

def analyze_dir_combined(directory, top_pct=5):
    combined_streaks = []
    combined_bins = {}
    total_combined = 0

    for f in os.listdir(directory):
        if f.lower().endswith(".csv"):
            file_path = os.path.join(directory, f)
            streaks_data, bins, total_streaks = analyze_csv(file_path)
            combined_streaks.extend(streaks_data)

            for k, v in bins.items():
                combined_bins[k] = combined_bins.get(k, 0) + v
            total_combined += total_streaks

    combined_sorted = sorted(combined_streaks, key=lambda x: x[0], reverse=True)
    top_count = max(1, int(top_pct/100 * len(combined_sorted)))
    top_combined = combined_sorted[:top_count]

    print(f"\n=== Top {top_pct}% langste streaks COMBINED van alle CSV's ===")
    print(f"Aantal streaks in top {top_pct}%: {top_count} / {len(combined_sorted)} totale streaks")

    for s_len, delta_first, delta_total, dt in top_combined:
        weekday = dt.strftime("%A")
        pct = combined_bins[s_len] / total_combined * 100
        print(f"Len: {s_len}, Delta_first: {delta_first:.2f}, Delta_total: {delta_total:.2f}, Start: {dt}, Weekday: {weekday}, Occurrence: {pct:.2f}%")

if __name__ == "__main__":
    demo_dir = "trading_cloud/tick-data/BTCUSD/csv"
    analyze_dir_combined(demo_dir, top_pct=5)
