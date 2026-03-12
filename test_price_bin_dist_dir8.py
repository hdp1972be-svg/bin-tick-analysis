
#!/usr/bin/env python3
import csv
import numpy as np
import os
from datetime import datetime

UP = 1
DOWN = 2


def analyze_csv(file_path):
    bins = {}
    streaks_data = []  # (streak_len, delta_first, delta_total, start_datetime)

    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None
    streak_start_time = None
    first_tick_price = None

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        ts_idx = header.index('timestamp') if 'timestamp' in header else 0
        price_idx = header.index('price') if 'price' in header else 2

        for row in reader:
            price = float(row[price_idx])
            dt = datetime.fromisoformat(row[ts_idx])

            if prev_price is None:
                prev_price = price
                prev_direction = None
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
                streak_start_time = dt
                first_tick_price = price

            counter += 1

            if prev_direction is not None and direction != prev_direction:
                delta_total = abs(prev_price - streak_start_price)
                delta_first = abs(first_tick_price - streak_start_price)
                streak_len = counter

                bins[streak_len] = bins.get(streak_len, 0) + 1
                streaks_data.append(
                    (streak_len, delta_first, delta_total, streak_start_time)
                )

                counter = 1
                streak_start_price = prev_price
                streak_start_time = dt
                first_tick_price = price

            prev_direction = direction
            prev_price = price

    total_streaks = sum(bins.values())

    delta_first_arr = np.array([s[1] for s in streaks_data])
    print(f"\n=== Eerste ticks van alle streaks in {os.path.basename(file_path)} ===")
    print(
        f"Count: {len(delta_first_arr)}, "
        f"Min: {delta_first_arr.min():.2f}, "
        f"Max: {delta_first_arr.max():.2f}, "
        f"Mean: {delta_first_arr.mean():.4f}, "
        f"Median: {np.median(delta_first_arr):.4f}, "
        f"P90: {np.percentile(delta_first_arr, 90):.2f}"
    )

    return streaks_data, bins, total_streaks


def analyze_dir_combined(directory, top_pct=5):
    combined = []
    combined_bins = {}
    total_combined = 0

    for f in os.listdir(directory):
        if f.lower().endswith(".csv"):
            streaks, bins, total = analyze_csv(os.path.join(directory, f))
            combined.extend(streaks)
            total_combined += total
            for k, v in bins.items():
                combined_bins[k] = combined_bins.get(k, 0) + v

    def top_n(data):
        return max(1, int(len(data) * top_pct / 100))

    def print_block(title, data):
        print(f"\n=== {title} ===")
        n = top_n(data)
        print(f"Aantal: {n} / {len(combined)}\n")

        for s_len, d_first, d_total, dt in data[:n]:
            weekday = dt.strftime("%A")
            occ = combined_bins[s_len] / total_combined * 100
            dpt = d_total / s_len
            print(
                f"Len: {s_len:>3}, "
                f"Δ_first: {d_first:>7.2f}, "
                f"Δ_total: {d_total:>8.2f}, "
                f"Δ/tick: {dpt:>6.2f}, "
                f"Start: {dt}, "
                f"Weekday: {weekday}, "
                f"Occurrence: {occ:>6.3f}%"
            )

    # 🔹 1. lengte ↓ daarna delta_total ↓
    sort_len_delta = sorted(
        combined, key=lambda x: (x[0], x[2]), reverse=True
    )

    # 🔹 2. puur grootste delta_total
    sort_delta = sorted(
        combined, key=lambda x: x[2], reverse=True
    )

    # 🔹 3. delta_total ↓ daarna lengte ↓
    sort_delta_len = sorted(
        combined, key=lambda x: (x[2], x[0]), reverse=True
    )

    print_block(
        f"Top {top_pct}% | Streaklengte ↓ daarna Delta_total ↓",
        sort_len_delta
    )

    print_block(
        f"Top {top_pct}% | Grootste Delta_total",
        sort_delta
    )

    print_block(
        f"Top {top_pct}% | Delta_total ↓ daarna Streaklengte ↓",
        sort_delta_len
    )


if __name__ == "__main__":
    demo_dir = "trading_cloud/tick-data/BTCUSD/csv"
    analyze_dir_combined(demo_dir, top_pct=1)  # 1, 5, 10 ...
