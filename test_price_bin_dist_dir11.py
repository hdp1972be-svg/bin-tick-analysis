
#!/usr/bin/env python3
import csv
import numpy as np
import os
from datetime import datetime

UP = 1
DOWN = 2

def analyze_csv(file_path):
    bins = {}
    streaks_data = []  # (streak_len, delta_first, delta_total, start_datetime, direction)

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
            ts = row[ts_idx]
            dt = datetime.fromisoformat(ts)

            if prev_price is None:
                prev_price = price
                prev_direction = None
                continue

            # bepaal richting
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

            # streak einde
            if prev_direction is not None and direction != prev_direction:
                delta_total = abs(prev_price - streak_start_price)
                delta_first = abs(first_tick_price - streak_start_price)
                streak_len = counter

                bins[streak_len] = bins.get(streak_len, 0) + 1
                streaks_data.append((streak_len, delta_first, delta_total, streak_start_time, prev_direction))

                # reset streak
                counter = 1
                streak_start_price = prev_price
                streak_start_time = dt
                first_tick_price = price

            prev_direction = direction
            prev_price = price

    total_streaks = sum(bins.values())

    # eerste tick statistieken
    delta_first_arr = np.array([s[1] for s in streaks_data])
    print(f"\n=== Eerste ticks van alle streaks in {os.path.basename(file_path)} ===")
    print(f"Count: {len(delta_first_arr)}, Min: {delta_first_arr.min():.2f}, Max: {delta_first_arr.max():.2f}, "
          f"Mean: {delta_first_arr.mean():.4f}, Median: {np.median(delta_first_arr):.4f}, "
          f"P90: {np.percentile(delta_first_arr, 90):.2f}")

    return streaks_data, bins, total_streaks


def analyze_dir_combined(directory, top_pct=5, max_rebound_len=3):
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

    # --- Top streaks gesorteerd op lengte ---
    combined_sorted = sorted(combined_streaks, key=lambda x: x[0], reverse=True)
    top_count = max(1, int(top_pct / 100 * len(combined_sorted)))
    top_combined = combined_sorted[:top_count]

    print(f"\n=== Top {top_pct}% langste streaks COMBINED van alle CSV's ===")
    print(f"Aantal streaks in top {top_pct}%: {top_count} / {len(combined_sorted)} totale streaks\n")

    for s_len, delta_first, delta_total, dt, direction in top_combined:
        weekday = dt.strftime("%A")
        pct = combined_bins[s_len] / total_combined * 100
        delta_per_tick = delta_total / s_len
        print(f"Len: {s_len:3}, Δ_first: {delta_first:7.2f}, Δ_total: {delta_total:7.2f}, "
              f"Δ/tick: {delta_per_tick:7.2f}, Start: {dt}, Weekday: {weekday}, Occurrence: {pct:6.3f}%")

    # --- Rebound detectie ---
    rebounds = []
    for i in range(len(combined_streaks) - 2):
        A, B, C = combined_streaks[i], combined_streaks[i + 1], combined_streaks[i + 2]

        # criteria: B korte tegenstreak, C in oorspronkelijke richting A
        if B[0] <= max_rebound_len and B[4] != A[4] and C[4] == A[4]:
            rebound_delta = C[2]
            rebound_len = C[0]
            rebound_pct = rebound_delta / A[2] * 100 if A[2] != 0 else 0
            rebounds.append((A, B, C, rebound_delta, rebound_len, rebound_pct))

    print(f"\n=== Rebounds detectie (tegenstreak ≤ {max_rebound_len}) ===")
    for A, B, C, delta, length, pct in rebounds[:50]:  # print top 50 voor overzicht
        print(f"A_Len:{A[0]}, B_Len:{B[0]}, C_Len:{C[0]}, Δ_C:{delta:.2f}, Δ%_vs_A:{pct:.2f}%, Start:{C[3]}")


if __name__ == "__main__":
    demo_dir = "trading_cloud/tick-data/BTCUSD/csv"
    analyze_dir_combined(demo_dir, top_pct=1, max_rebound_len=3)
