
#!/usr/bin/env python3
import csv
import numpy as np
import os
from datetime import datetime

UP = 1
DOWN = 2


def top_pct_slice(data, pct):
    n = max(1, int(len(data) * pct / 100))
    return data[:n], n


def analyze_csv(file_path):
    streaks = []  # (len, delta_first, delta_total, start_dt)

    prev_price = None
    prev_direction = None

    streak_len = 0
    streak_start_price = None
    streak_start_time = None
    first_tick_delta = None
    last_price = None

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
                continue

            # direction
            if price > prev_price:
                direction = UP
            elif price < prev_price:
                direction = DOWN
            else:
                prev_price = price
                continue

            # start of new streak
            if streak_len == 0:
                streak_start_price = prev_price
                streak_start_time = dt
                first_tick_delta = abs(price - prev_price)
                streak_len = 1
            else:
                streak_len += 1

            # streak ends
            if prev_direction is not None and direction != prev_direction:
                delta_total = abs(prev_price - streak_start_price)

                streaks.append((
                    streak_len,
                    first_tick_delta,
                    delta_total,
                    streak_start_time
                ))

                # reset but current tick starts new streak
                streak_start_price = prev_price
                streak_start_time = dt
                first_tick_delta = abs(price - prev_price)
                streak_len = 1

            prev_direction = direction
            prev_price = price
            last_price = price

    return streaks


def analyze_dir_combined(directory, top_pct=5):
    combined = []

    for fname in os.listdir(directory):
        if fname.lower().endswith(".csv"):
            combined.extend(analyze_csv(os.path.join(directory, fname)))

    total_streaks = len(combined)

    # occurrence per streak length
    bins = {}
    for s_len, *_ in combined:
        bins[s_len] = bins.get(s_len, 0) + 1

    # ---------- SORTS ----------
    sort_len_delta = sorted(combined, key=lambda x: (x[0], x[2]), reverse=True)
    sort_delta_only = sorted(combined, key=lambda x: x[2], reverse=True)
    sort_delta_len = sorted(combined, key=lambda x: (x[2], x[0]), reverse=True)

    # ---------- PRINT HELPERS ----------
    def print_block(title, data):
        top, n = top_pct_slice(data, top_pct)
        print(f"\n=== {title} ===")
        print(f"Aantal: {n} / {total_streaks}\n")

        for s_len, d_first, d_total, dt in top:
            weekday = dt.strftime("%A")
            occ = bins[s_len] / total_streaks * 100
            energy = d_total / s_len
            print(
                f"Len: {s_len:>3}, "
                f"Δ_first: {d_first:>7.2f}, "
                f"Δ_total: {d_total:>8.2f}, "
                f"Δ/tick: {energy:>6.2f}, "
                f"Start: {dt}, "
                f"Weekday: {weekday}, "
                f"Occurrence: {occ:>5.2f}%"
            )

    # ---------- OUTPUT ----------
    print_block(
        f"Top {top_pct}% | Lengte ↓ daarna Delta_total ↓",
        sort_len_delta
    )

    print_block(
        f"Top {top_pct}% | Grootste Delta_total (ongeacht lengte)",
        sort_delta_only
    )

    print_block(
        f"Top {top_pct}% | Delta_total ↓ daarna Lengte ↓",
        sort_delta_len
    )


if __name__ == "__main__":
    data_dir = "trading_cloud/tick-data/BTCUSD/csv"
    analyze_dir_combined(data_dir, top_pct=5)  # wijzig naar 1, 2, 10, ...
