#!/usr/local/bin/python3.11
import csv
import numpy as np

UP = 1
DOWN = 2

def read_csv(input_path):

    # Dynamische dicts voor bins
    bins = {}           # total streak counts
    bins_up = {}        # streaks ending UP
    bins_down = {}      # streaks ending DOWN
    price_moves = {}    # streak_length -> list of price deltas

    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None

    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            price = float(row[2])

            if prev_price is None:
                prev_price = price
                continue

            # Direction logic
            if price > prev_price:
                direction = UP
            elif price < prev_price:
                direction = DOWN
            else:
                # prijs gelijk → telt niet als hit
                prev_price = price
                continue

            # First move of new streak → set start
            if counter == 0:
                streak_start_price = prev_price

            counter += 1

            # On direction change → streak is complete
            if prev_direction is not None and direction != prev_direction:
                streak_len = counter
                delta = abs(prev_price - streak_start_price)

                # Register streak
                bins[streak_len] = bins.get(streak_len, 0) + 1
                price_moves.setdefault(streak_len, []).append(delta)

                if prev_direction == UP:
                    bins_up[streak_len] = bins_up.get(streak_len, 0) + 1
                else:
                    bins_down[streak_len] = bins_down.get(streak_len, 0) + 1

                # reset
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

        # change this to be printed in 4 cols

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

demo_tick_file = "trading_cloud/tick-data/BTCUSD/csv/BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv"
read_csv(demo_tick_file)
