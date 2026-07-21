#!/usr/local/bin/python3.11
import csv
import numpy as np
from scipy import stats

UP = 1
DOWN = 2

def read_csv(input_path):

    # Dictionaries voor streak stats
    bins = {}           # totale streak counts
    bins_up = {}        # streaks eindigend UP
    bins_down = {}      # streaks eindigend DOWN
    price_moves = {}    # streak_length -> list van delta's

    # Voor eerste tick analyse
    first_ticks = []         # alle eerste ticks van streaks
    single_ticks = []        # eerste tick van streaks van lengte 1
    first_of_long_streaks = []  # eerste tick van streaks van lengte >=2

    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None
    streak_first_tick = None

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
                prev_price = price
                continue  # gelijk → skip

            # Nieuwe streak detectie
            if counter == 0:
                streak_start_price = prev_price
                streak_first_tick = price - prev_price  # eerste delta

            counter += 1

            # Richtingsverandering → streak afsluiten
            if prev_direction is not None and direction != prev_direction:
                streak_len = counter - 1  # vorige streak
                delta = abs(prev_price - streak_start_price)

                # Streak stats bijhouden
                bins[streak_len] = bins.get(streak_len, 0) + 1
                price_moves.setdefault(streak_len, []).append(delta)
                first_ticks.append(streak_first_tick)
                if streak_len == 1:
                    single_ticks.append(streak_first_tick)
                else:
                    first_of_long_streaks.append(streak_first_tick)

                # reset voor nieuwe streak
                counter = 1
                streak_start_price = prev_price
                streak_first_tick = price - prev_price

            prev_direction = direction
            prev_price = price

    # Laatste streak verwerken
    if counter > 0:
        streak_len = counter
        delta = abs(prev_price - streak_start_price)
        bins[streak_len] = bins.get(streak_len, 0) + 1
        price_moves.setdefault(streak_len, []).append(delta)
        first_ticks.append(streak_first_tick)
        if streak_len == 1:
            single_ticks.append(streak_first_tick)
        else:
            first_of_long_streaks.append(streak_first_tick)

    return bins, price_moves, first_ticks, single_ticks, first_of_long_streaks


def summarize(arr, name):
    arr = np.array(arr)
    print(f"{name} | Count {len(arr)} | Min {arr.min():.8f} | Max {arr.max():.8f} | "
          f"Mean {arr.mean():.8f} | Median {np.median(arr):.8f} | "
          f"P10 {np.percentile(arr,10):.8f} | P25 {np.percentile(arr,25):.8f} | "
          f"P50 {np.percentile(arr,50):.8f} | P75 {np.percentile(arr,75):.8f} | "
          f"P90 {np.percentile(arr,90):.8f}")


def print_stats(bins, price_moves, first_ticks, single_ticks, first_of_long_streaks):
    total = sum(bins.values())
    print("\n=== Traditionele streak stats ===")
    for s in sorted(bins.keys()):
        arr = np.array(price_moves.get(s, []))
        print(f"Streak {s}: {bins[s]} hits ({bins[s]/total*100:.2f}%)")
        print(f"  min Δ: {arr.min():.8f}")
        print(f"  max Δ: {arr.max():.8f}")
        print(f"  mean Δ: {arr.mean():.8f}")
        print(f"  median Δ: {np.median(arr):.8f}")
        print(f"  p10: {np.percentile(arr,10):.8f}")
        print(f"  p25: {np.percentile(arr,25):.8f}")
        print(f"  p50: {np.percentile(arr,50):.8f}")
        print(f"  p75: {np.percentile(arr,75):.8f}")
        print(f"  p90: {np.percentile(arr,90):.8f}\n")

    print("\n=== Eerste ticks van alle streaks ===")
    summarize(first_ticks, "Eerste ticks")
    print("\n=== Single streaks vs eerste van lange streaks (≥2) ===")
    summarize(single_ticks, "Single streaks")
    summarize(first_of_long_streaks, "Eerste van lange streaks")

    # Statistische test
    if len(single_ticks) > 0 and len(first_of_long_streaks) > 0:
        stat, p = stats.mannwhitneyu(single_ticks, first_of_long_streaks, alternative='two-sided')
        print("\n=== Statistische test (Mann-Whitney U) ===")
        print(f"Mann-Whitney U statistic: {stat:.2f}, p-value: {p:.4f}")
        if p < 0.05:
            print("=> Eerste ticks van lange streaks verschillen significant van single ticks (p<0.05)")
        else:
            print("=> Geen significant verschil (p≥0.05)")


if __name__ == "__main__":
    demo_tick_file = "./testdata/BTCUSD-testdata.csv"
    bins, price_moves, first_ticks, single_ticks, first_of_long_streaks = read_csv(demo_tick_file)
    print_stats(bins, price_moves, first_ticks, single_ticks, first_of_long_streaks)
