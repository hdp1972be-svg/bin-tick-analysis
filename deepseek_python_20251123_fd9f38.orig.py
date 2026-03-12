#!/usr/local/bin/python3.11
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    return bins, price_moves

def plot_streak_stats(bins, price_moves):
    """Plot de streak statistieken"""
    
    # Filter streaks met voldoende data (minimaal 100 samples)
    valid_streaks = [s for s in sorted(bins.keys()) if len(price_moves.get(s, [])) >= 20]
    
    if not valid_streaks:
        print("Niet genoeg data om te plotten")
        return
    
    # Maak subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Streak frequentie
    streak_lengths = [s-1 for s in valid_streaks]
    frequencies = [bins[s] for s in valid_streaks]
    
    ax1.bar(streak_lengths, frequencies, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Streak Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Streak Length Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Percentiel evolutie over streaks
    percentiles = [10, 25, 50, 75, 90]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for p, color in zip(percentiles, colors):
        p_values = [np.percentile(price_moves[s], p) for s in valid_streaks]
        ax2.plot(streak_lengths, p_values, 'o-', label=f'p{p}', color=color, linewidth=2)
    
    ax2.set_xlabel('Streak Length')
    ax2.set_ylabel('Price Delta')
    ax2.set_title('Percentile Evolution by Streak Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Boxplot voor eerste 5 streaks
    boxplot_data = []
    boxplot_labels = []
    
    for s in valid_streaks[:5]:  # Eerste 5 streaks
        if len(price_moves[s]) > 0:
            boxplot_data.append(price_moves[s])
            boxplot_labels.append(f'Streak {s-1}')
    
    ax3.boxplot(boxplot_data, labels=boxplot_labels)
    ax3.set_ylabel('Price Delta')
    ax3.set_title('Price Delta Distribution by Streak Length')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean en Median vergelijking
    means = [np.mean(price_moves[s]) for s in valid_streaks]
    medians = [np.median(price_moves[s]) for s in valid_streaks]
    
    ax4.plot(streak_lengths, means, 'o-', label='Mean', linewidth=2, color='red')
    ax4.plot(streak_lengths, medians, 'o-', label='Median', linewidth=2, color='blue')
    ax4.set_xlabel('Streak Length')
    ax4.set_ylabel('Price Delta')
    ax4.set_title('Mean vs Median by Streak Length')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_percentile_profile(price_moves, streak_length=1, p_start=90, p_end=100, step=1):
    """Maak een gedetailleerd percentielprofiel voor een specifieke streak"""
    
    if streak_length not in price_moves or len(price_moves[streak_length]) == 0:
        print(f"Geen data voor streak {streak_length}")
        return None
    
    data = price_moves[streak_length]
    percentiles = range(p_start, p_end + 1, step)
    
    profile = {}
    print(f"\n=== Percentile Profile for Streak {streak_length-1} (p{p_start}-p{p_end}) ===")
    print(f"Sample size: {len(data)}")
    print("-" * 50)
    
    for p in percentiles:
        value = np.percentile(data, p)
        profile[p] = value
        print(f"p{p:2d}: {value:.8f}")
    
    # Plot het percentielprofiel
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(list(profile.keys()), list(profile.values()), 'o-', linewidth=2, markersize=6)
    plt.xlabel('Percentile')
    plt.ylabel('Price Delta')
    plt.title(f'Percentile Profile: Streak {streak_length-1}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Histogram van de data met percentiel markers
    plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(profile[90], color='red', linestyle='--', label='p90', linewidth=2)
    plt.axvline(profile[95], color='orange', linestyle='--', label='p95', linewidth=2)
    plt.axvline(profile[99], color='green', linestyle='--', label='p99', linewidth=2)
    plt.xlabel('Price Delta')
    plt.ylabel('Frequency')
    plt.title(f'Distribution with Percentiles: Streak {streak_length-1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return profile

# Hoofdprogramma
demo_tick_file = "trading_cloud/tick-data/BTCUSD/csv/BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv"
bins, price_moves = read_csv(demo_tick_file)

# Plot de basis statistieken
plot_streak_stats(bins, price_moves)

# Maak gedetailleerde percentielprofielen voor belangrijke streaks
for streak in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    if streak in price_moves and len(price_moves[streak]) >= 10:  # Minimaal 10 samples
        create_percentile_profile(price_moves, streak_length=streak, p_start=90, p_end=100, step=1)
