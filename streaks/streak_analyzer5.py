import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

UP = 1
DOWN = -1

# --- Data loader ---
def read_csv(path):
    price_moves = {}
    bins = {}
    ticks = 0
    interarrival = []
    prev_ts = None
    prev_price = None
    prev_dir = None
    streak_len = 0
    start_price = None
    streak_sequence = []

    price_sequence = []  # echte prijzen in volgorde

    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ticks += 1
            ts_str = row[0]
            dt = datetime.strptime(ts_str[:-5], "%Y-%m-%d %H:%M:%S.%f")
            ts = dt.timestamp()

            if prev_ts is not None:
                interarrival.append(ts - prev_ts)
            prev_ts = ts

            price = float(row[2])
            price_sequence.append(price)

            if prev_price is None:
                prev_price = price
                continue

            d = UP if price > prev_price else DOWN if price < prev_price else None
            if d is None:
                prev_price = price
                continue

            if streak_len == 0:
                start_price = prev_price
            streak_len += 1

            if prev_dir is not None and d != prev_dir:
                delta = abs(prev_price - start_price)
                price_moves.setdefault(streak_len, []).append(delta)
                bins[streak_len] = bins.get(streak_len, 0) + 1
                streak_sequence.append(streak_len)
                streak_len = 1
                start_price = prev_price

            prev_dir = d
            prev_price = price

    if streak_len > 0:
        delta = abs(price - start_price)
        price_moves.setdefault(streak_len, []).append(delta)
        bins[streak_len] = bins.get(streak_len, 0) + 1
        streak_sequence.append(streak_len)

    total_streaks = sum(bins.values())
    return bins, price_moves, ticks, total_streaks, interarrival, streak_sequence, price_sequence


# --- Lees CSV ---

path = "trading_cloud/tick-data/BTCUSD/csv/BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv"
#path = "data.csv"  # vervang door jouw CSV pad
bins, price_moves, ticks, total_streaks, interarrival, streak_sequence, price_sequence = read_csv(path)
streak_array = np.array(streak_sequence)
prices = np.array(price_sequence)
interarrival = np.array(interarrival)
timestamps = np.arange(len(prices))


# --- Bereken cumulatieve streaks voor annotaties ---
cumulative_streaks = []
idxs = []
i = 0
while i < len(streak_array):
    cum = streak_array[i]
    j = i + 1
    while j < len(streak_array) and j < i + 1:  # optioneel: alleen de huidige streak
        cum += streak_array[j]
        j += 1
    cumulative_streaks.append(cum)
    idxs.append(i)
    i += 1

cumulative_streaks = np.array(cumulative_streaks)
idxs = np.array(idxs)

# selecteer top 5 cumulatieve streaks
top5_indices = idxs[np.argsort(cumulative_streaks)[-5:]]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,7), sharex=True)

# Scatter plot van echte prijzen
ax1.scatter(timestamps, prices, s=10, color='blue')
ax1.set_ylabel("Ask price")
ax1.set_title("Ask price scatter met top 5 cumulatieve streaks")

# Markeer top 5 cumulatieve streaks
for idx in top5_indices:
    ax1.axvline(idx, color='red', linestyle='--')
    ax1.text(idx, prices[idx]+0.5, f"Streak {streak_array[idx]}", color='red', rotation=90,
             verticalalignment='bottom', fontsize=8)

# Bar plot van interarrival times
ax2.bar(timestamps[1:], interarrival, color='gray')
ax2.set_ylabel("Delta tijd (s)")
ax2.set_xlabel("Tick index")
ax2.set_title("Interarrival times tussen ticks")

plt.tight_layout()
plt.show()
