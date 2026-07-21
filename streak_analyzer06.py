#!/usr/local/bin/python3.11
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
    price_sequence = []

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
path = "./testdata/BTCUSD-testdata.csv"
bins, price_moves, ticks, total_streaks, interarrival, streak_sequence, price_sequence = read_csv(path)
streak_array = np.array(streak_sequence)
prices = np.array(price_sequence)
interarrival = np.array(interarrival)
timestamps = np.arange(len(prices))

# --- Top 5 cumulatieve streaks ---
cumulative_lengths = []
start_idxs = []
i = 0
while i < len(streak_array):
    cum_len = streak_array[i]
    start_idx = i
    cumulative_lengths.append(cum_len)
    start_idxs.append(start_idx)
    i += 1

cumulative_lengths = np.array(cumulative_lengths)
start_idxs = np.array(start_idxs)
top5_idx = start_idxs[np.argsort(cumulative_lengths)[-5:]]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,7), sharex=True)

# Scatter van echte prijzen
ax1.scatter(timestamps, prices, s=10, color='blue')
ax1.set_ylabel("Ask price")
ax1.set_title("Ask price scatter met top 5 cumulatieve streaks")

# Horizontale balken en annotaties voor top 5 streaks
max_streak = streak_array.max()
for idx in top5_idx:
    streak_len = streak_array[idx]
    start_price_val = prices[idx]
    end_idx = idx + streak_len - 1 if idx + streak_len - 1 < len(prices) else len(prices)-1
    end_price_val = prices[end_idx]
    delta_price = end_price_val - start_price_val

    # Kleurgradatie: groen voor stijgende, rood voor dalende, alpha gebaseerd op streak lengte
    color = (0, 1, 0, 0.3 + 0.7*(streak_len/max_streak)) if delta_price >= 0 else (1, 0, 0, 0.3 + 0.7*(streak_len/max_streak))
    ax1.hlines(y=start_price_val + delta_price/2, xmin=idx, xmax=end_idx, color=color, linewidth=6)

    # Annotatie met leader line
    ax1.annotate(f"{delta_price:+.2f}", xy=(idx + streak_len/2, start_price_val + delta_price/2),
                 xytext=(idx + streak_len/2, start_price_val + delta_price/2 + 1),
                 arrowprops=dict(arrowstyle='-|>', color='black'), ha='center', fontsize=9, color='black')

# Bar plot van interarrival times
ax2.bar(timestamps[1:], interarrival, color='gray')
ax2.set_ylabel("Delta tijd (s)")
ax2.set_xlabel("Tick index")
ax2.set_title("Interarrival times tussen ticks")

plt.tight_layout()
plt.show()
