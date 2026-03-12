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


def entropy_of_distribution(data, bins=50):
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))


def tail_ratio(data):
    return np.percentile(data, 99) / np.median(data)


def tail_slope(data):
    p50 = np.percentile(data, 50)
    return (np.percentile(data, 90) - p50) / p50


def conditional_collapse_probability(data):
    p25 = np.percentile(data, 25)
    return np.mean(np.diff(data) < p25)


def survival_curve(streak_counts):
    max_len = max(streak_counts.keys())
    survival = []
    for k in range(1, max_len + 1):
        survival.append(sum(v for s, v in streak_counts.items() if s >= k))
    survival = np.array(survival)
    return survival / survival[0]

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
    
    for s in valid_streaks[:9]:  # Eerste 5 streaks
        if len(price_moves[s]) > 0:
            boxplot_data.append(price_moves[s])
            boxplot_labels.append(f'Streak {s-1}')

    violins = ax3.violinplot(
        boxplot_data,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Kleuren & stijl
    colors = plt.cm.Set2(np.linspace(0, 1, len(boxplot_data)))
    for body, color in zip(violins['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    # Median markers (expliciet)
    medians = [np.median(d) for d in boxplot_data]
    ax3.scatter(
        range(1, len(medians) + 1),
        medians,
        color='black',
        s=40,
        zorder=3,
        label='median'
    )

    ax3.set_xticks(range(1, len(boxplot_labels) + 1))
    ax3.set_xticklabels(boxplot_labels)
    ax3.set_ylabel('Price Delta')
    ax3.set_title('Price Delta Distribution by Streak Length (Violin)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
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

def plot_streak_stats_extended(bins, price_moves, max_streaks=8):
    valid_streaks = [s for s in sorted(bins) if s <= max_streaks and len(price_moves[s]) >= 20]
    streak_labels = [f'Str {s-1}' for s in valid_streaks]

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 3)

    # === PLOT A: Split violins UP vs DOWN (1,2,8) ===
    ax_v = fig.add_subplot(gs[0, :2])
    data = [price_moves[s] for s in valid_streaks]

    violins = ax_v.violinplot(data, showextrema=False)
    for body in violins['bodies']:
        body.set_alpha(0.6)
        body.set_edgecolor('black')

    # IQR bands + medians + sample size
    for i, d in enumerate(data, start=1):
        q25, q75 = np.percentile(d, [25, 75])
        ax_v.fill_between([i-0.15, i+0.15], q25, q75, alpha=0.3, color='black')
        ax_v.scatter(i, np.median(d), color='black', zorder=3)
        ax_v.text(i, ax_v.get_ylim()[0], f'n={len(d)}', ha='center', va='bottom', fontsize=9)

    ax_v.set_xticks(range(1, len(streak_labels)+1))
    ax_v.set_xticklabels(streak_labels)
    ax_v.set_title('Streak Distributions (Violin + IQR + N)')
    ax_v.grid(alpha=0.3)

    # === PLOT B: Tail ratios (3) ===
    ax_tr = fig.add_subplot(gs[0, 2])
    ratios = [tail_ratio(price_moves[s]) for s in valid_streaks]
    ax_tr.scatter(streak_labels, ratios, s=60)
    ax_tr.set_title('Tail Ratio p99 / p50')
    ax_tr.grid(alpha=0.3)

    # === PLOT C: Tail slope (4) ===
    ax_ts = fig.add_subplot(gs[1, 0])
    slopes = [tail_slope(price_moves[s]) for s in valid_streaks]
    ax_ts.plot(streak_labels, slopes, 'o-')
    ax_ts.set_title('Tail Slope (p90–p50)/p50')
    ax_ts.grid(alpha=0.3)

    # === PLOT D: Entropy (5) ===
    ax_e = fig.add_subplot(gs[1, 1])
    entropies = [entropy_of_distribution(price_moves[s]) for s in valid_streaks]
    ax_e.plot(streak_labels, entropies, 'o-', color='purple')
    ax_e.set_title('Entropy per Streak')
    ax_e.grid(alpha=0.3)

    # === PLOT E: Volatility acceleration (6) ===
    ax_va = fig.add_subplot(gs[1, 2])
    medians = [np.median(price_moves[s]) for s in valid_streaks]
    accel = np.diff(medians) / medians[:-1]
    ax_va.plot(streak_labels[1:], accel, 'o-', color='darkred')
    ax_va.set_title('Volatility Acceleration')
    ax_va.grid(alpha=0.3)

    # === PLOT F: Conditional collapse (7) ===
    ax_cc = fig.add_subplot(gs[2, :])
    collapse = [conditional_collapse_probability(price_moves[s]) for s in valid_streaks]
    ax_cc.plot(streak_labels, collapse, 'o-', color='orange')
    ax_cc.set_title('Conditional Collapse Probability')
    ax_cc.grid(alpha=0.3)

    # === PLOT G: Survival curve (9) ===
    ax_sv = fig.add_subplot(gs[3, 0])
    surv = survival_curve(bins)
    ax_sv.plot(range(1, len(surv)+1), surv)
    ax_sv.set_yscale('log')
    ax_sv.set_title('Streak Survival Curve')
    ax_sv.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

from scipy.stats import skew, kurtosis

def plot_streak_stats_full(bins, price_moves, max_streaks=8):
    """
    Complete streak-analyse:
    - Violins per streak
    - IQR overlay + sample size
    - Tail ratio, tail slope, conditional collapse
    - Entropy, volatility acceleration
    - Survival curve
    - Skew, kurtosis, min/max delta, fraction extreme, MAD
    - Cumulative delta
    """
    
    valid_streaks = [s for s in sorted(bins.keys()) if s <= max_streaks and len(price_moves[s]) >= 20]
    streak_labels = [f'Str {s-1}' for s in valid_streaks]

    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(5, 3)

    # === Violins + IQR + sample size + min/max + MAD + fraction extreme (1,2,4,5,6) ===
    ax_v = fig.add_subplot(gs[0, :2])
    data_list = [price_moves[s] for s in valid_streaks]
    violins = ax_v.violinplot(data_list, showextrema=False)
    for body in violins['bodies']:
        body.set_alpha(0.6)
        body.set_edgecolor('black')

    for i, d in enumerate(data_list, start=1):
        # IQR band
        q25, q75 = np.percentile(d, [25, 75])
        ax_v.fill_between([i-0.15, i+0.15], q25, q75, alpha=0.3, color='black')
        # Median
        median_val = np.median(d)
        ax_v.scatter(i, median_val, color='black', zorder=3)
        # Sample size
        ax_v.text(i, ax_v.get_ylim()[0], f'n={len(d)}', ha='center', va='bottom', fontsize=9)
        # Min/Max
        ax_v.vlines(i, min(d), max(d), color='grey', alpha=0.5, linewidth=1)
        # MAD
        mad_val = np.median(np.abs(d - median_val))
        ax_v.scatter(i, median_val + mad_val, color='blue', marker='^', s=30, label='MAD' if i==1 else "")
        # Fraction extreme (p90+)
        p90 = np.percentile(d, 90)
        frac_ext = np.mean(np.array(d) >= p90)
        ax_v.scatter(i, median_val + 2*mad_val, color='red', marker='x', s=30, label='Frac p90+' if i==1 else "")

    ax_v.set_xticks(range(1, len(streak_labels)+1))
    ax_v.set_xticklabels(streak_labels)
    ax_v.set_title('Streak Violins + IQR + Min/Max + MAD + Fraction Extreme')
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc='upper left')

    # === Tail ratio (3) ===
    ax_tr = fig.add_subplot(gs[0, 2])
    ratios = [tail_ratio(price_moves[s]) for s in valid_streaks]
    ax_tr.scatter(streak_labels, ratios, s=60, color='orange')
    ax_tr.set_title('Tail Ratio (p99/p50)')
    ax_tr.grid(alpha=0.3)

    # === Tail slope (4) ===
    ax_ts = fig.add_subplot(gs[1, 0])
    slopes = [tail_slope(price_moves[s]) for s in valid_streaks]
    ax_ts.plot(streak_labels, slopes, 'o-', color='green')
    ax_ts.set_title('Tail Slope (p90–p50)/p50')
    ax_ts.grid(alpha=0.3)

    # === Entropy (5) ===
    ax_e = fig.add_subplot(gs[1, 1])
    entropies = [entropy_of_distribution(price_moves[s]) for s in valid_streaks]
    ax_e.plot(streak_labels, entropies, 'o-', color='purple')
    ax_e.set_title('Entropy per Streak')
    ax_e.grid(alpha=0.3)

    # === Volatility acceleration (6) ===
    ax_va = fig.add_subplot(gs[1, 2])
    medians = [np.median(price_moves[s]) for s in valid_streaks]
    accel = np.diff(medians) / medians[:-1]
    ax_va.plot(streak_labels[1:], accel, 'o-', color='darkred')
    ax_va.set_title('Volatility Acceleration')
    ax_va.grid(alpha=0.3)

    # === Conditional collapse probability (7) ===
    ax_cc = fig.add_subplot(gs[2, :])
    collapse = [conditional_collapse_probability(price_moves[s]) for s in valid_streaks]
    ax_cc.plot(streak_labels, collapse, 'o-', color='orange')
    ax_cc.set_title('Conditional Collapse Probability')
    ax_cc.grid(alpha=0.3)

    # === Survival curve (9) ===
    ax_sv = fig.add_subplot(gs[3, 0])
    surv = survival_curve(bins)
    ax_sv.plot(range(1, len(surv)+1), surv)
    ax_sv.set_yscale('log')
    ax_sv.set_title('Streak Survival Curve')
    ax_sv.grid(alpha=0.3)

    # === Skewness + Kurtosis (extra) ===
    ax_sk = fig.add_subplot(gs[3, 1])
    skews = [skew(price_moves[s]) for s in valid_streaks]
    kurts = [kurtosis(price_moves[s]) for s in valid_streaks]
    ax_sk.plot(streak_labels, skews, 'o-', label='Skew')
    ax_sk.plot(streak_labels, kurts, 's-', label='Kurtosis')
    ax_sk.set_title('Skewness & Kurtosis per Streak')
    ax_sk.legend()
    ax_sk.grid(alpha=0.3)

    # === Cumulative delta per streak (extra) ===
    ax_cd = fig.add_subplot(gs[3, 2])
    for s in valid_streaks:
        cum_data = np.cumsum(price_moves[s])
        ax_cd.plot(range(1, len(cum_data)+1), cum_data, label=f'Str {s-1}')
    ax_cd.set_title('Cumulative Delta per Streak')
    ax_cd.set_xlabel('Step in Streak')
    ax_cd.set_ylabel('Cumulative Δ')
    ax_cd.legend(fontsize=8)
    ax_cd.grid(alpha=0.3)

    # === Extreme clustering / heatmap (10) ===
    ax_hm = fig.add_subplot(gs[4, :])
    max_len = max([len(price_moves[s]) for s in valid_streaks])
    heat_data = np.full((len(valid_streaks), max_len), np.nan)
    for i, s in enumerate(valid_streaks):
        arr = price_moves[s]
        heat_data[i, :len(arr)] = arr
    c = ax_hm.imshow(heat_data, aspect='auto', cmap='Reds', origin='lower')
    ax_hm.set_yticks(range(len(valid_streaks)))
    ax_hm.set_yticklabels(streak_labels)
    ax_hm.set_xlabel('Step in Streak')
    ax_hm.set_title('Extreme Clustering Heatmap (Price Delta)')
    fig.colorbar(c, ax=ax_hm, orientation='vertical', label='Price Δ')

    plt.tight_layout()
    plt.show()

from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import numpy as np

def plot_streak_summary(bins, price_moves, max_streaks=10):
    """
    Compacte streak-analyse: violins, tails, skew/kurtosis, cumulative delta, survival, heatmap
    """
    # Filter streaks
    valid_streaks = [s for s in sorted(bins.keys()) if s <= max_streaks and len(price_moves[s]) >= 20]
    streak_labels = [f'Str {s-1}' for s in valid_streaks]
    n_streaks = len(valid_streaks)
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1.2])

    # === Row 1: Violins + IQR + Min/Max + MAD + Fraction Extreme ===
    ax_v = fig.add_subplot(gs[0, :])
    data_list = [price_moves[s] for s in valid_streaks]
    violins = ax_v.violinplot(data_list, showextrema=False)
    for body in violins['bodies']:
        body.set_alpha(0.6)
        body.set_edgecolor('black')
    
    for i, d in enumerate(data_list, start=1):
        median_val = np.median(d)
        q25, q75 = np.percentile(d, [25, 75])
        mad_val = np.median(np.abs(d - median_val))
        p90 = np.percentile(d, 90)
        frac_ext = np.mean(np.array(d) >= p90)

        # IQR
        ax_v.fill_between([i-0.15, i+0.15], q25, q75, alpha=0.3, color='black')
        # Median
        ax_v.scatter(i, median_val, color='black', zorder=3)
        # Min/Max
        ax_v.vlines(i, min(d), max(d), color='grey', alpha=0.5, linewidth=1)
        # MAD
        ax_v.scatter(i, median_val + mad_val, color='blue', marker='^', s=30, label='MAD' if i==1 else "")
        # Fraction extreme
        ax_v.scatter(i, median_val + 2*mad_val, color='red', marker='x', s=30, label='Frac p90+' if i==1 else "")
        # Sample size
        ax_v.text(i, ax_v.get_ylim()[0], f'n={len(d)}', ha='center', va='bottom', fontsize=9)

    ax_v.set_xticks(range(1, n_streaks+1))
    ax_v.set_xticklabels(streak_labels)
    ax_v.set_title('Violins + IQR + Min/Max + MAD + Fraction Extreme')
    ax_v.legend(loc='upper left')
    ax_v.grid(alpha=0.3)

    # === Row 2: Skew, Kurtosis, Tail Slope, Tail Ratio, Entropy, Volatility Accel ===
    metrics = {
        'Skew': [skew(price_moves[s]) for s in valid_streaks],
        'Kurtosis': [kurtosis(price_moves[s]) for s in valid_streaks],
        'TailSlope': [tail_slope(price_moves[s]) for s in valid_streaks],
        'TailRatio': [tail_ratio(price_moves[s]) for s in valid_streaks],
        'Entropy': [entropy_of_distribution(price_moves[s]) for s in valid_streaks],
    }
    # Volatility acceleration
    medians = [np.median(price_moves[s]) for s in valid_streaks]
    metrics['VolAccel'] = [0] + list(np.diff(medians) / medians[:-1])

    ax2 = fig.add_subplot(gs[1, :])
    markers = ['o', 's', '^', 'x', 'D', '*']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for idx, (name, vals) in enumerate(metrics.items()):
        ax2.plot(streak_labels, vals, marker=markers[idx], color=colors[idx], label=name)
    ax2.set_title('Skew, Kurtosis, Tail, Entropy, Volatility Acceleration')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')

    # === Row 3 Left: Cumulative Delta per streak ===
    ax_cd = fig.add_subplot(gs[2, 0])
    for s in valid_streaks:
        cum_data = np.cumsum(price_moves[s])
        ax_cd.plot(range(1, len(cum_data)+1), cum_data, label=f'Str {s-1}')
    ax_cd.set_title('Cumulative Delta per Streak')
    ax_cd.set_xlabel('Step in Streak')
    ax_cd.set_ylabel('Cumulative Δ')
    ax_cd.grid(alpha=0.3)
    ax_cd.legend(fontsize=8)

    # === Row 3 Middle: Survival Curve ===
    ax_sv = fig.add_subplot(gs[2, 1])
    surv = survival_curve(bins)
    ax_sv.plot(range(1, len(surv)+1), surv)
    ax_sv.set_yscale('log')
    ax_sv.set_title('Streak Survival Curve')
    ax_sv.grid(alpha=0.3)

    # === Row 3 Right: Extreme clustering heatmap ===
    ax_hm = fig.add_subplot(gs[2, 2])
    max_len = max([len(price_moves[s]) for s in valid_streaks])
    heat_data = np.full((len(valid_streaks), max_len), np.nan)
    for i, s in enumerate(valid_streaks):
        arr = price_moves[s]
        heat_data[i, :len(arr)] = arr
    c = ax_hm.imshow(heat_data, aspect='auto', cmap='Reds', origin='lower')
    ax_hm.set_yticks(range(len(valid_streaks)))
    ax_hm.set_yticklabels(streak_labels)
    ax_hm.set_xlabel('Step in Streak')
    ax_hm.set_title('Extreme Clustering Heatmap')
    fig.colorbar(c, ax=ax_hm, orientation='vertical', label='Price Δ')

    plt.tight_layout()
    plt.show()

from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import numpy as np

def plot_streak_summary_annotated(bins, price_moves, max_streaks=10):
    """
    Compacte streak-analyse met violins + annotaties + tails + skew/kurtosis + cumulative + survival + heatmap
    """
    # Filter streaks
    valid_streaks = [s for s in sorted(bins.keys()) if s <= max_streaks and len(price_moves[s]) >= 20]
    streak_labels = [f'Str {s-1}' for s in valid_streaks]
    n_streaks = len(valid_streaks)
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1.2])

    # === Row 1: Violins + annotations ===
    ax_v = fig.add_subplot(gs[0, :])
    data_list = [price_moves[s] for s in valid_streaks]
    violins = ax_v.violinplot(data_list, showextrema=False)
    for body in violins['bodies']:
        body.set_alpha(0.6)
        body.set_edgecolor('black')
    
    for i, d in enumerate(data_list, start=1):
        median_val = np.median(d)
        q25, q75 = np.percentile(d, [25, 75])
        mad_val = np.median(np.abs(d - median_val))
        p90 = np.percentile(d, 90)
        frac_ext = np.mean(np.array(d) >= p90)

        # IQR band
        ax_v.fill_between([i-0.15, i+0.15], q25, q75, alpha=0.3, color='black')
        # Median
        ax_v.scatter(i, median_val, color='black', zorder=3)
        # Min/Max
        ax_v.vlines(i, min(d), max(d), color='grey', alpha=0.5, linewidth=1)
        # MAD
        ax_v.scatter(i, median_val + mad_val, color='blue', marker='^', s=40, label='MAD' if i==1 else "")
        # Fraction extreme (p90+)
        ax_v.scatter(i, median_val + 2*mad_val, color='red', marker='x', s=40, label='Frac p90+' if i==1 else "")
        # Sample size
        ax_v.text(i, ax_v.get_ylim()[0], f'n={len(d)}', ha='center', va='bottom', fontsize=9)

        # Extra annotaties: min/max en median+MAD
        ax_v.annotate(f"min: {min(d):.3f}", xy=(i, min(d)), xytext=(0, -15),
                      textcoords='offset points', ha='center', fontsize=8, alpha=0.7)
        ax_v.annotate(f"max: {max(d):.3f}", xy=(i, max(d)), xytext=(0, 5),
                      textcoords='offset points', ha='center', fontsize=8, alpha=0.7)
        ax_v.annotate(f"MAD: {mad_val:.3f}", xy=(i, median_val + mad_val), xytext=(5, 0),
                      textcoords='offset points', fontsize=8, color='blue', alpha=0.7)
        ax_v.annotate(f"Frac90+: {frac_ext:.2f}", xy=(i, median_val + 2*mad_val), xytext=(5, 0),
                      textcoords='offset points', fontsize=8, color='red', alpha=0.7)

    ax_v.set_xticks(range(1, n_streaks+1))
    ax_v.set_xticklabels(streak_labels)
    ax_v.set_title('Violins + IQR + Min/Max + MAD + Fraction Extreme')
    ax_v.legend(loc='upper left')
    ax_v.grid(alpha=0.3)

    # === Row 2: Skew, Kurtosis, Tail Slope, Tail Ratio, Entropy, Volatility Accel ===
    metrics = {
        'Skew': [skew(price_moves[s]) for s in valid_streaks],
        'Kurtosis': [kurtosis(price_moves[s]) for s in valid_streaks],
        'TailSlope': [tail_slope(price_moves[s]) for s in valid_streaks],
        'TailRatio': [tail_ratio(price_moves[s]) for s in valid_streaks],
        'Entropy': [entropy_of_distribution(price_moves[s]) for s in valid_streaks],
    }
    medians = [np.median(price_moves[s]) for s in valid_streaks]
    metrics['VolAccel'] = [0] + list(np.diff(medians) / medians[:-1])

    ax2 = fig.add_subplot(gs[1, :])
    markers = ['o', 's', '^', 'x', 'D', '*']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for idx, (name, vals) in enumerate(metrics.items()):
        ax2.plot(streak_labels, vals, marker=markers[idx], color=colors[idx], label=name)
    ax2.set_title('Skew, Kurtosis, Tail, Entropy, Volatility Acceleration')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')

    # === Row 3 Left: Cumulative Delta per streak ===
    ax_cd = fig.add_subplot(gs[2, 0])
    for s in valid_streaks:
        cum_data = np.cumsum(price_moves[s])
        ax_cd.plot(range(1, len(cum_data)+1), cum_data, label=f'Str {s-1}')
    ax_cd.set_title('Cumulative Delta per Streak')
    ax_cd.set_xlabel('Step in Streak')
    ax_cd.set_ylabel('Cumulative Δ')
    ax_cd.grid(alpha=0.3)
    ax_cd.legend(fontsize=8)

    # === Row 3 Middle: Survival Curve ===
    ax_sv = fig.add_subplot(gs[2, 1])
    surv = survival_curve(bins)
    ax_sv.plot(range(1, len(surv)+1), surv)
    ax_sv.set_yscale('log')
    ax_sv.set_title('Streak Survival Curve')
    ax_sv.grid(alpha=0.3)

    # === Row 3 Right: Extreme clustering heatmap ===
    ax_hm = fig.add_subplot(gs[2, 2])
    max_len = max([len(price_moves[s]) for s in valid_streaks])
    heat_data = np.full((len(valid_streaks), max_len), np.nan)
    for i, s in enumerate(valid_streaks):
        arr = price_moves[s]
        heat_data[i, :len(arr)] = arr
    c = ax_hm.imshow(heat_data, aspect='auto', cmap='Reds', origin='lower')
    ax_hm.set_yticks(range(len(valid_streaks)))
    ax_hm.set_yticklabels(streak_labels)
    ax_hm.set_xlabel('Step in Streak')
    ax_hm.set_title('Extreme Clustering Heatmap')
    fig.colorbar(c, ax=ax_hm, orientation='vertical', label='Price Δ')

    plt.tight_layout()
    plt.show()

def plot_streak_summary_extremes(bins, price_moves, max_streaks=10):
    """
    Compacte streak-analyse met violins + annotaties + extreme highlight + tails + skew/kurtosis
    """
    import matplotlib.pyplot as plt
    from scipy.stats import skew, kurtosis
    import numpy as np

    valid_streaks = [s for s in sorted(bins.keys()) if s <= max_streaks and len(price_moves[s]) >= 20]
    streak_labels = [f'Str {s-1}' for s in valid_streaks]
    n_streaks = len(valid_streaks)
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1.2])
    
    # === Row 1: Violins + median/MAD/extremes/p10/p90 ===
    ax_v = fig.add_subplot(gs[0, :])
    data_list = [price_moves[s] for s in valid_streaks]
    violins = ax_v.violinplot(data_list, showextrema=False)
    for body in violins['bodies']:
        body.set_alpha(0.6)
        body.set_edgecolor('black')
    
    for i, d in enumerate(data_list, start=1):
        median_val = np.median(d)
        mean_val = np.mean(d)
        q25, q75 = np.percentile(d, [25, 75])
        p10, p90 = np.percentile(d, [10, 90])
        mad_val = np.median(np.abs(d - median_val))
        extreme_threshold = np.percentile(d, 95)
        top_extremes = [x for x in d if x >= extreme_threshold]
        frac_ext = len(top_extremes) / len(d)

        # IQR band
        ax_v.fill_between([i-0.15, i+0.15], q25, q75, alpha=0.3, color='black')
        # Median
        ax_v.scatter(i, median_val, color='black', zorder=3)
        # Mean
        ax_v.scatter(i, mean_val, color='orange', marker='D', s=40, label='Mean' if i==1 else "")
        # MAD
        ax_v.scatter(i, median_val + mad_val, color='blue', marker='^', s=40, label='MAD' if i==1 else "")
        # Fraction top 5% extreme
        ax_v.scatter(i, median_val + 2*mad_val, color='red', marker='x', s=40, label='Top5%' if i==1 else "")
        # p10/p90 lines
        ax_v.vlines(i-0.1, p10, p90, color='purple', linewidth=2, alpha=0.6)
        # Annotaties min/max
        ax_v.annotate(f"min: {min(d):.3f}", xy=(i, min(d)), xytext=(0, -15),
                      textcoords='offset points', ha='center', fontsize=8, alpha=0.7)
        ax_v.annotate(f"max: {max(d):.3f}", xy=(i, max(d)), xytext=(0, 5),
                      textcoords='offset points', ha='center', fontsize=8, alpha=0.7)
        # Annotatie fraction extreme
        ax_v.annotate(f"Top5% frac: {frac_ext:.2f}", xy=(i, median_val + 2*mad_val), xytext=(5, 0),
                      textcoords='offset points', fontsize=8, color='red', alpha=0.7)
        # Annotatie sample size
        ax_v.text(i, ax_v.get_ylim()[0], f'n={len(d)}', ha='center', va='bottom', fontsize=9)
        # Annotatie mean-median verschil
        ax_v.annotate(f"Δmean-med: {mean_val - median_val:.3f}", xy=(i, median_val), xytext=(5,5),
                      textcoords='offset points', fontsize=8, color='orange', alpha=0.7)
        # Extreme points highlight
        ax_v.scatter([i]*len(top_extremes), top_extremes, color='darkred', marker='*', s=50, label='Extreme' if i==1 else "")

    ax_v.set_xticks(range(1, n_streaks+1))
    ax_v.set_xticklabels(streak_labels)
    ax_v.set_title('Violins + Annotated Extremes + MAD + Top5% Highlight + p10/p90 + Mean')
    ax_v.legend(loc='upper left')
    ax_v.grid(alpha=0.3)
    
    # === Row 2: Metrics: skew/kurtosis/tail/entropy/vol accel ===
    metrics = {
        'Skew': [skew(price_moves[s]) for s in valid_streaks],
        'Kurtosis': [kurtosis(price_moves[s]) for s in valid_streaks],
        'TailSlope': [tail_slope(price_moves[s]) for s in valid_streaks],
        'TailRatio': [tail_ratio(price_moves[s]) for s in valid_streaks],
        'Entropy': [entropy_of_distribution(price_moves[s]) for s in valid_streaks],
    }
    medians = [np.median(price_moves[s]) for s in valid_streaks]
    metrics['VolAccel'] = [0] + list(np.diff(medians) / medians[:-1])
    
    ax2 = fig.add_subplot(gs[1, :])
    markers = ['o', 's', '^', 'x', 'D', '*']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for idx, (name, vals) in enumerate(metrics.items()):
        ax2.plot(streak_labels, vals, marker=markers[idx], color=colors[idx], label=name)
    ax2.set_title('Skew, Kurtosis, Tail, Entropy, Volatility Acceleration')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')
    
    # === Row 3 Left: Cumulative Delta ===
    ax_cd = fig.add_subplot(gs[2, 0])
    for s in valid_streaks:
        cum_data = np.cumsum(price_moves[s])
        ax_cd.plot(range(1, len(cum_data)+1), cum_data, label=f'Str {s-1}')
    ax_cd.set_title('Cumulative Delta per Streak')
    ax_cd.set_xlabel('Step in Streak')
    ax_cd.set_ylabel('Cumulative Δ')
    ax_cd.grid(alpha=0.3)
    ax_cd.legend(fontsize=8)
    
    # === Row 3 Middle: Survival Curve ===
    ax_sv = fig.add_subplot(gs[2, 1])
    surv = survival_curve(bins)
    ax_sv.plot(range(1, len(surv)+1), surv)
    ax_sv.set_yscale('log')
    ax_sv.set_title('Streak Survival Curve')
    ax_sv.grid(alpha=0.3)
    
    # === Row 3 Right: Extreme clustering heatmap ===
    ax_hm = fig.add_subplot(gs[2, 2])
    max_len = max([len(price_moves[s]) for s in valid_streaks])
    heat_data = np.full((len(valid_streaks), max_len), np.nan)
    for i, s in enumerate(valid_streaks):
        arr = price_moves[s]
        heat_data[i, :len(arr)] = arr
    c = ax_hm.imshow(heat_data, aspect='auto', cmap='Reds', origin='lower')
    ax_hm.set_yticks(range(len(valid_streaks)))
    ax_hm.set_yticklabels(streak_labels)
    ax_hm.set_xlabel('Step in Streak')
    ax_hm.set_title('Extreme Clustering Heatmap')
    fig.colorbar(c, ax=ax_hm, orientation='vertical', label='Price Δ')
    
    plt.tight_layout()
    plt.show()

def plot_streak_full_analysis(bins, price_moves, max_streaks=10):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import skew, kurtosis

    # Valid streaks
    valid_streaks = [s for s in sorted(bins.keys()) if s <= max_streaks and len(price_moves[s]) >= 20]
    streak_labels = [f'Str {s-1}' for s in valid_streaks]
    n_streaks = len(valid_streaks)

    # Directional cumulative data
    all_up = []
    all_down = []
    for s in valid_streaks:
        # Voorbeeld: als we weten welke richting per streak, hier dummy split
        data = price_moves[s]
        # Simuleer gelijk verdeeld
        half = len(data)//2
        all_up.extend(data[:half])
        all_down.extend(data[half:])

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1])

    # === Row 1: Violins per streak met p90/p100 highlight + tail-skew ===
    ax_v = fig.add_subplot(gs[0, :])
    data_list = [price_moves[s] for s in valid_streaks]
    violins = ax_v.violinplot(data_list, showextrema=False)
    for body in violins['bodies']:
        body.set_alpha(0.5)
        body.set_edgecolor('black')

    for i, d in enumerate(data_list, start=1):
        median_val = np.median(d)
        mean_val = np.mean(d)
        mad_val = np.median(np.abs(d - median_val))
        p90 = np.percentile(d, 90)
        p95 = np.percentile(d, 95)
        p10 = np.percentile(d, 10)
        top_extremes = [x for x in d if x >= p95]
        frac_p90 = np.mean(np.array(d) >= p90)

        # Core (p0-p90)
        ax_v.violinplot([x for x in d if x <= p90], positions=[i], showextrema=False)
        # Tail (p90-p100)
        ax_v.violinplot([x for x in d if x > p90], positions=[i], showextrema=False)

        # Annotations
        ax_v.scatter(i, median_val, color='black', zorder=3)
        ax_v.scatter(i, mean_val, color='orange', marker='D', s=40)
        ax_v.scatter(i, median_val + mad_val, color='blue', marker='^', s=40)
        ax_v.scatter([i]*len(top_extremes), top_extremes, color='red', marker='*', s=50)
        ax_v.text(i, ax_v.get_ylim()[0], f'n={len(d)}', ha='center', va='bottom', fontsize=9)
        ax_v.annotate(f"Frac p90+: {frac_p90:.2f}", xy=(i, median_val+2*mad_val), xytext=(5,0),
                      textcoords='offset points', fontsize=8, color='red')

        # Tail skew pijltjes
        skew_val = skew(d)
        if skew_val > 0.3:
            ax_v.annotate('→', xy=(i, median_val), xytext=(0,5), textcoords='offset points', fontsize=12, color='purple')
        elif skew_val < -0.3:
            ax_v.annotate('←', xy=(i, median_val), xytext=(0,5), textcoords='offset points', fontsize=12, color='purple')

    ax_v.set_xticks(range(1, n_streaks+1))
    ax_v.set_xticklabels(streak_labels)
    ax_v.set_title('Violins with Median, Mean, MAD, Top5%, p90/p100 & Tail Skew')
    ax_v.grid(alpha=0.3)

    # === Row 2: Directional violins (UP vs DOWN) ===
    ax_dir = fig.add_subplot(gs[1, :])
    ax_dir.violinplot([all_up, all_down], showextrema=True)
    ax_dir.set_xticks([1,2])
    ax_dir.set_xticklabels(['UP', 'DOWN'])
    ax_dir.set_title('Directional Violins: cumulative streaks UP vs DOWN')
    ax_dir.grid(alpha=0.3)

    # === Row 3: Survival curves per direction ===
    ax_sv = fig.add_subplot(gs[2, 0])
    surv_all = survival_curve(bins)
    ax_sv.plot(range(1, len(surv_all)+1), surv_all)
    ax_sv.set_yscale('log')
    ax_sv.set_title('Streak Survival Curve')
    ax_sv.grid(alpha=0.3)

    # === Row 3: Cumulative delta ===
    ax_cd = fig.add_subplot(gs[2, 1:])
    for s in valid_streaks:
        cum_data = np.cumsum(price_moves[s])
        ax_cd.plot(range(1,len(cum_data)+1), cum_data, label=f'Str {s-1}')
    ax_cd.set_title('Cumulative Delta per Streak')
    ax_cd.grid(alpha=0.3)

    # === Row 4: Heatmap extreme clustering ===
    ax_hm = fig.add_subplot(gs[3, :])
    max_len = max([len(price_moves[s]) for s in valid_streaks])
    heat_data = np.full((len(valid_streaks), max_len), np.nan)
    for i,s in enumerate(valid_streaks):
        arr = price_moves[s]
        heat_data[i, :len(arr)] = arr
    c = ax_hm.imshow(heat_data, aspect='auto', cmap='Reds', origin='lower')
    ax_hm.set_yticks(range(len(valid_streaks)))
    ax_hm.set_yticklabels(streak_labels)
    ax_hm.set_xlabel('Step in Streak')
    ax_hm.set_title('Extreme Clustering Heatmap')
    fig.colorbar(c, ax=ax_hm, orientation='vertical', label='Price Δ')

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
plot_streak_stats_extended(bins, price_moves)
plot_streak_stats_full(bins, price_moves)
plot_streak_summary(bins, price_moves)
plot_streak_summary_annotated(bins, price_moves)
plot_streak_summary_extremes(bins, price_moves)
plot_streak_full_analysis(bins, price_moves)

# Maak gedetailleerde percentielprofielen voor belangrijke streaks
for streak in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    if streak in price_moves and len(price_moves[streak]) >= 10:  # Minimaal 10 samples
        create_percentile_profile(price_moves, streak_length=streak, p_start=90, p_end=100, step=1)
