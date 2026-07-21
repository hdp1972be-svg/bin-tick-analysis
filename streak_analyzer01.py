#!/usr/local/bin/python3.11
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
from datetime import datetime
from statsmodels.tsa.stattools import acf
from collections import defaultdict, Counter
from matplotlib.ticker import ScalarFormatter
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib import colormaps
from matplotlib.lines import Line2D

import json,os

UP = 1
DOWN = -1

MAX_VIOLINS = 5

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False

# ----------------------------
# Drawing Utilities
# ----------------------------

def ecdf(arr):
    ''' DOCSTRING PLACEHOLDER '''
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def stats_dict(arr):
    ''' DOCSTRING PLACEHOLDER '''
    med = np.median(arr)
    return {
        "min": np.min(arr),
        "max": np.max(arr),
        "mean": np.mean(arr),
        "mad": np.median(np.abs(arr - med)),
        "p90": np.percentile(arr, 90),
    }

def draw_regime_background(ax, center_x, width, y0, y1, side, color, alpha=0.25, zorder=0):
    if side == "left":
        x0 = center_x - width
    else:
        x0 = center_x

    rect = Rectangle(
        (x0, y0),
        width,
        y1 - y0,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(rect)

def clip_violin_half(body, center_x, side):
    verts = body.get_paths()[0].vertices
    if side == "left":
        verts[:, 0] = np.minimum(verts[:, 0], center_x)
    else:
        verts[:, 0] = np.maximum(verts[:, 0], center_x)

def violin_half_width(violin_body, center_x):
    ''' DOCSTRING PLACEHOLDER '''
    verts = violin_body.get_paths()[0].vertices
    xs = verts[:, 0]
    return np.max(np.abs(xs - center_x))

def annotate_stats(ax, x, stats, side, color, violin=None, leader_length=0.05):
    """
    side: 'left' of 'right'
    violin: het violon object (v['bodies'][0]) voor deze data
    leader_length: lengte van de leader line
    """
    if violin is None:
        # fallback: gewoon dx
        dx = -leader_length if side == "left" else leader_length
        ha = "right" if side == "left" else "left"
        start_x = x
        for k, y in stats.items():
            ax.annotate(
                f"{k}: {y:.2f}".rstrip('0').rstrip('.'),
                xy=(start_x, y),
                xytext=(start_x + dx, y),
                arrowprops=dict(arrowstyle='-', lw=0.8),
                fontsize=8,
                ha=ha,
                va="center",
                color=color,
                alpha=0.85
            )
        return

    # haal vertices van de body
    verts = violin.get_paths()[0].vertices
    xs = verts[:, 0]
    ys = verts[:, 1]

    # voor elke stat: vind de x aan de rand bij die y
    for k, y in stats.items():
        # vind vertices dichtbij de y waarde
        idx = np.argmin(np.abs(ys - y))
        if side == "left":
            start_x = min(xs[ys == ys[idx]])
        else:
            start_x = max(xs[ys == ys[idx]])

        dx = -leader_length if side == "left" else leader_length
        ha = "right" if side == "left" else "left"

        ax.annotate(
            f"{k}: {y:.2f}".rstrip('0').rstrip('.'),
            xy=(start_x, y),
            xytext=(start_x + dx, y),
            arrowprops=dict(arrowstyle='-', lw=0.8),
            fontsize=8,
            ha=ha,
            va="center",
            color=color,
            alpha=0.85
        )

# ----------------------------
# Stats Utilities
# ----------------------------

def poisson_lambda(iat):
    iat = np.array(iat)
    return 1.0 / iat.mean()

def poisson_ks_stat(iat):
    ''' Goodness-of-fit against exponential '''
    iat = np.array(iat)
    lam = poisson_lambda(iat)
    sorted_iat = np.sort(iat)
    ecdf = np.arange(1, len(iat)+1) / len(iat)
    model = 1 - np.exp(-lam * sorted_iat)
    return np.max(np.abs(ecdf - model))

def mad(arr):
    """Median absolute deviation"""
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def suggest_global_grid_width(price_moves, streaks):
    gaps = []
    for s in streaks:
        data = np.asarray(price_moves[s])
        if len(data) < 20:
            continue
        med = np.median(data)
        p90 = np.percentile(data, 90)
        gaps.append(p90 - med)
    if not gaps:
        return None
    return np.median(gaps)

def suggested_grid_width(arr):
    """Voorstel voor grid width gebaseerd op p90 - median"""
    arr = np.array(arr)
    med = np.median(arr)
    p90 = np.percentile(arr, 90)
    return float(p90 - med)

def compute_regimes(arr):
    """
    Bepaal regime-grenzen zoals p90 en p99
    """
    arr = np.array(arr)
    return {
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "p99.9": float(np.percentile(arr, 99.9)),
        "max": float(arr.max())
    }

def calculate_mass_ratios_old(arr, p90, p99):
    arr = np.array(arr)
    normal = np.sum(arr <= p90) / len(arr)
    burst = np.sum((arr > p90) & (arr <= p99)) / len(arr)
    extreme = np.sum(arr > p99) / len(arr)
    return {"normal": normal, "burst": burst, "extreme": extreme}

def calculate_mass_ratios(arr, p90, p99):
    arr = np.array(arr)
    total = len(arr)
    normal = np.sum(arr <= p90) / total if total > 0 else 0
    burst = np.sum((arr > p90) & (arr <= p99)) / total if total > 0 else 0
    extreme = np.sum(arr > p99) / total if total > 0 else 0
    return {"normal": normal, "burst": burst, "extreme": extreme}

def directional_asymmetry(up_data, down_data):
    if len(up_data) == 0 or len(down_data) == 0:
        return 0.0
    mean_up = np.mean(up_data)
    mean_down = np.mean(down_data)
    denominator = mean_up + mean_down
    if denominator == 0:
        return 0.0
    return (mean_up - mean_down) / denominator

def stability_score_old(arr):
    med = np.median(arr)
    mad_val = np.median(np.abs(arr - med))
    return mad_val / (med + 1e-10)

def stability_score(arr):
    med = np.median(arr)
    if med == 0:
        return 0.0
    mad_val = np.median(np.abs(arr - med))
    return mad_val / med

def calculate_burstiness(streak_sequence):
    """
    Bereken burstiness index:
    B = (σ - μ) / (σ + μ)
    waar μ en σ de mean en std van streak lengths zijn.
    
    B ≈ -1: regelmatig (periodiek)
    B ≈ 0: willekeurig (Poisson)
    B ≈ 1: bursty
    """
    if len(streak_sequence) < 2:
        return 0.0
    
    arr = np.array(streak_sequence)
    mu = np.mean(arr)
    sigma = np.std(arr)
    
    if sigma + mu == 0:
        return 0.0
    
    return float((sigma - mu) / (sigma + mu))

def entropy_of_deltas_old(arr, bins=20):
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def entropy_of_deltas(arr, bins=20):
    if len(arr) == 0:
        return 0.0
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

# ----------------------------
# Data reader
# ----------------------------

def read_csv(path):
    price_moves = {}
    price_moves_up = {}
    price_moves_down = {}
    bins = {}
    ticks = 0
    interarrival = []
    prev_ts = None

    prev_price = None
    prev_dir = None
    streak_len = 0
    start_price = None

    streak_sequence = []  # lijst van alle streak lengths in volgorde
    prices = [] 
    event_times = []

    # --- ADDED: streak start/end indices ---
    streak_start_idx = []  # ADDED
    streak_end_idx = []    # ADDED

    current_streak_length = 0

    with open(path) as f:
        r = csv.reader(f)
        next(r)

        for row in r:

            ticks += 1

            ts_str = row[0]  # bv '2025-04-01 00:00:00.953+0200'
            dt = datetime.strptime(ts_str[:-5], "%Y-%m-%d %H:%M:%S.%f")  # strip timezone +0200
            ts = dt.timestamp()  # float in seconden

            price = float(row[2])
            prices.append(price)  # <--- opslaan

            event_times.append(ts)

            if prev_ts is not None:
                interarrival.append(ts - prev_ts)

            prev_ts = ts

            price = float(row[2])

            if prev_price is None:
                prev_price = price
                continue

            if price > prev_price:
                d = UP
            elif price < prev_price:
                d = DOWN
            else:
                prev_price = price
                continue

            if streak_len == 0:
                start_price = prev_price
                streak_start_tick = len(prices) - 1  # ADDED: mark start index

            streak_len += 1

            if prev_dir is not None and d != prev_dir:
                delta = abs(prev_price - start_price)

                price_moves.setdefault(streak_len, []).append(delta)
                bins[streak_len] = bins.get(streak_len, 0) + 1

                if prev_dir == UP:
                    price_moves_up.setdefault(streak_len, []).append(delta)
                else:
                    price_moves_down.setdefault(streak_len, []).append(delta)

                streak_sequence.append(streak_len)
                streak_start_idx.append(streak_start_tick)  # ADDED
                streak_end_idx.append(len(prices) - 1)      # ADDED

                streak_len = 1
                start_price = prev_price
                streak_start_tick = len(prices) - 1  # ADDED: reset start for new streak

            prev_dir = d
            prev_price = price

    # Laatste streak toevoegen
    if streak_len > 0:
        delta = abs(price - start_price)
        price_moves.setdefault(streak_len, []).append(delta)
        bins[streak_len] = bins.get(streak_len, 0) + 1
        streak_sequence.append(streak_len)
        streak_start_idx.append(streak_start_tick)  # ADDED
        streak_end_idx.append(len(prices) - 1)      # ADDED

    total_streaks = sum(bins.values())

    return bins, price_moves, price_moves_up, price_moves_down, ticks, total_streaks, interarrival, streak_sequence, streak_start_idx, streak_end_idx, prices, event_times

def generate_streak_info(prices):
    """
    Genereer streak_id en streak_pos arrays uit een lijst van prijzen.
    Up = +1, Down = -1
    """
    prices = np.asarray(prices)
    signed_deltas = np.diff(prices)
    
    streak_id = np.zeros(len(signed_deltas), dtype=int)
    streak_pos = np.zeros(len(signed_deltas), dtype=int)
    
    current_id = 1
    pos = 1
    streak_id[0] = current_id
    streak_pos[0] = pos
    
    for i in range(1, len(signed_deltas)):
        if np.sign(signed_deltas[i]) != np.sign(signed_deltas[i-1]):
            current_id += 1
            pos = 1
        else:
            pos += 1
        streak_id[i] = current_id
        streak_pos[i] = pos
    
    return signed_deltas, streak_id, streak_pos

# ----------------------------
# Stat calculators
# ----------------------------

def analyze_streak_runs(streak_sequence):
    """
    Analyze runs (consecutive streaks of same length).
    
    Example: [1,1,1,2,1,1,3,3,3,3,1] →
        1: runs of length 3
        2: runs of length 1  
        1: runs of length 2
        3: runs of length 4
        1: runs of length 1
    
    Returns:
        runs_by_length: {streak_length: [run_lengths]}
        run_stats: {streak_length: {mean, max, std, ...}}
    """
    runs_by_length = {}
    
    if not streak_sequence:
        return runs_by_length, {}
    
    current = streak_sequence[0]
    count = 1
    
    for s in streak_sequence[1:]:
        if s == current:
            count += 1
        else:
            runs_by_length.setdefault(current, []).append(count)
            current = s
            count = 1
    
    # Laatste run
    runs_by_length.setdefault(current, []).append(count)
    
    # Bereken statistieken per streak length
    run_stats = {}
    for s, runs in runs_by_length.items():
        arr = np.array(runs)
        run_stats[s] = {
            "mean_run_length": float(np.mean(arr)),
            "max_run_length": int(np.max(arr)),
            "std_run_length": float(np.std(arr)),
            "p90_run_length": float(np.percentile(arr, 90)),
            "total_runs": len(runs),
            "fraction_long_runs": float(np.sum(arr > 1) / len(arr)) if len(arr) > 0 else 0
        }
    
    return runs_by_length, run_stats

def calculate_transition_matrix(streak_sequence):
    """
    Bereken overgangskansen tussen streak lengths.
    
    Returns:
        transition_matrix: dict of dict
        stationaire_distributie: berekend uit eigenvectoren
    """
    unique_streaks = sorted(set(streak_sequence))
    n = len(unique_streaks)
    
    # Maak mapping naar indices
    idx_map = {s: i for i, s in enumerate(unique_streaks)}
    
    # Initialiseer matrix
    matrix = np.zeros((n, n))
    
    # Tel transities
    for i in range(len(streak_sequence) - 1):
        from_idx = idx_map[streak_sequence[i]]
        to_idx = idx_map[streak_sequence[i+1]]
        matrix[from_idx, to_idx] += 1
    
    # Normaliseer naar kansen
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # voorkom delen door 0
    transition_matrix = matrix / row_sums
    
    # Converteer terug naar dict vorm
    trans_dict = {}
    for i, s_from in enumerate(unique_streaks):
        trans_dict[s_from] = {}
        for j, s_to in enumerate(unique_streaks):
            if transition_matrix[i, j] > 0:
                trans_dict[s_from][s_to] = float(transition_matrix[i, j])
    
    # Bereken stationaire distributie (indien mogelijk)
    stationary = None
    if n > 0:
        try:
            # Bereken eigenvector voor eigenwaarde 1
            eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
            idx = np.where(np.abs(eigvals - 1.0) < 1e-10)[0]
            if len(idx) > 0:
                stationary_vec = np.real(eigvecs[:, idx[0]])
                stationary_vec = stationary_vec / stationary_vec.sum()
                stationary = {unique_streaks[i]: float(stationary_vec[i]) for i in range(n)}
        except:
            pass
    
    return trans_dict, stationary

def calculate_optimal_grid_params(streak_sequence, price_moves):
    """
    Bepaal grid parameters gebaseerd op sequentie-analyse.
    """
    burstiness = calculate_burstiness(streak_sequence)
    
    if burstiness > 0.3:
        # Trending markt: minder grid levels, verder uit elkaar
        return {
            "strategy": "trend_following_grid",
            "n_levels": 3,
            "spacing": "wide",
            "warning": "Clusters of long streaks detected"
        }
    else:
        # Ranging markt: meer grid levels, dichter bij elkaar
        return {
            "strategy": "dense_grid",
            "n_levels": 10,
            "spacing": "tight",
            "warning": "Mean-reverting behavior detected"
        }

def calculate_autocorrelation(streak_sequence, max_lag=10):
    """
    Bereken autocorrelatie van streak lengths.
    """
    if len(streak_sequence) < max_lag + 1:
        return {}
    
    
    try:
        autocorr = acf(streak_sequence, nlags=max_lag, fft=False)
        return {f"lag_{i}": float(autocorr[i]) for i in range(1, min(max_lag + 1, len(autocorr)))}
    except:
        # Fallback naar simpele correlatie
        autocorr = {}
        seq = np.array(streak_sequence)
        for lag in range(1, min(max_lag + 1, len(seq) // 2)):
            if len(seq) > lag:
                corr = np.corrcoef(seq[:-lag], seq[lag:])[0, 1]
                autocorr[f"lag_{lag}"] = float(corr) if not np.isnan(corr) else 0.0
        return autocorr

def continuation_stats(streak_sequence, price_moves):
    """
    For streaks that repeat (double, triple, …),
    estimate continuation probability.
    """
    out = {}
    seq = np.asarray(streak_sequence)

    for s in np.unique(seq):
        idx = np.where(seq == s)[0]
        cont = 0
        for i in idx:
            if i + 1 < len(seq) and seq[i + 1] == s:
                cont += 1
        out[s] = {
            "count": int(len(idx)),
            "continuation_prob": cont / len(idx) if len(idx) else 0
        }
    return out

# ----------------------------
# Plot functions
# ----------------------------

def plot_streak_run_distribution_old(
    streak_lengths,
    max_streak=5,
    normalize=True
):
    """
    streak_lengths: 1D array of streak identifiers (e.g. 1,1,1,2,1,1,3,...)
    max_streak: plot up to this streak length
    normalize: percentages vs absolute counts
    """

    fig, ax = plt.subplots()

    run_counts = {k: [] for k in range(1, max_streak + 1)}

    current = streak_lengths[0]
    run_len = 1

    for s in streak_lengths[1:]:
        if s == current:
            run_len += 1
        else:
            if current in run_counts:
                run_counts[current].append(run_len)
            current = s
            run_len = 1

    # last run
    if current in run_counts:
        run_counts[current].append(run_len)

    for k, runs in run_counts.items():
        if not runs:
            continue
        vals, counts = np.unique(runs, return_counts=True)
        if normalize:
            counts = counts / counts.sum() * 100
        ax.plot(vals, counts, marker="o", label=f"streak {k}")

    ax.set_xlabel("aantal keer na elkaar")
    ax.set_ylabel("percentage" if normalize else "count")
    ax.legend()
    ax.set_title("Run-length verdeling per streak")

    return fig

def plot_streak_run_distribution(
    streak_lengths,
    max_streak=5,
    normalize=True
):
    """
    Grouped bar plot:
    run-length distributie per streak-type
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    run_counts = {k: [] for k in range(1, max_streak + 1)}

    current = streak_lengths[0]
    run_len = 1

    for s in streak_lengths[1:]:
        if s == current:
            run_len += 1
        else:
            if current in run_counts:
                run_counts[current].append(run_len)
            current = s
            run_len = 1

    if current in run_counts:
        run_counts[current].append(run_len)

    # collect all run lengths
    all_runs = sorted(
        set(r for runs in run_counts.values() for r in runs)
    )

    x = np.arange(len(all_runs))
    width = 0.8 / max_streak

    cmap = plt.cm.tab10

    for i, k in enumerate(range(1, max_streak + 1)):
        runs = run_counts[k]
        if not runs:
            continue

        vals, counts = np.unique(runs, return_counts=True)

        freq = np.zeros(len(all_runs))
        for v, c in zip(vals, counts):
            freq[all_runs.index(v)] = c

        if normalize and freq.sum() > 0:
            freq = freq / freq.sum() * 100
            freq = np.clip(freq, 0.1, None)  # 0.1% minimum om log goed te tonen

        ax.bar(
            x + i * width,
            freq,
            width=width,
            color=cmap(i),
            alpha=0.85,
            label=f"streak {k}"
        )

    ax.set_xticks(x + width * (max_streak - 1) / 2)
    ax.set_xticklabels(all_runs)
    ax.set_xlabel("run-length (aantal keer na elkaar)")
    ax.set_ylabel("percentage" if normalize else "count")
    ax.set_title("Run-length verdeling per streak-type")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.ticklabel_format(style='plain', axis='y')

#    for i, label in enumerate(ax.get_xticklabels()):
#        offset = 5 if i % 2 == 0 else -5
#        label.set_y(label.get_position()[1] + offset / fig.dpi)

    ax.set_yscale("log")
#    ax.set_ylim(0.1, None) 

    return fig

def plot_cumulative_streak_violins_old(
    signed_deltas,
    streak_ids,
    max_depth=5,
    quantiles=(0.5, 0.9)
):
    """
    signed_deltas: delta per streak
    streak_ids: streak index/order (1,2,3,...)
    max_depth: cumulative depth (1-2, 1-3, ...)
    """

    fig, ax = plt.subplots()

    violins = []
    labels = []

    for d in range(2, max_depth + 1):
        mask = streak_ids <= d
        cum = np.cumsum(signed_deltas[mask])
        violins.append(cum)
        labels.append(f"1→{d}")

    vp = ax.violinplot(violins, showmeans=False, showmedians=False)

    for i, data in enumerate(violins, start=1):
        for q in quantiles:
            y = np.quantile(data, q)
            ax.plot(i, y, marker="_", color="black")

    ax.axhline(0, linestyle="--")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("cumulatieve delta")
    ax.set_title("Cumulatieve streak payoff (empirisch)")

    return fig

def plot_cumulative_streak_violins_old(
    signed_deltas,
    streak_id,
    streak_pos,
    max_depth=5,
    quantiles=(0.5, 0.9, 0.99),
):
    """
    Empirische cumulatieve violins:
    som van delta(1..d) per streak
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    violins = []
    labels = []

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, max_depth - 1))

    for i, d in enumerate(range(2, max_depth + 1)):
        sums = []

        for sid in np.unique(streak_id):
            mask = (streak_id == sid) & (streak_pos <= d)
            if mask.sum() == d:
                sums.append(signed_deltas[mask].sum())

        violins.append(np.array(sums))
        labels.append(f"1→{d}")

    vp = ax.violinplot(
        violins,
        showmeans=False,
        showmedians=False,
        widths=0.8
    )

    # kleur violins
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.8)

    # quantiles
    for i, data in enumerate(violins, start=1):
        for q in quantiles:
            y = np.quantile(data, q)
            ax.scatter(
                i, y,
                s=60,
                marker="_",
                linewidths=3,
                label=f"p{int(q*100)}" if i == 1 else "",
                color="black"
            )

    ax.axhline(0, linestyle="--", color="grey", alpha=0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("cumulatieve Δ price")
    ax.set_title("Cumulatieve streak payoff — empirische violins")

    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    return fig

def plot_cumulative_streak_violins(
    signed_deltas,
    streak_id,
    streak_pos,
    max_depth=5,
    quantiles=(0.5, 0.9, 0.99),
    log_scale=False
):
    """
    Empirische cumulatieve violins per streak:
    - Gesplitst up/down
    - Signed vs absolute violins naast elkaar
    - Annotatie van p50, p90, p99
    - Optioneel log-scaling voor fat tails
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_signed, ax_abs = axes

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, max_depth-1))

    # Helper om data te verzamelen per depth
    def collect_cumulative(streak_mask, deltas, depth):
        sums = []
        for sid in np.unique(streak_id):
            mask = (streak_id == sid) & (streak_pos <= depth) & streak_mask
            if mask.sum() == depth:
                sums.append(deltas[mask].sum())
        return np.array(sums)

    labels = [f"1→{d}" for d in range(2, max_depth+1)]

    for i, d in enumerate(range(2, max_depth+1)):
        # signed violins: split up/down
        up_mask = signed_deltas > 0
        down_mask = signed_deltas < 0

        up_data = collect_cumulative(up_mask, signed_deltas, d)
        down_data = collect_cumulative(down_mask, signed_deltas, d)

        signed_data = [up_data, down_data]
        abs_data = [np.abs(up_data), np.abs(down_data)]

        # Plot signed violins
        vp = ax_signed.violinplot(
            signed_data,
            positions=[2*i+1, 2*i+2],
            showmeans=False, showmedians=False, widths=0.8
        )
        for body, color in zip(vp["bodies"], [colors[i]]*2):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.8)

        # Plot absolute violins
        vp_abs = ax_abs.violinplot(
            abs_data,
            positions=[2*i+1, 2*i+2],
            showmeans=False, showmedians=False, widths=0.8
        )
        for body, color in zip(vp_abs["bodies"], [colors[i]]*2):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.8)

        # Annotatie quantiles voor signed
        for q in quantiles:
            for pos, data in zip([2*i+1, 2*i+2], signed_data):
                if len(data) > 0:
                    y = np.quantile(data, q)
                    ax_signed.scatter(pos, y, s=50, marker="_", linewidths=2, color="black")
                    if pos == 1:
                        ax_signed.text(pos, y, f"p{int(q*100)}", fontsize=8, va='bottom')

            # Annotatie quantiles voor abs
            for pos, data in zip([2*i+1, 2*i+2], abs_data):
                if len(data) > 0:
                    y = np.quantile(data, q)
                    ax_abs.scatter(pos, y, s=50, marker="_", linewidths=2, color="black")

    # General formatting
    ax_signed.axhline(0, linestyle="--", color="grey", alpha=0.6)
    ax_signed.set_xticks(range(1, 2*(max_depth-1)+1, 2))
    ax_signed.set_xticklabels(labels)
    ax_signed.set_ylabel("Cumulatieve Δ price (signed)")
    ax_signed.set_title("Signed cumulative streaks")

    ax_abs.axhline(0, linestyle="--", color="grey", alpha=0.6)
    ax_abs.set_xticks(range(1, 2*(max_depth-1)+1, 2))
    ax_abs.set_xticklabels(labels)
    ax_abs.set_ylabel("Cumulatieve Δ price (absolute)")
    ax_abs.set_title("Absolute cumulative streaks")

    if log_scale:
        ax_signed.set_yscale("symlog")
        ax_abs.set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.25, which="both")

    fig.suptitle("Cumulative streak payoff — empirical violins", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

def plot_price_and_timedelta_events(
    prices,
    event_times,
    streak_lengths,
    top_pct=0.05
):

    fig, (ax_p, ax_dt) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    events = np.arange(len(prices))
    deltas = np.diff(event_times)

    sc = ax_p.scatter(
        events, prices, c=streak_lengths, s=8
    )
    ax_p.set_ylabel("open / ask price")

    ax_dt.bar(events[1:], deltas)
    ax_dt.set_ylabel("Δt")

    # mark top X% streaks
    threshold = np.quantile(streak_lengths, 1 - top_pct)
    for i, s in enumerate(streak_lengths):
        if s >= threshold:
            ax_p.axvline(i, alpha=0.2)
            ax_p.annotate(
                f"streak {s}",
                (i, prices[i]),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=7
            )

    ax_dt.set_xlabel("event index")
    fig.colorbar(sc, ax=ax_p, label="streak length")

    fig.suptitle("Event-based prijs + latency structuur")

    return fig

def plot_extreme_streak_origins(
    prices,
    streak_lengths,
    percentile=0.95
):

    fig, ax = plt.subplots()

    threshold = np.quantile(streak_lengths, percentile)
    mask = streak_lengths >= threshold

    ax.scatter(
        np.where(mask)[0],
        prices[mask],
        c=streak_lengths[mask],
        s=40
    )

    ax.set_title(f"Startpunten streaks ≥ p{int(percentile*100)}")
    ax.set_xlabel("event index")
    ax.set_ylabel("price")

    return fig

def to_mdate(ts):
    import datetime as dt
    """Convert float timestamp(s) to matplotlib date numbers."""
    if isinstance(ts, (list, np.ndarray)):
        return np.array([mdates.date2num(dt.datetime.fromtimestamp(t)) for t in ts])
    return mdates.date2num(dt.datetime.fromtimestamp(ts))

def plot_top_streak_timeline_stacked(
    prices,
    event_times,
    streak_lengths,
    streak_start_idx,
    streak_end_idx,
    top_n=3,
    rect_height_px=5
):
    """
    Plot prices over time with top-N streaks shown as stacked small colored rectangles
    at the top of the plot. Overlapping streaks do not overwrite each other.
    """
    prices = np.asarray(prices)
    streak_lengths = np.asarray(streak_lengths)
    streak_start_idx = np.asarray(streak_start_idx)
    streak_end_idx = np.asarray(streak_end_idx)
    event_times = np.asarray(event_times)

    fig, ax = plt.subplots(figsize=(14, 6))

    # --- base price line ---
    ax.plot(to_mdate(event_times), prices, color="black", alpha=0.25, linewidth=0.7)

    # --- select top-N unique streak lengths ---
    unique_lengths = np.unique(streak_lengths)
    top_lengths = np.sort(unique_lengths)[-top_n:]

    base_cmap = colormaps.get_cmap("tab10")
    colors = base_cmap(np.linspace(0, 1, len(top_lengths)))
    color_map = {
        L: colors[i]
        for i, L in enumerate(top_lengths)
    }


    # --- y-position of stacked rectangles ---
    ylim = ax.get_ylim()
    y_top = ylim[1]
    rect_height = (rect_height_px / fig.dpi) * (ylim[1] - ylim[0])  # data units per pixel
    layers = []  # track end positions of rectangles in each layer

    for L, s, e in zip(streak_lengths, streak_start_idx, streak_end_idx):
        if L not in color_map:
            continue

        # find a free layer (no overlap)
        layer_idx = 0
        while layer_idx < len(layers):
            if s > layers[layer_idx]:
                break
            layer_idx += 1
        if layer_idx == len(layers):
            layers.append(e)
        else:
            layers[layer_idx] = e

        y_bottom = y_top - rect_height * (layer_idx + 1)
        y_rect_top = y_bottom + rect_height

        # draw rectangle
        rect = Rectangle(
            (to_mdate(event_times[s]), y_bottom),
            width=to_mdate(event_times[e]) - to_mdate(event_times[s]),
            height=rect_height,
            color=color_map[L],
            alpha=0.9
        )
        ax.add_patch(rect)

        # optional: start/end thin vertical lines
        ax.vlines(to_mdate(event_times[s]), ymin=ylim[0], ymax=ylim[1], color=color_map[L], alpha=0.2, linestyle="--", linewidth=1)
        ax.vlines(to_mdate(event_times[e]), ymin=ylim[0], ymax=ylim[1], color=color_map[L], alpha=0.2, linestyle="--", linewidth=1)

    # --- discrete legend ---
    handles = [Line2D([0], [0], color=color_map[L], lw=4, label=f"streak {L}") for L in top_lengths]
    ax.legend(handles=handles, title="Top streak lengths", loc="upper left")

    # --- format x-axis as dates ---
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()

    ax.set_ylabel("price")
    ax.set_title(f"Top {top_n} cumulative streaks (stacked timeline)")

    return fig

def plot_streak_sequences(streak_sequence, runs_by_length, run_stats):
    """
    Plot de nieuwe sequence analyses.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
# 1. Run length distribution per streak length
    ax1 = axes[0, 0]
    for s in sorted(runs_by_length.keys()):
        runs = runs_by_length[s]
        if runs:
            ax1.hist(runs, bins=range(1, max(runs) + 2), alpha=0.5, 
                    label=f"Str {s-1}", density=True)
    ax1.set_xlabel("Run length (consecutive occurrences)")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution of consecutive streak occurrences")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Heatmap van run lengths
    ax2 = axes[0, 1]
    max_streak = max(runs_by_length.keys()) if runs_by_length else 0
    max_run = 0
    for runs in runs_by_length.values():
        if runs:
            max_run = max(max_run, max(runs))
    
    heatmap_data = np.zeros((max_streak, max_run))
    for s, runs in runs_by_length.items():
        for run_len in runs:
            if s <= max_streak and run_len <= max_run:
                heatmap_data[s-1, run_len-1] += 1
    
    if heatmap_data.sum() > 0:
        im = ax2.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        ax2.set_xlabel("Run length")
        ax2.set_ylabel("Streak length")
        ax2.set_title("Heatmap: Streak length × Run length")
        plt.colorbar(im, ax=ax2)
    
    # 3. Transition matrix visualisatie
    ax3 = axes[1, 0]
    trans_dict, stationary = calculate_transition_matrix(streak_sequence)
    
    if trans_dict:
        streak_keys = sorted(trans_dict.keys())
        n = len(streak_keys)
        trans_matrix = np.zeros((n, n))
        
        for i, s_from in enumerate(streak_keys):
            for j, s_to in enumerate(streak_keys):
                trans_matrix[i, j] = trans_dict[s_from].get(s_to, 0)
        
        im = ax3.imshow(trans_matrix, aspect='auto', cmap='Blues')
        ax3.set_xlabel("To streak length")
        ax3.set_ylabel("From streak length")
        ax3.set_title("Transition probabilities")
        plt.colorbar(im, ax=ax3)
        
        # Annotate significant transitions (>0.1)
        for i in range(n):
            for j in range(n):
                if trans_matrix[i, j] > 0.1:
                    ax3.text(j, i, f"{trans_matrix[i, j]:.2f}", 
                            ha="center", va="center", color="white", fontsize=8)
    
    # 4. Autocorrelatie plot
    ax4 = axes[1, 1]
    autocorr = calculate_autocorrelation(streak_sequence, max_lag=20)
    
    if autocorr:
        lags = list(autocorr.keys())
        values = list(autocorr.values())
        ax4.bar(range(len(lags)), values)
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("Autocorrelation")
        ax4.set_title(f"Autocorrelation of streak lengths\nBurstiness: {calculate_burstiness(streak_sequence):.3f}")
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_ecdf_fan(price_moves, max_streak=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    streaks = sorted(price_moves.keys())
    if max_streak:
        streaks = [s for s in streaks if s <= max_streak]

    for s in streaks:
        arr = np.array(price_moves[s])
        if len(arr) < 20:
            continue
        x, y = ecdf(arr)
        lw = 1.5 + 0.15 * s          # langere streak → dikkere lijn
        alpha = min(0.5, 0.12 + 0.01 * s)
        ax.plot(x, y, alpha=alpha, lw=lw, label=f"Str {s}")

    ax.set_xlabel("Δ price")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF fan plot (regime bifurcation view)")
    ax.grid(alpha=0.3)

    return fig

def plot_streak_timeline_old(streak_sequence):
    """
    Plot streaks als een tijdlijn met kleurcodering.
    """
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Maak een cumulative sum om hoogte te bepalen
    y = 0
    for i, s in enumerate(streak_sequence):
        # Kleur op basis van lengte
        if s == 1:
            color = 'gray'
            height = 1
        elif s <= 3:
            color = 'blue'
            height = 2
        else:
            color = 'red'
            height = 3
        
        ax.bar(i, height, width=0.8, color=color, edgecolor='black')
        
        # Annotate streak length
        ax.text(i, height/2, str(s-1), ha='center', va='center', 
                fontsize=8, color='white' if s > 3 else 'black')
    
    ax.set_xlabel("Streak index")
    ax.set_ylabel("Streak magnitude")
    ax.set_title("Streak sequence timeline (gray=1, blue=2-3, red=4+)")
    ax.grid(axis='y', alpha=0.3)
    
    return fig

def plot_streak_timeline(streak_sequence):
    """
    Snelle plot van streaks als tijdlijn met kleurcodering.
    """
    streak_sequence = np.array(streak_sequence)
    indices = np.arange(len(streak_sequence))
    
    # Hoogtes en kleuren bepalen
    heights = np.ones_like(streak_sequence)
    colors = np.full_like(streak_sequence, 'gray', dtype=object)
    
    heights[streak_sequence <= 3] = 2
    heights[streak_sequence > 3] = 3
    
    colors[streak_sequence <= 3] = 'blue'
    colors[streak_sequence > 3] = 'red'
    
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.bar(indices, heights, width=0.8, color=colors, edgecolor='black')
    
    ax.set_xlabel("Streak index")
    ax.set_ylabel("Streak magnitude")
    ax.set_title("Streak sequence timeline (gray=1, blue=2-3, red=4+)")
    ax.grid(axis='y', alpha=0.3)
    
    return fig

def plot_interarrival_old(ax, iat):
    lam = poisson_lambda(iat)

    ax.hist(iat, bins=50, density=True, alpha=0.6, label="Empirical")

    xs = np.linspace(0, max(iat), 300)
    ax.plot(xs, lam * np.exp(-lam * xs), label=f"Exp fit λ={lam:.3f}")

    ax.set_title("Interarrival times")
    ax.set_xlabel("Δt")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()

def poisson_lambda(iat):
    # lambda is gewoon 1 / gemiddelde interarrival tijd
    if len(iat) == 0:
        return 0
    return 1 / np.mean(iat)

def plot_interarrival(iat, bins=50, figsize=(8, 5)):
    """
    Plot histogram van interarrival times met een exponentiële fit.
    Geeft de fig en ax terug.
    """
    if len(iat) == 0:
        raise ValueError("Interarrival times array is leeg")

    lam = poisson_lambda(iat)

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram van empirische data
    ax.hist(iat, bins=bins, density=True, alpha=0.6, label="Empirical")

    # Exponentiële fit
    xs = np.linspace(0, max(iat), 300)
    ax.plot(xs, lam * np.exp(-lam * xs), label=f"Exp fit λ={lam:.3f}", color="red")

    ax.set_title("Interarrival times")
    ax.set_xlabel("Δt")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()

    return fig

def plot_histogram(bins):
    fig, ax = plt.subplots(figsize=(12, 4))
    print(len(bins))
    xs = sorted(bins.keys())
    ys = [bins[x] for x in xs]

    ax.bar(xs, ys, width=0.8)
    for x, y in zip(xs, ys):
        ax.text(x, y, f"N={y}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Streak length")
    ax.set_ylabel("Count")
    ax.set_title("Streak length distribution")
    ax.grid(alpha=0.3)
    return fig

def plot_extremes(price_moves):
    fig, axes = plt.subplots(len(price_moves), 3, figsize=(22, 4 * len(price_moves)))

    for i, (s, data) in enumerate(sorted(price_moves.items())):
        p90 = np.percentile(data, 90)
        ext = [x for x in data if x >= p90]

        if len(ext) < 5:
            continue

        axes[i, 0].hist(ext, bins=30)
        axes[i, 0].set_title(f"Str {s-1} p90–p100")

        ps = [90, 95, 99, 99.9, 100]
        vals = [np.percentile(data, p) for p in ps]

        axes[i, 2].barh(ps, vals)
        for p, v in zip(ps, vals):
            axes[i, 2].text(v, p, f"{v:.2e}", va="center")

    plt.tight_layout()
    return fig

def plot_quantile_heatmap(price_moves):
    quantiles = [50, 75, 90, 95, 99]
    streaks = sorted(price_moves.keys())

    mat = np.zeros((len(streaks), len(quantiles)))

    for i, s in enumerate(streaks):
        arr = np.array(price_moves[s])
        if len(arr) == 0:
            continue
        mat[i] = np.percentile(arr, quantiles)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto", cmap="magma")

    ax.set_xticks(range(len(quantiles)))
    ax.set_xticklabels([f"p{q}" for q in quantiles])
    ax.set_yticks(range(len(streaks)))
    ax.set_yticklabels([f"Str {s}" for s in streaks])

    plt.colorbar(im, ax=ax, label="Δ price")
    ax.set_title("Quantile heatmap (risk explosion zones)")

    return fig

def plot_interarrival_logbins_old(interarrival):
    iat = np.asarray(interarrival)
    bins = np.logspace(np.log10(iat.min()), np.log10(iat.max()), 30)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(iat, bins=bins, density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Δt")
    ax.set_ylabel("Density")
    ax.set_title("Inter-arrival time (log-binned)")
    ax.grid(alpha=0.3)

    return fig

def plot_interarrival_logbins(interarrival, bins_count=30, figsize=(8, 4)):
    iat = np.asarray(interarrival)
    if len(iat) == 0:
        raise ValueError("Interarrival array is leeg")

    bins = np.logspace(np.log10(iat.min()), np.log10(iat.max()), bins_count)

    fig, ax = plt.subplots(figsize=figsize)
    
    # histogram met density=True
    counts, bin_edges, patches = ax.hist(iat, bins=bins, density=True, alpha=0.7, color="skyblue", edgecolor="black")

    # percentages berekenen per bar
    total = counts.sum()
    percentages = counts / total * 100

    # annotaties toevoegen
    for count, perc, patch in zip(counts, percentages, patches):
        height = patch.get_height()
        if height > 0:  # alleen annoteren als er een bar is
            ax.text(
                patch.get_x() + patch.get_width()/2, 
                height, 
                f"{perc:.1f}%\n({count:.2f})", 
                ha='center', va='bottom', fontsize=8
            )

    ax.set_xscale("log")
    ax.set_xlabel("Δt")
    ax.set_ylabel("Density")
    ax.set_title("Inter-arrival time (log-binned)")
    ax.grid(alpha=0.3, which="both")

    return fig

def plot_main(args, price_moves, up, down, bins, csv, ticks, total_streaks):
    all_streaks = sorted(price_moves.keys())
    n_violins = len(all_streaks)
    n_cols = MAX_VIOLINS
    n_rows = int(np.ceil(n_violins / n_cols))

    fig = plt.figure(figsize=(22, 5 + 4 * n_rows))
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=[2] + [1]*n_rows)

    fig.text(
        0.01, 0.98,
        f"Source: {csv}",
        fontsize=9,
        color="gray",
        ha="left",
        va="top"
    )

    # -------- Top-left: scatter (2/3) --------
    ax_scatter = fig.add_subplot(gs[0, :2])
    for s, data in price_moves.items():
        ax_scatter.scatter(
            np.full(len(data), s),
            data,
            s=6,
            alpha=0.25
        )
    ax_scatter.set_title("All streak deltas")
    ax_scatter.set_xlabel("Streak length")
    ax_scatter.set_xticks(sorted(price_moves.keys()))
    ax_scatter.set_ylabel("Δ price")
    ax_scatter.grid(alpha=0.3)
    if not args.nology:
        ax_scatter.set_yscale("log")

## !!TODO!! DE HISOGRAM BINS STARTEN VANAF 0, ZOU VANAF 1 MOETEN ZIJN

    # -------- Top-right: histogram (1/3) --------
    ax_hist = fig.add_subplot(gs[0, 2:4])
    xs = sorted(bins.keys())
    ys = np.array([bins[x] for x in xs])
    perc = ys / ys.sum() * 100
    ax_hist.bar(xs, ys, width=0.8)
    ax_hist.set_xticks(xs)          # exacte tick-positions
    ax_hist.set_xticklabels(xs)     # labels = streak lengths

    ax_hist.set_xlim(0.5, max(xs) + 0.5)
    for x, y, p in zip(xs, ys, perc):
        ax_hist.text(x, y, f"{p:.2f}%", ha="center", va="bottom", fontsize=6)
    ax_hist.set_title("Streak distribution")
    ax_hist.set_xlabel("Streak length")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)
    ax_hist.legend([f"Total streaks: {total_streaks}\nTotal ticks: {ticks}"], frameon=False)

# !! TODO / ERROR !! ER WORDT EEN VIOLIN Str = 0 getoond, die bestaat gewoon niet, maw alle violins zouden 1 (kwa index) moeten opschuiven
# TODO er worden log scales in de y-richting weergegeven, gewone scales van maken, en indien mogelijk ook het font en grootte wat aanpassen

    # -------- violins vanaf tweede rij --------
    axes = []
    for i in range(n_rows):
        row_axes = []
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= n_violins:
                break
            ax = fig.add_subplot(gs[i + 1, j])
            row_axes.append(ax)
        axes.append(row_axes)

    for i, s in enumerate(all_streaks):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]

        if ax is None:
            continue

        x = 1
        def shifted(data):
            return np.asarray(data)

        # -------- violins UP/DOWN --------
        if args.nosplit:
            v = ax.violinplot(
                shifted(price_moves[s]),
                positions=[x],
                showextrema=False
            )
            v['bodies'][0].set_alpha(0.6)
        else:
            if s in up:
                vu = ax.violinplot(
                    shifted(up[s]),
                    positions=[x],
                    showextrema=False
                )
                bu = vu['bodies'][0]
                bu.set_facecolor("steelblue")
                bu.set_alpha(0.6)
                clip_violin_half(bu, x, "left")
                annotate_stats(ax, x, stats_dict(up[s]), "left", "steelblue", violin=bu)

                if not args.noregimes:
                    arr_up = np.array(up[s])
                    w = violin_half_width(bu, x)
                    reg = compute_regimes(arr_up)

                    draw_regime_background(ax, x, w, 0, reg["p90"], "left", "lightgreen", 0.6)
                    draw_regime_background(ax, x, w, reg["p90"], reg["p99"], "left", "khaki", 0.6)
                    draw_regime_background(ax, x, w, reg["p99"], reg["max"], "left", "lightcoral", 0.6)

            if s in down:
                vd = ax.violinplot(
                    shifted(down[s]),
                    positions=[x],
                    showextrema=False
                )
                bd = vd['bodies'][0]
                bd.set_facecolor("darkorange")
                bd.set_alpha(0.6)
                clip_violin_half(bd, x, "right")
                annotate_stats(ax, x, stats_dict(down[s]), "right", "darkorange", violin=bd)

                if not args.noregimes:
                    arr_down = np.array(down[s])
                    w = violin_half_width(bd, x)
                    reg = compute_regimes(np.array(down[s]))

                    draw_regime_background(ax, x, w, 0, reg["p90"], "right", "lightgreen", 0.6)
                    draw_regime_background(ax, x, w, reg["p90"], reg["p99"], "right", "khaki", 0.6)
                    draw_regime_background(ax, x, w, reg["p99"], reg["max"], "right", "lightcoral", 0.6)

        ax.set_frame_on(False)
        # eventueel spines volledig weg
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(left=True, bottom=True, labelbottom=True, labelleft=True)
        ax.grid(alpha=0.25)

        # -------- ECDF --------
        #if not args.noecdf:
        #    if s in up:
        #        x_ecdf, _ = ecdf(up[s])
        #        ax.plot(np.full_like(x_ecdf, x - 0.15), x_ecdf, alpha=0.25)
        #    if s in down:
        #        x_ecdf, _ = ecdf(down[s])
        #        ax.plot(np.full_like(x_ecdf, x + 0.15), x_ecdf, alpha=0.25)

        # -------- ECDF --------
        if not args.noecdf:
            if s in up and len(up[s]) > 0:
                x_ecdf, y_ecdf = ecdf(up[s])
                ax.plot(np.full_like(x_ecdf, x - 0.15), y_ecdf, alpha=0.25, color="steelblue")
            if s in down and len(down[s]) > 0:
                x_ecdf, y_ecdf = ecdf(down[s])
                ax.plot(np.full_like(x_ecdf, x + 0.15), y_ecdf, alpha=0.25, color="darkorange")

        # -------- labels & grid --------
        total = sum(bins.values())
        n = bins[s]
        pct = 100 * n / total

        up_n = len(up.get(s, []))
        down_n = len(down.get(s, []))
        up_pct = 100 * up_n / n if n else 0
        down_pct = 100 * down_n / n if n else 0

        ax.set_xticklabels([
            f"Str {s}\n"
            f"N={n} ({pct:.2f}%)\n"
            f"↑ {up_n} ({up_pct:.0f}%) | ↓ {down_n} ({down_pct:.0f}%)"
        ])

        ax.set_xticks([1])
#        ax.set_xticklabels([f"Str {s-1} (N={bins.get(s,0)})"])

        if col == 0:
            ax.set_ylabel("Δ price")
        else:
            ax.set_ylabel("")
#            ax.tick_params(labelleft=False)

#        ax.ticklabel_format(
#            style='plain',
#            axis='y',
#            useOffset=False
#        )
#        ax.yaxis.set_major_formatter(
#            FuncFormatter(lambda y, _: f"{y:.6f}".rstrip('0').rstrip('.'))
#        )
        ax.grid(alpha=0.25)

        all_data = []
        if s in up and len(up[s]) > 0:
            all_data.extend(up[s])
        if s in down and len(down[s]) > 0:
            all_data.extend(down[s])

        if not args.nology:
            ax.set_yscale("log")
        else:
            if all_data:
                ax.set_ylim(0, np.percentile(all_data, 99) * 1.1)

        #ymin = min(all_data)
        #ymax = max(all_data)
        #pad = 0.08 * (ymax - ymin)

        #ax.set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(f"Streak analysis", fontsize=16)

    legend_patches = [
        Patch(facecolor='steelblue', edgecolor='k', label='UP'),
        Patch(facecolor='darkorange', edgecolor='k', label='DOWN')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, frameon=False, fontsize=8)
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig



# ------------------------------------------------------------
# TODO x.7 – Run-conditioned transition matrix
# ------------------------------------------------------------

def run_conditioned_transitions(streak_lengths, max_run=10, max_next=10):
    """
    Returns:
    transitions[L0][run_len][L1] = probability
    """


    # transitions[L0][run_len] = Counter(next_L)
    raw = defaultdict(lambda: defaultdict(Counter))

    i = 0
    n = len(streak_lengths)

    while i < n - 1:
        L0 = streak_lengths[i]
        run_len = 1

        # count how many consecutive L0
        while i + run_len < n and streak_lengths[i + run_len] == L0:
            run_len += 1

        next_idx = i + run_len
        if next_idx < n:
            L1 = streak_lengths[next_idx]
            if run_len <= max_run and L1 <= max_next:
                raw[L0][run_len][L1] += 1

        i += run_len

    # normalize to probabilities
    transitions = defaultdict(dict)

    for L0, runs in raw.items():
        for run_len, cnt in runs.items():
            total = sum(cnt.values())
            transitions[L0][run_len] = {
                L1: v / total * 100.0
                for L1, v in cnt.items()
            }

    return transitions

# ------------------------------------------------------------
# TODO x.8 – Run-conditioned grouped bar chart
# ------------------------------------------------------------

def plot_run_transition_bars(transitions, base_streak,
                             max_run=8, max_next=6):
    """
    transitions: output of run_conditioned_transitions
    base_streak: L0 to visualise (e.g. 1 or 4)
    """

    if base_streak not in transitions:
        raise ValueError("Base streak not in transitions")
    
    runs = sorted(
        r for r in transitions[base_streak].keys()
        if r <= max_run
    )

    next_streaks = list(range(1, max_next + 1))

    bar_width = 0.8 / len(next_streaks)
    x = np.arange(len(runs))

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, L1 in enumerate(next_streaks):
        heights = [
            transitions[base_streak][r].get(L1, 0.0)
            for r in runs
        ]
        ax.bar(
            x + i * bar_width,
            heights,
            width=bar_width,
            label=f"→ streak {L1}",
            alpha=0.8
        )

    ax.set_xticks(x + bar_width * (len(next_streaks) - 1) / 2)
    ax.set_xticklabels([f"{r}× streak {base_streak}" for r in runs])

    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Run length of previous streaks")
    ax.set_title(
        f"Conditional transitions after streak {base_streak}"
    )

    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.3)

    return fig

# ---------------------------
# 1. Streak duration histogram in time
# ---------------------------
def plot_streak_duration_histogram_old(interarrival, streak_sequence):
    streak_times = []
    interarrival = np.array(interarrival)  # <--- converteer naar np.array

    durations = []
    idx = 0
    for l in streak_sequence:
        if l > 1:
            duration = interarrival[idx:idx+l-1].sum()  # nu werkt .sum()
        else:
            duration = 0
        durations.append(duration)
        idx += l

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(streak_times, bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel("Streak duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram van streak duur in seconden")
    return fig


def plot_streak_duration_histogram(interarrival, streak_sequence, bins=50, figsize=(6,4)):
    """
    Histogram van de duur van streaks in seconden.
    """
    interarrival = np.asarray(interarrival)
    durations = []
    idx = 0

    for l in streak_sequence:
        if l > 1:
            # neem de som van interarrivals in de streak
            # beveilig tegen index errors
            if idx + l - 1 <= len(interarrival):
                duration = interarrival[idx:idx+l-1].sum()
            else:
                duration = interarrival[idx:].sum()
        else:
            duration = 0
        durations.append(duration)
        idx += l

    durations = np.array(durations)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(durations, bins=bins, color='skyblue', edgecolor='black')
    ax.set_xlabel("Streak duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram van streak duur in seconden")
    ax.grid(alpha=0.3)

    return fig

# ---------------------------
# 2. Cumulative price move per streak length
# ---------------------------
def plot_cumulative_price_move_old(price_sequence, streak_sequence):
    idx = 0
    streak_moves = []
    streak_lengths = []
    for l in streak_sequence:
        start = price_sequence[idx]
        end_idx = idx + l -1
        if end_idx >= len(price_sequence):
            end_idx = len(price_sequence)-1
        end = price_sequence[end_idx]
        streak_moves.append(abs(end-start))
        streak_lengths.append(l)
        idx += l
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(streak_lengths, streak_moves, alpha=0.6)
    ax.set_xlabel("Streak length")
    ax.set_ylabel("Cumulative price move")
    ax.set_title("Cumulative price move per streak length")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig

def plot_cumulative_price_move_new(price_sequence, streak_sequence):
    idx = 0
    streak_moves = []
    streak_lengths = []

    n = len(price_sequence)

    for l in streak_sequence:
        if idx >= n:
            break   # niks meer te mappen

        start = price_sequence[idx]

        end_idx = idx + l - 1
        if end_idx >= n:
            end_idx = n - 1

        end = price_sequence[end_idx]

        streak_moves.append(abs(end - start))
        streak_lengths.append(l)

        idx += l

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(streak_lengths, streak_moves, alpha=0.6)
    ax.set_xlabel("Streak length")
    ax.set_ylabel("Cumulative price move")
    ax.set_title("Cumulative price move per streak length")
    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig

def plot_cumulative_price_move(price_sequence, streak_sequence):
    prices = np.asarray(price_sequence)
    deltas = np.abs(np.diff(prices))  # price moves

    idx = 0
    streak_moves = []
    streak_lengths = []

    n = len(deltas)

    for l in streak_sequence:
        if idx >= n:
            break

        end_idx = idx + l
        if end_idx > n:
            end_idx = n

        move = deltas[idx:end_idx].sum()

        streak_moves.append(move)
        streak_lengths.append(l)

        idx += l

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(streak_lengths, streak_moves, alpha=0.6)

    ax.set_xlabel("Streak length")
    ax.set_ylabel("Cumulative price move (Σ|Δp|)")
    ax.set_title("Cumulative price move per streak length")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # Forceer normale notatie ipv scientific
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.ticklabel_format(style='plain', axis='y')

    return fig

# ---------------------------
# 3. Delta price vs streak length
# ---------------------------
def plot_delta_vs_streak_old(price_sequence, streak_sequence):
    idx = 0
    deltas = []
    lengths = []
    for l in streak_sequence:
        start = price_sequence[idx]
        end_idx = idx + l -1
        if end_idx >= len(price_sequence):
            end_idx = len(price_sequence)-1
        end = price_sequence[end_idx]
        delta = abs(end-start)
        deltas.append(delta)
        lengths.append(l)
        idx += l
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(lengths, deltas, alpha=0.6)
    ax.set_xlabel("Streak length")
    ax.set_ylabel("Delta price")
    ax.set_title("Delta price vs streak length")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig

def plot_delta_vs_streak_old2(price_sequence, streak_sequence):
    prices = np.asarray(price_sequence)
    deltas = np.abs(np.diff(prices))   # elementaire price moves

    idx = 0
    streak_moves = []
    streak_lengths = []

    n = len(deltas)

    for l in streak_sequence:
        if idx >= n:
            break

        end_idx = idx + l
        if end_idx > n:
            end_idx = n

        move = deltas[idx:end_idx].sum()

        streak_moves.append(move)
        streak_lengths.append(l)

        idx += l

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(streak_lengths, streak_moves, alpha=0.6)

    ax.set_xlabel("Streak length")
    ax.set_ylabel("Cumulative |Δ price|")
    ax.set_title("Price excursion vs streak length")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    return fig

def plot_delta_vs_streak(price_sequence, streak_sequence, figsize=(6,4)):
    prices = np.asarray(price_sequence)
    deltas = np.abs(np.diff(prices))   # elementaire price moves

    idx = 0
    streak_moves = []
    streak_lengths = []

    n = len(deltas)

    for l in streak_sequence:
        if idx >= n:
            break
        end_idx = min(idx + l, n)
        move = deltas[idx:end_idx].sum()
        streak_moves.append(move)
        streak_lengths.append(l)
        idx += l

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(streak_lengths, streak_moves, alpha=0.6, color="teal")

    ax.set_xlabel("Streak length")
    ax.set_ylabel("Cumulative |Δ price|")
    ax.set_title("Price excursion vs streak length")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # Forceer normale notatie ipv scientific notation
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.ticklabel_format(style='plain', axis='y')

    return fig

# ---------------------------
# 4. Delta price vs interarrival
# ---------------------------
def plot_delta_vs_interarrival_old(price_sequence, streak_sequence, interarrival):
    idx = 0
    deltas = []
    delta_times = []
    for l in streak_sequence:
        start = price_sequence[idx]
        end_idx = idx + l -1
        if end_idx >= len(price_sequence):
            end_idx = len(price_sequence)-1
        end = price_sequence[end_idx]
        delta = abs(end-start)
        deltas.append(delta)
        delta_time = interarrival[idx:idx+l-1].sum() if l>1 else interarrival[idx] if idx < len(interarrival) else 0
        delta_times.append(delta_time)
        idx += l
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(delta_times, deltas, alpha=0.6)
    ax.set_xlabel("Total interarrival time (s)")
    ax.set_ylabel("Delta price")
    ax.set_title("Delta price vs interarrival time")
    return fig

def plot_delta_vs_interarrival_old(price_sequence, streak_sequence, interarrival):
    prices = np.asarray(price_sequence)
    deltas = np.abs(np.diff(prices))          # |Δp|
    interarr = np.asarray(interarrival)       # Δt per move

    idx = 0
    streak_moves = []
    streak_times = []

    n = min(len(deltas), len(interarr))

    for l in streak_sequence:
        if idx >= n:
            break

        end_idx = idx + l
        if end_idx > n:
            end_idx = n

        move = deltas[idx:end_idx].sum()
        duration = interarr[idx:end_idx].sum()

        streak_moves.append(move)
        streak_times.append(duration)

        idx += l

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(streak_times, streak_moves, alpha=0.6)

    ax.set_xlabel("Cumulatieve interarrival time (s)")
    ax.set_ylabel("Cumulatieve |Δ price|")
    ax.set_title("Price excursion vs time spent in streak")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    return fig

def plot_delta_vs_interarrival(price_sequence, streak_sequence, interarrival, figsize=(6,4)):
    """
    Plot cumulatieve price moves versus cumulatieve interarrival time per streak.
    """
    prices = np.asarray(price_sequence)
    deltas = np.abs(np.diff(prices))          # |Δp|
    interarr = np.asarray(interarrival)       # Δt per move

    idx = 0
    streak_moves = []
    streak_times = []

    n = min(len(deltas), len(interarr))

    for l in streak_sequence:
        if idx >= n:
            break
        end_idx = min(idx + l, n)
        move = deltas[idx:end_idx].sum()
        duration = interarr[idx:end_idx].sum()
        streak_moves.append(move)
        streak_times.append(duration)
        idx += l

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(streak_times, streak_moves, alpha=0.6, color="purple")

    ax.set_xlabel("Cumulatieve interarrival time (s)")
    ax.set_ylabel("Cumulatieve |Δ price|")
    ax.set_title("Price excursion vs time spent in streak")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # Forceer normale notatie ipv scientific
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.ticklabel_format(style='plain', axis='y')

    return fig

def plot_top_cumulative_streaks_with_interarrival(event_times, streak_sequence, interarrival, top_n=5):
    """
    Plots top N cumulatieve streaks over time and the corresponding cumulative interarrival times.
    
    Parameters
    ----------
    event_times : list or np.array
        Timestamps of each tick/event.
    streak_sequence : list or np.array
        List of streak lengths.
    interarrival : list or np.array
        List of interarrival times between ticks.
    top_n : int
        Number of top cumulative streaks to highlight.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    event_times = np.asarray(event_times)
    streak_array = np.asarray(streak_sequence)
    interarrival = np.asarray(interarrival)

    # Map streaks to start times and cumulative interarrival
    streak_times = []
    streak_lengths = []
    streak_interarrivals = []

    idx = 0
    n = min(len(event_times), len(interarrival))
    for l in streak_array:
        if idx >= n:
            break
        streak_times.append(event_times[idx])
        streak_lengths.append(l)
        duration = interarrival[idx:min(idx+l, n)].sum()
        streak_interarrivals.append(duration)
        idx += l

    streak_times = np.asarray(streak_times)
    streak_lengths = np.asarray(streak_lengths)
    streak_interarrivals = np.asarray(streak_interarrivals)

    # Cumulatieve streaks
    cumulative_streaks = np.cumsum(streak_lengths)
    top_idxs = np.argsort(cumulative_streaks)[-top_n:]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Boven: top N cumulatieve streaks
    ax1.scatter(streak_times, streak_lengths, alpha=0.2, color='grey', label="all streaks")
    ax1.scatter(streak_times[top_idxs], streak_lengths[top_idxs], color='red', alpha=0.8, label=f"Top {top_n} cumulative")
    for idx in top_idxs:
        ax1.axvline(streak_times[idx], color='blue', linestyle='--', alpha=0.7)
        ax1.text(streak_times[idx], streak_lengths[idx]+0.5, f"{streak_lengths[idx]}", color='blue',
                 rotation=90, verticalalignment='bottom', fontsize=8)
    ax1.set_ylabel("Streak length")
    ax1.set_title(f"Top {top_n} cumulative streaks over time")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    # Onder: interarrival per streak
    ax2.scatter(streak_times, streak_interarrivals, alpha=0.5, color='green')
    ax2.set_ylabel("Cumulative interarrival (s)")
    ax2.set_xlabel("Time")
    ax2.set_title("Cumulative interarrival per streak")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig

# ---------------------------
# 5. Autocorrelatie van streak lengths
# ---------------------------
def plot_streak_autocorrelation(streak_sequence, max_lag=100):
    streak_seq = np.array(streak_sequence)
    mean = streak_seq.mean()
    var = streak_seq.var()
    n = len(streak_seq)
    acf = []
    for lag in range(1, max_lag+1):
        cov = np.sum((streak_seq[:n-lag]-mean)*(streak_seq[lag:]-mean))/(n-lag)
        acf.append(cov/var)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(1,max_lag+1), acf, color='skyblue')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelatie van streak lengths")
    return fig

# ---------------------------
# 6. Top N% streaks per dag/uur
# ---------------------------
def plot_top_streaks_over_time_old(timestamps, streak_sequence, top_percent=5):
    streak_array = np.array(streak_sequence)
    threshold = np.percentile(streak_array, 100-top_percent)
    top_idxs = np.where(streak_array>=threshold)[0]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(timestamps[top_idxs], streak_array[top_idxs], color='red')
    ax.set_xlabel("Tick index")
    ax.set_ylabel("Streak length")
    ax.set_title(f"Top {top_percent}% streaks over time")
    return fig

def plot_top_streaks_over_time_old(timestamps, streak_sequence, top_percent=5):
    timestamps = np.asarray(timestamps)
    streak_array = np.asarray(streak_sequence)

    streak_times = []
    streak_lengths = []

    idx = 0
    n = len(timestamps)

    for l in streak_array:
        if idx >= n:
            break

        streak_times.append(timestamps[idx])   # starttijd van streak
        streak_lengths.append(l)

        idx += l

    streak_times = np.asarray(streak_times)
    streak_lengths = np.asarray(streak_lengths)

    threshold = np.percentile(streak_lengths, 100 - top_percent)
    top_mask = streak_lengths >= threshold

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(
        streak_times,
        streak_lengths,
        alpha=0.2,
        color="grey",
        label="all streaks"
    )

    ax.scatter(
        streak_times[top_mask],
        streak_lengths[top_mask],
        color="red",
        alpha=0.8,
        label=f"top {top_percent}%"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Streak length")
    ax.set_title(f"Top {top_percent}% streaks over time")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    return fig

def plot_top_streaks_over_time(timestamps, streak_sequence, top_percent=5, figsize=(8,4)):
    """
    Plot van streaks over tijd, met top X% geaccentueerd.
    """
    timestamps = np.asarray(timestamps)
    streak_array = np.asarray(streak_sequence)

    streak_times = []
    streak_lengths = []

    idx = 0
    n = len(timestamps)

    for l in streak_array:
        if idx >= n:
            break
        streak_times.append(timestamps[idx])   # starttijd van streak
        streak_lengths.append(l)
        idx += l

    streak_times = np.array([np.datetime64(int(ts), 's') for ts in streak_times])
    streak_lengths = np.asarray(streak_lengths)

    threshold = np.percentile(streak_lengths, 100 - top_percent)
    top_mask = streak_lengths >= threshold

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        streak_times,
        streak_lengths,
        alpha=0.2,
        color="grey",
        label="all streaks"
    )

    ax.scatter(
        streak_times[top_mask],
        streak_lengths[top_mask],
        color="red",
        alpha=0.8,
        label=f"top {top_percent}%"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Streak length")
    ax.set_title(f"Top {top_percent}% streaks over time")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Formatteer x-as als datum/tijd
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))

    fig.autofmt_xdate()  # draait labels automatisch
    return fig


import pandas as pd

def plot_top_streaks_with_trend(timestamps, streak_sequence, top_percent=5, rolling_window='1D'):
    fig = plot_top_streaks_over_time(timestamps, streak_sequence, top_percent=top_percent)

    # Prepare DataFrame
    streak_times = np.array(timestamps)
    streak_lengths = np.array(streak_sequence)
    df = pd.DataFrame({'time': pd.to_datetime(streak_times, unit='s'), 'streak': streak_lengths})

    # Rollling mean / sum per day
    rolling_mean = df.set_index('time')['streak'].rolling(rolling_window).mean()
    rolling_max = df.set_index('time')['streak'].rolling(rolling_window).max()

    ax.plot(rolling_mean.index, rolling_mean.values, color='blue', label='rolling mean')
    ax.plot(rolling_max.index, rolling_max.values, color='green', label='rolling max')
    ax.legend(frameon=False)

    return fig

# ---------------------------
# 7. Leader-laggard plots
# ---------------------------
def plot_leader_laggard(streak_sequence):
    s = np.array(streak_sequence)
    leaders = s[:-1]
    laggards = s[1:]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(leaders, laggards, alpha=0.6)
    ax.set_xlabel("Leader streak length")
    ax.set_ylabel("Laggard streak length")
    ax.set_title("Leader-Laggard streak plot")
    return fig

# ---------------------------
# 8. Density plots van price moves
# ---------------------------
def plot_price_move_density(price_sequence):
    deltas = np.abs(np.diff(price_sequence))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(deltas, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    ax.set_xlabel("Delta price")
    ax.set_ylabel("Density")
    ax.set_title("Density plot van price moves")
    return fig

def export_json(streak_moves, streak_bins, out_path, source, ticks, total_streaks, inter, up, down, streak_sequence=None):

    # ANALYZE STREAK RUNS & CORRELATION MATRIX DUMPEN

    iat = np.array(inter)
    lam = poisson_lambda(iat)
    ks = poisson_ks_stat(iat)

    p90s = {s: np.percentile(v, 90) for s, v in streak_moves.items()}

    # Bereken de timing stats
    timing_data = calculate_streak_timing_stats(event_times, streak_sequence, streak_start_idx, streak_end_idx)

    export = {
        "_meta": {
            "source_file": source,
            "total_ticks": ticks,
            "total_streaks": total_streaks,
            # interarrival section is poisson data of the interarrival time
            "interarrival": {
                "lambda": float(lam),
                "ks_stat": float(ks),
                "n_samples": len(iat),
            },
            # NIEUW: Streak timing data
            "streak_timing_ms": timing_data
        },

#        "_hftsettings": {
#            "core_zone": [0, r["p90"]],                 # TODO << is this correct r["p90"] van streak 1 ? dus p9._streak1
#            "tail_zone": [r["p90"], r["max"]],          # TODO << is this correct r["p90"] van streak 1  en r["max"] van streak1
#            "tail_mass": mass["burst"] + mass["extreme"],
        # TODO ADD THESE BELOW
        #    "hit_rate_core": ??,
        #    "hit_rate_tail": ??,
#            "expected_tail_delta": float(arr[arr > r["p90"]].mean()),
#        },

        "_gridsettings": {
            "global_suggested_width": suggest_global_grid_width(streak_moves, list(streak_moves.keys()))
        #    "core_zone" [p90_streak1, inf],
        }
    }

    run_stats = None

    if streak_sequence:
        runs_by_length, run_stats = analyze_streak_runs(streak_sequence)
        trans_dict, stationary = calculate_transition_matrix(streak_sequence)
        autocorr = calculate_autocorrelation(streak_sequence)
        
        export["_meta"]["sequence_analysis"] = {
            "total_sequences": len(streak_sequence),
            "burstiness_index": calculate_burstiness(streak_sequence),
            "autocorrelation": autocorr,
            "stationary_distribution": stationary
        }

        export["_meta"]["transition_matrix"] = trans_dict

    for s, data in streak_moves.items():
        arr = np.array(data)
        r = compute_regimes(arr)
        mass = calculate_mass_ratios(arr, r["p90"], r["p99"])

        asym = None
        if s in up and s in down and (len(up[s]) > 0 and len(down[s]) > 0):
            asym = directional_asymmetry(up[s], down[s])

        accel = None
        if (s + 1) in p90s:
            accel = p90s[s + 1] / p90s[s]

        export[s-1] = {
            "count": streak_bins.get(s, 0),
            # TODO: ADD percent count
            # TODO: ADD Up_count & percent
            # TODO: ADD Down count & percent
            "stats": {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "mad": float(mad(arr)),
                "p25": r["p25"],
                "p50": r["p50"],
                "p90": r["p90"],
                "p99": r["p99"],
                "p99.9": r["p99.9"],
            },
            "grid": {
                "suggested_width": suggested_grid_width(arr),
                # TODO: "suggested_width_method_2": suggest_grid_width2(arr),
                "zone": [0, r["p90"]]
            },
            "burst": {
                "zone": [r["p90"], r["p99"]]
            },
            "extreme": {
                "zone": [r["p99"], r["max"]]
            },

            # tail mask ratio: tail_mass = P(delta > p90) | extreme_mass = P(delta > p99)
            # Next is regime mass ratio for each streak
            #  "normal": % below p90,
            #  "burst": % p90–p99,
            #  "extreme": % above p99
            "mass": mass,

            # Directrional asymmetry index (mean_up - mean_down) / (mean_up + mean_down) 
            "dir_assymmetry": asym, 

            # Quantile acceleration accel(s) = p90(s+1) / p90(s) etc
            "quantile_acceleration_p90": accel,

            # Stability score: MAD / median | Low = regime-stable | High = structurally unstable 
            "stability": stability_score(arr),

            # Entropy of deltas | Shannon entropy op binned deltas per streak. | lage entropy → voorspelbaar | hoge entropy → noise-dominant |  use function entropy_of_deltas
            "entropy": entropy_of_deltas(arr),
        }
        if run_stats:
            if s in run_stats:
                export[s-1]["runs"] = run_stats[s]

    # TODO: fix next thing somewhere in the json
    # continuations = pa.continuation_stats(streak_seq, pm)
    # print(continuations)

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"JSON exported to {out_path}")

### EXTRA PLOTS 

# 1.De "Fee-Adjusted Break-Even Ratio" (De n=3 Validator)
# Waarom: Je wilt oogsten op n=3
# n=3, Δ, n=3, De Metric:
# Maar de ongerealiseerde winst bij  moet groter zijn dan 2x Taker fee (0.05%) + 1x Maker fee (0.02%) = ~0.12% van je 
# positie. Als de gemiddelde  price van  kleiner is dan dat, gooi je geld weg.
# Bereken per streak-lengte de ratio van de netto winst (na fees) ten opzichte van de bruto winst.

def calculate_fee_adjusted_edge(price_moves, fee_rate_taker=0.0004, fee_rate_maker=0.0002):
    """
    berekent of een streak winstgevend is na fees.
    We gaan uit van 1 maker entry, en 1 taker entry (de harvest).
    """
    edge_stats = {}
    for s, deltas in price_moves.items():
        arr = np.array(deltas)
        # aproximeer de gemiddelde instapprijs (we gebruiken de mediaan van de delta als proxy)
        # In een echte bot pak je de absoulte prijs, maar voor relatieve sterkte is dit voldoende.
        median_delta = np.median(arr)

        # Fee kosten als benadering van percentage van de move
        total_fee_cost = median_delta * (fee_rate_maker + fee_rate_taker)

        net_profit = median_delta - total_fee_cost
        edge_ratio = net_profit / median_delta if median_delta > 0 else 0

        edge_stats[s] = {
            "median_gross": float(median_delta),
            "est_fee_cost": float(total_fee_cost),
            "median_net": float(net_profit),
            "edge_ratio": float(edge_ratio),
            "is_profitable_after_fees": bool(edge_ratio > 0)
        }
    return edge_stats

# 2. Maximum Adverse Excursion (MAE) per Streak (De Mode 8 Fallback Validator)
#
# Waarom: In Mode 8 zet je Limit orders als fallback. Als een breakout faalt, zakt de prijs in je fallback grid. Hoe 
# diep zakt hij gemiddeld door je grid voordat hij weer omhoog gaat? Dit bepaalt exact hoeveel DCA levels je nodig 
# hebt.
# De Metric: Meet de maximale terugtrekking (retrace) tijdens een streak, voordat de streak eindigt.

def calculate_mae_per_streak(prices, event_times, streak_sequence, streak_start_idx, streak_end_idx):
    """
    Berekent de Maximum Adverse Excursion tijdens een streak.
    Hoe ver ging de prijs de verkeerde kant op voordat de streak brak?
    """
    mae_stats = {}
    
    for i, (length, s_idx, e_idx) in enumerate(zip(streak_sequence, streak_start_idx, streak_end_idx)):
        if length < 2 or s_idx >= e_idx:
            continue
            
        # Bepaal richting van de streak op basis van de eerste 2 prijzen
        start_price = prices[s_idx]
        next_price = prices[s_idx + 1]
        direction = 1 if next_price > start_price else -1
        
        # Zoek de laagste/hoogste prijs binnen de streak
        streak_prices = np.array(prices[s_idx:e_idx+1])
        
        if direction == 1: # UP streak
            # MAE is de daling vanaf het begin
            mae = start_price - np.min(streak_prices)
        else: # DOWN streak
            # MAE is de stijging vanaf het begin
            mae = np.max(streak_prices) - start_price
            
        mae_stats.setdefault(length, []).append(abs(mae))

    # Aggregeer
    mae_summary = {}
    for s, arr in mae_stats.items():
        arr = np.array(arr)
        mae_summary[s] = {
            "mae_p50": float(np.percentile(arr, 50)),
            "mae_p90": float(np.percentile(arr, 90)),
            "mae_max": float(arr.max())
        }
    return mae_summary

#3. Tick Velocity Clustering (De API Stress Proxy)
#
# Waarom: Je zei zelf: "Als Binance errors gooit, is er een crash." Ook zonder API errors kun je in de data zien dat 
# de markt onder druk staat door de snelheid van de ticks. Een n=3
# De Metric: binnen streak die in 0.1 seconden gebeurt, is veel gevaarlijker (en winstgevender) dan eentje die 10 
# seconden duurt. Bereken de interarrival tijd (in ms) per tick  een streak. Vergelijk de eerste tick met de latere 
# ticks.

# Als de acceleration_factor < 1 is (de rest van de ticks is sneller dan de eerste), zit je in een momentum cascade. 
# Jouw bot kan dan besluiten: "We zitten in een snelle crash, ik mag meteen blindelings market orders gaan hameren."

def calculate_velocity_stats(event_times, streak_start_idx, streak_end_idx):
    """
    Meet of ticks versnellen tijdens een streak (acceleratie = momentum).
    """
    velocity_stats = {}
    for s, s_idx, e_idx in zip(streak_sequence, streak_start_idx, streak_end_idx):
        if e_idx - s_idx < 2:
            continue
            
        timestamps = np.array(event_times[s_idx:e_idx+1])
        iats = np.diff(timestamps) * 1000  # naar milliseconden
        
        # Snelheid van de eerste 2 ticks vs de rest
        first_iat = np.mean(iats[:2]) if len(iats) >= 2 else iats[0]
        rest_iat = np.mean(iats[2:]) if len(iats) > 2 else first_iat
        
        velocity_stats.setdefault(s, []).append({
            "first_tick_ms": float(first_iat),
            "rest_tick_ms": float(rest_iat),
            "acceleration_factor": float(first_iat / rest_iat) if rest_iat > 0 else 1.0
        })
    return velocity_stats

#4. Asymmetry Index (Bull vs Bear Micro-structuur)
#
# Waarom: In crypto zijn dumps sneller en brutaler dan pumps. Jouw bot moet asymmetrische parameters hebben voor 
# Longs en Shorts. 
# De Metric: Vergelijk de p90 van UP streaks met de p90 van DOWN streaks op hetzelfde level.

def calculate_asymmetry_index(up_data, down_data):
    """
    Berekent of de markt sneller daalt dan stijgt op micro-niveau.
    """
    asymmetry = {}
    for s in sorted(set(list(up_data.keys()) + list(down_data.keys()))):
        up_arr = np.array(up_data.get(s, []))
        down_arr = np.array(down_data.get(s, []))
        
        if len(up_arr) == 0 or len(down_arr) == 0:
            continue
            
        up_p90 = np.percentile(up_arr, 90)
        down_p90 = np.percentile(down_arr, 90)
        
        asymmetry[s] = {
            "up_p90": float(up_p90),
            "down_p90": float(down_p90),
            "bearish_bias": float(down_p90 / up_p90) if up_p90 > 0 else 0
        }
        # Als bearish_bias > 1.0, vallen de neerwaartse streaks harder uit.
        # De bot kan voor SHORT breakouts een grotere spacing aanhouden dan voor LONG.
    return asymmetry

#5. Continuation Probability Matrix (De "Volgende Stap" Calculator)
#
# Waarom: Je wilt de wiskundige grens bepalen van je parabool. Als je op n=3  zit, wat is de exacte kans dat je n=4
# haalt? Als die kans onder de 30% zakt, is je afroom-percentage op n=3 (50%) wiskundig gerechtvaardigd.

def calculate_continuation_probability(streak_sequence):
    """
    Berekent P(Streak = N+1 | Streak = N)
    """
    prob_matrix = {}
    counts = Counter(streak_sequence)
    
    for i in range(len(streak_sequence) - 1):
        n = streak_sequence[i]
        prob_matrix.setdefault(n, {"total": 0, "continued": 0})
        prob_matrix[n]["total"] += 1
        if streak_sequence[i+1] == n + 1:
            prob_matrix[n]["continued"] += 1
            
    probabilities = {}
    for n, data in prob_matrix.items():
        probabilities[n] = float(data["continued"] / data["total"]) if data["total"] > 0 else 0.0
        
    return probabilities

#6. Time-to-Resolution (TTR) per Streak
#
# Waarom dit essentieel is: Voor je "API Hammering" logica. Als een n=3 streak gemiddeld 50 milliseconden duurt, 
# móét je bot Market Taker orders gebruiken en doorhammen, want een Limit order is te traag. Duurt een n=3
# streak echter 2 seconden, dan kun je rustig een Maker Limit order uitschrijven en fees besparen.
# De Metric: De absolute duur (in ms) van het begin van de eerste tick tot het einde van de streak.

def calculate_ttr_per_streak(event_times, streak_sequence, streak_start_idx, streak_end_idx):
    """
    Meet de duur van een streak in milliseconden.
    """
    ttr_stats = {}
    
    for length, s_idx, e_idx in zip(streak_sequence, streak_start_idx, streak_end_idx):
        if e_idx <= s_idx:
            continue
        start_ts = event_times[s_idx]
        end_ts = event_times[e_idx]
        duration_ms = (end_ts - start_ts) * 1000.0
        
        ttr_stats.setdefault(length, []).append(duration_ms)

    ttr_summary = {}
    for s, arr in ttr_stats.items():
        arr = np.array(arr)
        ttr_summary[s] = {
            "ttr_p10": float(np.percentile(arr, 10)), # Snelste 10%
            "ttr_p50": float(np.percentile(arr, 50)), # Mediaan
            "ttr_p90": float(np.percentile(arr, 90))  # Traagste 10%
        }
    return ttr_summary

# 7. Directional Autocorrelation (Trend vs Mean-Reversion Indicator)
#
# Waarom dit essentieel is: Je hebt al autocorrelatie op streak lengtes, maar we willen weten of de richting (UP/DOWN) 
# clustert. Komt na een UP-streak vaker een UP-streak (momentum/trend)? Of komt na een UP-streak vaker een DOWN-streak 
# (mean reversion/chop)? Dit bepaalt of je bot agressief moet doorladderen (Mode 8 breakout) of juist moet scalpen.
# De Metric: Lag-1 autocorrelatie op de +/- 1 tekens van de streak richting.

def calculate_directional_autocorrelation(streak_directions):
    """
    Berekent of de markt neigt naar trend-following (>0) of mean-reversion (<0).
    streak_directions: een lijst van +1 (UP) of -1 (DOWN)
    """
    if len(streak_directions) < 2:
        return 0.0
        
    dirs = np.array(streak_directions)
    mean = dirs.mean()
    if mean == 0: # Perfecte 50/50 verdeling
        numerator = np.sum(dirs[:-1] * dirs[1:])
        denominator = len(dirs) - 1
    else:
        numerator = np.sum((dirs[:-1] - mean) * (dirs[1:] - mean))
        denominator = np.sum((dirs - mean) ** 2)
        
    if denominator == 0:
        return 0.0
        
    return float(numerator / denominator)

# 8. Micro-Fakeout Ratio (De Mode 8 Trigger)
#
# Waarom dit essentieel is: Je Mode 8 idee (Breakout + Fallback) leeft van fakeouts. Hoe vaak breekt de prijs precies 
# 1 level (n=1 of n=2) en keert dan keihard om? Als dit percentage hoog is, mag je je afroom-pas zeker niet op n=2
# zetten, maar moet je hem op n=3 of n=4 zetten.
# De Metric: De ratio van streaks die exact eindigen op lengte 1 of 2, ten opzichte van streaks die lengte 3 of hoger 
# halen.

def calculate_fakeout_ratio(bins):
    """
    Berekent de kans dat een breakout faalt voordat n=3.
    """
    total = sum(bins.values())
    if total == 0:
        return 0.0
        
    fakeouts = bins.get(1, 0) + bins.get(2, 0)
    breakouts = sum(v for k, v in bins.items() if k >= 3)
    
    return float(fakeouts / (fakeouts + breakouts)) if (fakeouts + breakouts) > 0 else 0.0

# 9. Streak length statistics >>
#
# schrijf dan eens een stukkie code erbij om de timing van elke streak (n=1, n=2, n=3, n=4 etc te bepalen en dan apart 
# daar de p90, median etc van), dan hebben we direct hoe lang streaks gemiddeld duren 

import numpy as np

def calculate_streak_timing_stats(event_times, streak_sequence, streak_start_idx, streak_end_idx):
    """
    Berekent de duur (in milliseconden) van elke streak en groepeert deze per streak-lengte.
    Geeft statistieken terug (min, max, mean, median, p10, p90, etc.) per lengte.
    """
    timing_stats = {}

    # Loop door alle gevonden streaks
    for length, s_idx, e_idx in zip(streak_sequence, streak_start_idx, streak_end_idx):
        # Basis checks (skip als indices niet kloppen)
        if e_idx <= s_idx or e_idx >= len(event_times) or s_idx >= len(event_times):
            continue

        start_ts = event_times[s_idx]
        end_ts = event_times[e_idx]
        
        # Duur in milliseconden
        duration_ms = (end_ts - start_ts) * 1000.0
        
        # Groepeer per streak lengte
        timing_stats.setdefault(length, []).append(duration_ms)

    # Bereken statistieken per streak lengte
    timing_summary = {}
    for s, durations in sorted(timing_stats.items()):
        arr = np.array(durations)
        
        timing_summary[s] = {
            "count": int(len(arr)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p10_ms": float(np.percentile(arr, 10)),  # Snelste 10% (extreme velocity)
            "p25_ms": float(np.percentile(arr, 25)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p75_ms": float(np.percentile(arr, 75)),
            "p90_ms": float(np.percentile(arr, 90)),  # Traagste 10% (DODOde markt)
            "p99_ms": float(np.percentile(arr, 99))
        }

    return timing_summary

# 10. jamaja, berekent je code per streak length de p10_ms tot p90_ms ... of misschien nog beter: kunnen we van die 
# lengte ook geen violin plots maken ? 
# Ja, absoluut! Dat is zelfs een veel beter idee dan alleen de p10/p90 getallen. Een violin plot laat niet alleen de 
# mediaan en de percentielen zien, maar ook de vorm van de distributie. 
#
# Stel dat een n=3 streak twee pieken heeft (bimodiaal): één piek op 5ms (de HFT flash crashes) en één piek op 2000ms 
# (normale volatiliteit). Een simpel p10/p90 getal verbergt die structuur, maar een violin plot verraadt direct dat je 
# bot twee totaal verschillende executie-modi nodig heeft.
#
# Hier is de code om prachtige violin plots van de streak-timing te maken. Ik heb er direct een logaritmische schaal 
# aan toegevoegd, omdat timings in de micro-structuur van 1 milliseconde tot 10 seconden lopen, en je anders alles 
# geplet ziet onderaan de grafiek.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

def plot_streak_timing_violins(event_times, streak_sequence, streak_start_idx, streak_end_idx, max_streak=12):
    """
    Maakt een violin plot van de duur (in ms) per streak lengte.
    Omdat timings enorm variëren (1ms tot 10s) gebruiken we een log-y as.
    """
    timing_data = {}

    # 1. Verzamel ruwe data per streak length
    for length, s_idx, e_idx in zip(streak_sequence, streak_start_idx, streak_end_idx):
        if e_idx <= s_idx or e_idx >= len(event_times) or s_idx >= len(event_times):
            continue
        
        # Beperk tot max_streak om de grafiek leesbaar te houden
        if length > max_streak:
            continue
            
        start_ts = event_times[s_idx]
        end_ts = event_times[e_idx]
        duration_ms = (end_ts - start_ts) * 1000.0
        
        # Filter ongeldige timings eruit (soms kloppen CSV timestamps niet)
        if duration_ms > 0:
            timing_data.setdefault(length, []).append(duration_ms)

    if not timing_data:
        print("Geen timing data beschikbaar voor violin plot.")
        return None

    # Sorteer op streak length
    sorted_lengths = sorted(timing_data.keys())
    
    # Data voorbereiden voor matplotlib
    data_to_plot = [timing_data[s] for s in sorted_lengths]
    positions = np.arange(1, len(sorted_lengths) + 1)

    # 2. Maak de plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Violins tekenen
    vp = ax.violinplot(data_to_plot, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    
    # Styling van de violins
    for body in vp['bodies']:
        body.set_facecolor("teal")
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    # 3. Quantiles toevoegen (p10, p50, p90) als dikke markers
    for i, data in enumerate(data_to_plot):
        if len(data) == 0:
            continue
        pos = positions[i]
        
        p10 = np.percentile(data, 10)
        p50 = np.percentile(data, 50) # Mediaan
        p90 = np.percentile(data, 90)
        
        # Teken de markers
        ax.scatter(pos, p10, marker="_", color="blue", s=200, linewidths=2, zorder=3, label="p10 (Snelste)" if i == 0 else "")
        ax.scatter(pos, p50, marker="_", color="white", s=200, linewidths=2, zorder=3, label="p50 (Mediaan)" if i == 0 else "")
        ax.scatter(pos, p90, marker="_", color="red", s=200, linewidths=2, zorder=3, label="p90 (Traagste)" if i == 0 else "")

    # 4. Opmaak
    ax.set_xticks(positions)
    ax.set_xticklabels([f"n={s}" for s in sorted_lengths])
    ax.set_xlabel("Streak Lengte (n)")
    ax.set_ylabel("Duur in milliseconden (ms)")
    ax.set_title("Streak Duur Distributie per Lengte (Log Schaal)\nBlauw=p10 | Wit=p50 | Rood=p90")
    
    # Log schaal is essentieel voor timings!
    ax.set_yscale("log")
    
    # Forceer normale getallen (1, 10, 100) in plaats van 10^1, 10^2
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')
    
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig

# 11. Streak Path Efficiency (SPE) - De "Laser" vs de "Zager"
#
# Waarom de bot dit nodig heeft: 
# In je Mode 8 (Breakout + Fallback) wil je weten of een n=3 breakout een "schone" breakout is. Beweegt de prijs in 3 
# rechte ticks omhoog (een laser), of stuitert hij onderweg iets naar beneden en dan pas omhoog (een zaag)? Als de 
# breakout efficiënt is (weinig ruis), kun je je Fallback Grid verder van de prijs zetten. Als het een zaag is, moet 
# je fallback grid dichter bij elkaar staan, want de prijs wiebelt sneller door je levels heen.
# De Metric: De verhouding tussen de nettobeweging en de bruto afgelegde weg per tick.

def calculate_path_efficiency(prices, streak_start_idx, streak_end_idx):
    """
    Meet hoe 'efficiënt' de prijs beweegt tijdens een streak.
    1.0 = perfecte rechte lijn (laser). Lage waarden = veel ruis (zaag).
    """
    efficiency_stats = {}
    
    for length, s_idx, e_idx in zip(streak_sequence, streak_start_idx, streak_end_idx):
        if length < 2 or e_idx <= s_idx or e_idx >= len(prices):
            continue
            
        # Pak alle prijzen binnen deze streak
        streak_prices = np.array(prices[s_idx:e_idx+1])
        
        # Netto beweging (beginpunt tot eindpunt)
        net_move = abs(streak_prices[-1] - streak_prices[0])
        
        # Bruto beweging (som van alle absolute verschillen per tick)
        gross_move = np.sum(np.abs(np.diff(streak_prices)))
        
        if gross_move > 0:
            eff = net_move / gross_move
            efficiency_stats.setdefault(length, []).append(eff)

    # Samenvatten
    summary = {}
    for s, arr in efficiency_stats.items():
        arr = np.array(arr)
        summary[s] = {
            "efficiency_p50": float(np.percentile(arr, 50)),
            "efficiency_p10": float(np.percentile(arr, 10)) # De ruisigste breakouts
        }
    return summary

# 12. Intra-Streak Tick Acceleration (Momentum Uitputting)
# 
# Waarom de bot dit nodig heeft: Je bot oogst op n=3. Maar stel dat n=3 is bereikt, moet de bot dan wachten op n=4
# of alvast winst nemen? Als de ticks vertragen aan het einde van de n=3 streak, is het momentum uitgeput en keert de 
# prijs waarschijnlijk om. Blijft de snelheid constant of versnelt hij? Dan moet de bot wachten op n=4
#
# De Metric: Vergelijk de gemiddelde interarrival tijd (ms) van de eerste helft van de streak met de tweede helft.

# Bot Beslissing: Als accel_p50 op n=3
#  boven de 1.5 ligt, betekent dit dat de streak drastisch vertraagt. De C-bot moet op dat moment direct de 
# oogst-market-order uitschrijven, omdat de breakout stikt.

def calculate_tick_acceleration(event_times, streak_start_idx, streak_end_idx):
    """
    Meet of ticks versnellen of vertragen tijdens een streak.
    Ratio < 1.0 = Versnelling (Momentum neemt toe)
    Ratio > 1.0 = Vertraging (Momentum stopt, time to harvest!)
    """
    accel_stats = {}
    
    for length, s_idx, e_idx in zip(streak_sequence, streak_start_idx, streak_end_idx):
        if length < 4 or e_idx <= s_idx: # Hebben minimaal 4 ticks nodig voor een betrouwbare meting
            continue
            
        timestamps = np.array(event_times[s_idx:e_idx+1])
        iats = np.diff(timestamps) * 1000.0 # naar ms
        
        # Splits de streak in tweeën
        midpoint = len(iats) // 2
        if midpoint == 0:
            continue
            
        first_half_speed = np.mean(iats[:midpoint])
        second_half_speed = np.mean(iats[midpoint:])
        
        if first_half_speed > 0:
            ratio = second_half_speed / first_half_speed
            accel_stats.setdefault(length, []).append(ratio)

    summary = {}
    for s, arr in accel_stats.items():
        arr = np.array(arr)
        summary[s] = {
            "accel_p50": float(np.percentile(arr, 50)),
            "accel_p90": float(np.percentile(arr, 90)) # Traagheid aan het einde
        }
    return summary

# 13. Counter-Strike Velocity (De "Slingshot" / V-orm)
# 
# Waarom de bot dit nodig heeft: Dit is de ultieme test voor je API Hammer module. Als een n=3  UP streak breekt, 
# hoe hard crasht hij dan naar beneden? Als de eerste tick van de tegenstreak gigantisch is (bijv. een slingshot 
# van 2% in 1 tick), weet de bot dat hij geen tijd heeft om netjes orders te plaatsen, maar direct in de "panic 
# hammer" modus moet schieten.
# De Metric: De absolute delta en de duur (ms) van de allereerste tick van de tegenstreak na een streak van n≥3

def calculate_slingshot_velocity(prices, event_times, streak_sequence, streak_end_idx):
    """
    Meet hoe hard de markt terugkaatst (of doorzakt) direct na een streak break.
    """
    slingshot_stats = {}
    
    # We hebben de index ná het einde van de streak nodig
    for i in range(len(streak_sequence) - 1):
        length = streak_sequence[i]
        if length < 3: # We meten alleen de slingshot na een significante breakout (n>=3)
            continue
            
        e_idx = streak_end_idx[i]
        next_idx = e_idx + 1
        
        if next_idx >= len(prices):
            continue
            
        # De prijs aan het einde van de streak, en de eerste tick erna
        break_price = prices[e_idx]
        slingshot_price = prices[next_idx]
        
        # De tijd die het kostte
        time_ms = (event_times[next_idx] - event_times[e_idx]) * 1000.0
        
        delta = abs(slingshot_price - break_price)
        
        slingshot_stats.setdefault(length, []).append({
            "delta": delta,
            "time_ms": time_ms
        })

    summary = {}
    for s, data in slingshot_stats.items():
        deltas = np.array([d["delta"] for d in data])
        times = np.array([d["time_ms"] for d in data])
        
        summary[s] = {
            "slingshot_delta_p90": float(np.percentile(deltas, 90)), # Hardste klap
            "slingshot_time_p10": float(np.percentile(times, 10))    # Snelste klap (ms)
        }
    return summary

# 14. Theoretical maximal profit
# Dit is de ultieme "Sanity Check". Door de theoretische maximale winst te berekenen (een perfecte backtest in een 
# vacuüm zonder fees, latency of slippage), bepaal je het absolute plafond van je strategie. 
#
# Als deze simulatie op een dag 50% winst laat zien, weet je: "Oké, zelfs als ik de helft verlies aan frictie, maak 
# ik nog 25%." Als de simulatie echter 2% per dag oplevert, weet je dat de strategie na fees en latency waarschijnlijk 
# verliesdraaiend is.
#
# Hier is de Python code voor een Perfect Execution Simulator. Hij loopt door je ruwe tick-data, simuleert de grid 
# levels, telt de n-streaks, hanteert jouw n=3 afroom-matrix, en berekent de winst door het gemiddelde van je mandje 
# (BEP) op te schuiven.

import numpy as np

def simulate_theoretical_max_profit(prices, grid_spacing, harvest_n=3, harvest_pct=0.40, order_size=1.0):
    """
    Simuleer deperfecte uitvoering van de n=3 afroom-strategie.
    - Geen fees, geen latency, geen slippage.
    - Volgt de regels: ladderen bij grid cross, afroomen bij streak >= harvest_n.
    """
    if len(prices) < 2:
        return {"total_profit": 0.0, "total_orders": 0, "total_harvests": 0}

    # State variables
    last_cross_price = prices[0]
    direction = 0      # 1 = UP, -1 = DOWN
    streak = 0
    
    pos_size = 0.0
    pos_value = 0.0    # Totaal investering (om gemiddelde prijs te berekenen)
    realized_pnl = 0.0 # Afgeroomde winst op de bank
    
    total_orders = 0
    total_harvests = 0

    for p in prices[1:]:
        # Check voor UP grid cross
        if p >= last_cross_price + grid_spacing:
            cross_price = last_cross_price + grid_spacing
            
            # 1. Laddering (Positie vergroten)
            pos_size += order_size
            pos_value += cross_price * order_size
            total_orders += 1
            
            # 2. Streak logic
            if direction != 1:
                direction = 1
                streak = 1
            else:
                streak += 1
                
            last_cross_price = cross_price
            
            # 3. Afroom Logica (Harvest)
            if streak >= harvest_n and pos_size > 0:
                avg_price = pos_value / pos_size
                profit_per_unit = cross_price - avg_price
                unrealized = profit_per_unit * pos_size
                
                if unrealized > 0:
                    pnl_to_take = unrealized * harvest_pct
                    # Hoeveel units moeten we verkopen om deze winst te pakken?
                    vol_to_sell = pnl_to_take / (cross_price - avg_price)
                    
                    # Positie verkleinen en waarde aanpassen
                    pos_size -= vol_to_sell
                    pos_value -= vol_to_sell * avg_price
                    realized_pnl += pnl_to_take
                    total_harvests += 1

        # Check voor DOWN grid cross
        elif p <= last_cross_price - grid_spacing:
            cross_price = last_cross_price - grid_spacing
            
            # 1. Laddering (Short posities vergroten in een breakout naar beneden)
            pos_size += order_size
            # Bij een short betekent dit dat we het verkopen, dus pos_value gaat omlaag
            pos_value -= cross_price * order_size
            total_orders += 1
            
            # 2. Streak logic
            if direction != -1:
                direction = -1
                streak = 1
            else:
                streak += 1
                
            last_cross_price = cross_price
            
            # 3. Afroom Logica (Harvest)
            if streak >= harvest_n and pos_size > 0:
                # Voor shorts is de gemiddelde prijs de prijs waarop we short gegaan zijn
                avg_price = pos_value / pos_size
                # Winst is (instap - huidig)
                profit_per_unit = avg_price - cross_price
                unrealized = profit_per_unit * pos_size
                
                if unrealized > 0:
                    pnl_to_take = unrealized * harvest_pct
                    vol_to_buy = pnl_to_take / (avg_price - cross_price)
                    
                    pos_size -= vol_to_buy
                    pos_value -= vol_to_buy * avg_price
                    realized_pnl += pnl_to_take
                    total_harvests += 1

    # Einde van de dag: sluit resterende open positie af tegen de laatste prijs
    final_price = prices[-1]
    if pos_size > 0.0001:
        if direction == 1: # Long positie sluiten
            unrealized_end = (final_price - (pos_value/pos_size)) * pos_size
        else: # Short positie sluiten
            unrealized_end = ((pos_value/pos_size) - final_price) * pos_size
            
        realized_pnl += max(0, unrealized_end) # We tellen alleen de winst, geen verlies voor theoretisch max

    return {
        "total_profit_units": float(realized_pnl),
        "total_profit_pct": float(realized_pnl / (prices[0] * 1.0) * 100), # Als % van startprijs
        "total_orders": int(total_orders),
        "total_harvests": int(total_harvests),
        "avg_profit_per_order": float(realized_pnl / total_orders) if total_orders > 0 else 0
    }

# 15. Theoretical max profit (model 2)

import numpy as np

def simulate_theoretical_max_profit2(prices, grid_spacing_pct, harvest_n=3, harvest_pct=0.40, order_size=1.0):
    """
    Simuleer de perfecte uitvoering van de afroom-strategie.
    - Geen fees, geen latency, geen slippage.
    - Reset de positie (sluit alles) zodra de streak breekt (richting omkeert).
    """
    if len(prices) < 2:
        return 0.0, 0, 0

    # Bepaal absolute spacing op basis van percentage
    start_price = prices[0]
    grid_spacing = start_price * (grid_spacing_pct / 100.0)

    last_cross_price = start_price
    direction = 0  # 1 = UP, -1 = DOWN
    streak = 0
    
    pos_size = 0.0
    pos_cost_basis = 0.0  # Totaal investering (long) of opbrengst (short)
    realized_pnl = 0.0
    
    total_orders = 0
    total_harvests = 0

    def close_position(price):
        """Sluit de hele positie en bereken winst/verlies."""
        nonlocal pos_size, pos_cost_basis, realized_pnl
        if pos_size > 0:
            if direction == 1: # Long sluiten
                realized_pnl += (price - (pos_cost_basis/pos_size)) * pos_size
            elif direction == -1: # Short sluiten
                realized_pnl += ((pos_cost_basis/pos_size) - price) * pos_size
            pos_size = 0
            pos_cost_basis = 0

    for p in prices[1:]:
        # UP grid cross
        if p >= last_cross_price + grid_spacing:
            cross_price = last_cross_price + grid_spacing
            
            # Als we van richting wisselen, sluiten we de oude positie
            if direction != 1:
                close_position(cross_price)
                direction = 1
                streak = 1
            else:
                streak += 1
                
            # 1. Laddering (Positie vergroten)
            pos_size += order_size
            pos_cost_basis += cross_price * order_size
            total_orders += 1
            last_cross_price = cross_price
            
            # 2. Afroom Logica (Harvest)
            if streak >= harvest_n and pos_size > 0:
                avg_price = pos_cost_basis / pos_size
                unrealized = (cross_price - avg_price) * pos_size
                if unrealized > 0:
                    pnl_to_take = unrealized * harvest_pct
                    vol_to_sell = pnl_to_take / (cross_price - avg_price)
                    pos_size -= vol_to_sell
                    pos_cost_basis -= vol_to_sell * avg_price
                    realized_pnl += pnl_to_take
                    total_harvests += 1

        # DOWN grid cross
        elif p <= last_cross_price - grid_spacing:
            cross_price = last_cross_price - grid_spacing
            
            if direction != -1:
                close_position(cross_price)
                direction = -1
                streak = 1
            else:
                streak += 1
                
            # 1. Laddering (Short positie vergroten)
            pos_size += order_size
            pos_cost_basis += cross_price * order_size
            total_orders += 1
            last_cross_price = cross_price
            
            # 2. Afroom Logica (Harvest)
            if streak >= harvest_n and pos_size > 0:
                avg_price = pos_cost_basis / pos_size
                unrealized = (avg_price - cross_price) * pos_size
                if unrealized > 0:
                    pnl_to_take = unrealized * harvest_pct
                    vol_to_buy = pnl_to_take / (avg_price - cross_price)
                    pos_size -= vol_to_buy
                    pos_cost_basis -= vol_to_buy * avg_price
                    realized_pnl += pnl_to_take
                    total_harvests += 1

    # Sluit resterende open positie aan einde van de dag
    close_position(prices[-1])

    return realized_pnl, total_orders, total_harvests


def run_parameter_optimization_loop(prices):
    """
    Voert een grid-search uit om de optimale spacing en harvest N te vinden.
    """
    if len(prices) < 100:
        print("Niet genoeg data voor optimalisatie.")
        return

    # De spacings (in %) die we willen testen
    spacings_to_test = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    # De harvest momenten (n) die we willen testen
    harvest_ns_to_test = [2, 3, 4, 5, 6]
    
    harvest_pct = 0.40 # vaste afroom ratio
    order_size = 1.0

    print("\n" + "="*80)
    print(" THEORETISCHE MAX WINST OPTIMALISATIE (Perfect Execution) ".center(80))
    print("="*80)
    print(f" Startprijs: {prices[0]:.2f} | Data Points: {len(prices)} | Afroom: {harvest_pct*100:.0f}%")
    print("-" * 80)
    print(f"{'Spacing %':<12} | {'Harvest N':<12} | {'Winst (units)':<15} | {'Winst %':<10} | {'# Orders':<10} | {'# Oogsten':<10}")
    print("-" * 80)

    results = []
    
    for spacing_pct in spacings_to_test:
        for n_harvest in harvest_ns_to_test:
            pnl, orders, harvests = simulate_theoretical_max_profit2(
                prices, 
                grid_spacing_pct=spacing_pct, 
                harvest_n=n_harvest, 
                harvest_pct=harvest_pct,
                order_size=order_size
            )
            
            pnl_pct = (pnl / prices[0]) * 100
            results.append((spacing_pct, n_harvest, pnl, pnl_pct, orders, harvests))
            
            print(f"{spacing_pct:<12.2f} | {n_harvest:<12} | {pnl:<15.4f} | {pnl_pct:<10.2f} | {orders:<10} | {harvests:<10}")
            
    print("-" * 80)
    
    # Vind de absolute winnaar
    best_result = max(results, key=lambda x: x[3])
    print(f"\n=> WINNAAR: Spacing {best_result[0]:.2f}% | n={best_result[1]} | Winst: {best_result[3]:.2f}%")
    print("=> (Let op: Dit is zonder fees. Halveer dit getal voor een realistische bot verwachting)\n")


# 16. Theoretical max profit #3

import numpy as np

def simulate_theoretical_max_profit_absolute(prices, grid_spacing, harvest_n=3, harvest_pct=0.40, order_size=1.0):
    """
    Simuleer de perfecte uitvoering met een absolute grid spacing.
    Reset de positie (sluit alles) zodra de streak breekt (richting omkeert).
    """
    if len(prices) < 2 or grid_spacing <= 0:
        return 0.0, 0, 0

    last_cross_price = prices[0]
    direction = 0  # 1 = UP, -1 = DOWN
    streak = 0
    
    pos_size = 0.0
    pos_cost_basis = 0.0  
    realized_pnl = 0.0
    
    total_orders = 0
    total_harvests = 0

    def close_position(price):
        nonlocal pos_size, pos_cost_basis, realized_pnl
        if pos_size > 0:
            if direction == 1: 
                realized_pnl += (price - (pos_cost_basis/pos_size)) * pos_size
            elif direction == -1: 
                realized_pnl += ((pos_cost_basis/pos_size) - price) * pos_size
            pos_size = 0
            pos_cost_basis = 0

    for p in prices[1:]:
        # UP grid cross
        if p >= last_cross_price + grid_spacing:
            cross_price = last_cross_price + grid_spacing
            if direction != 1:
                close_position(cross_price)
                direction = 1
                streak = 1
            else:
                streak += 1
                
            pos_size += order_size
            pos_cost_basis += cross_price * order_size
            total_orders += 1
            last_cross_price = cross_price
            
            if streak >= harvest_n and pos_size > 0:
                avg_price = pos_cost_basis / pos_size
                unrealized = (cross_price - avg_price) * pos_size
                if unrealized > 0:
                    pnl_to_take = unrealized * harvest_pct
                    vol_to_sell = pnl_to_take / (cross_price - avg_price)
                    pos_size -= vol_to_sell
                    pos_cost_basis -= vol_to_sell * avg_price
                    realized_pnl += pnl_to_take
                    total_harvests += 1

        # DOWN grid cross
        elif p <= last_cross_price - grid_spacing:
            cross_price = last_cross_price - grid_spacing
            if direction != -1:
                close_position(cross_price)
                direction = -1
                streak = 1
            else:
                streak += 1
                
            pos_size += order_size
            pos_cost_basis += cross_price * order_size
            total_orders += 1
            last_cross_price = cross_price
            
            if streak >= harvest_n and pos_size > 0:
                avg_price = pos_cost_basis / pos_size
                unrealized = (avg_price - cross_price) * pos_size
                if unrealized > 0:
                    pnl_to_take = unrealized * harvest_pct
                    vol_to_buy = pnl_to_take / (avg_price - cross_price)
                    pos_size -= vol_to_buy
                    pos_cost_basis -= vol_to_buy * avg_price
                    realized_pnl += pnl_to_take
                    total_harvests += 1

    close_position(prices[-1])
    return realized_pnl, total_orders, total_harvests


def run_data_driven_optimization_loop(prices, price_moves):
    """
    Gebruikt de p25, p50 (mediaan), p75 en p90 van de n=1, n=2 en n=3 streaks
    als grid spacing voor de theoretische simulatie.
    """
    if len(prices) < 100:
        print("Niet genoeg data voor optimalisatie.")
        return

    # 1. Bepaal de kandidaat spacings op basis van echte streak data
    spacings_to_test = {}
    for n in [1, 2, 3]:
        if n in price_moves and len(price_moves[n]) > 0:
            arr = np.array(price_moves[n])
            spacings_to_test[f"n{n}_p25"] = float(np.percentile(arr, 25))
            spacings_to_test[f"n{n}_p50"] = float(np.percentile(arr, 50))
            spacings_to_test[f"n{n}_p75"] = float(np.percentile(arr, 75))
            spacings_to_test[f"n{n}_p90"] = float(np.percentile(arr, 90))

    harvest_ns_to_test = [2, 3, 4]
    harvest_pct = 0.40
    order_size = 1.0

    print("\n" + "="*90)
    print(" DATA-DRIVEN THEORETISCHE MAX WINST (Gebaseerd op streak statistieken) ".center(90))
    print("="*90)
    print(f" Startprijs: {prices[0]:.2f} | Data Points: {len(prices)} | Afroom: {harvest_pct*100:.0f}%")
    print("-" * 90)
    print(f"{'Bron Statistiek':<15} | {'Abs. Spacing':<12} | {'Harvest N':<10} | {'Winst (units)':<15} | {'Winst %':<10} | {'# Orders':<10} | {'# Oogsten':<10}")
    print("-" * 90)

    results = []
    
    for stat_name, spacing_val in spacings_to_test.items():
        for n_harvest in harvest_ns_to_test:
            pnl, orders, harvests = simulate_theoretical_max_profit_absolute(
                prices, 
                grid_spacing=spacing_val, 
                harvest_n=n_harvest, 
                harvest_pct=harvest_pct,
                order_size=order_size
            )
            
            pnl_pct = (pnl / prices[0]) * 100
            results.append((stat_name, spacing_val, n_harvest, pnl, pnl_pct, orders, harvests))
            
            print(f"{stat_name:<15} | {spacing_val:<12.6f} | {n_harvest:<10} | {pnl:<15.4f} | {pnl_pct:<10.2f} | {orders:<10} | {harvests:<10}")
            
    print("-" * 90)
    
    # Vind de absolute winnaar
    best_result = max(results, key=lambda x: x[4])
    print(f"\n=> WINNAAR: Bron={best_result[0]} ({best_result[1]:.6f}) | n={best_result[2]} | Winst: {best_result[4]:.2f}%")
    print("=> (Let op: Dit is zonder fees. Halveer dit getal voor een realistische bot verwachting)\n")

# ----------------------------
# Main
# ----------------------------

def main():
    # ARGS UITBREIDEN, BV ARGUMENT PER PLOT
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--nosplit", action="store_true")
    p.add_argument("--noecdf", action="store_true")
    p.add_argument("--nology", action="store_true")
    p.add_argument("--noextremes", action="store_true")
    p.add_argument("--noregimes", action="store_true")
    p.add_argument("--noplot", action="store_true")
    p.add_argument("--outfile", action="store_true")
    args = p.parse_args()

    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV not found: {args.csv}")
        csv_path = args.csv
    else:
        csv_path = "./testdata/BTCUSD-testdata.csv"
        if not os.path.exists(csv_path):
            print(f"Warning: Default CSV not found: {csv_path}")
            return

    bins, pm, up, down, ticks, total_streaks, interarr, streak_seq, start_idx, end_idx, prices, event_times = read_csv(csv_path)

    signed_deltas, streak_ids, streak_pos = generate_streak_info(prices)

#    print(streak_seq)
    print(start_idx)
    print(end_idx)

    trans = run_conditioned_transitions(streak_seq, max_run=10, max_next=10)

    if args.outfile:
        outfile = args.outfile
    else:
        outfile = "streaks_export.json"

    # TRANSITIES (trans) EN EXTRA DARA in json exporteren
    # todo 21/07/2026 next line is debug commented out, should work in real code
    #export_json(pm, bins, outfile, csv_path, ticks, total_streaks, interarr, up, down) 

# DEBUG LINES:
#    print("STREAK_SEQ:", streak_seq)
#    print(trans)

    if not args.noplot:

# FIG1: een hele hoop randdetails te veranderen in de plots
#       fig1 = plot_main(args, pm, up, down, bins, csv_path, ticks, total_streaks)

# FIG2: TODO: deze plot duurt extreem lang om te plotten ? how come ?
#       fig2 = plot_streak_timeline(streak_seq)
       
# FIG3: OK 
#        fig3 = plot_interarrival(interarr)
        
# FIG4: OK
#        fig4 = plot_ecdf_fan(pm, max_streak=None)

# FIG5: TODO: Zelfde prob als eerste plots, alles mag 1 index opschuiven, en de delta schaal lijkt niet te kloppen (die loopt nu tot max 100
# maar er zijn sowieso delta's groter dan 100)
#        fig5 =plot_quantile_heatmap(pm)

# FIG6: OK
#        fig6 = plot_interarrival_logbins(interarr)

# FIG7: TODO: Zelfde schaal probleem als de eerste plots, alles 1 index opschuiven en een forloop maken die alle transition plots doet (ttz alle mogelijke base_streaks) en ze in een grid zet
        fig7 = plot_run_transition_bars(trans, base_streak=4)

# FIG8: TODO:  veel te trage plot
#      fig8 = plot_cumulative_steak_violins(signed_deltas, streak_ids, streak_pos, max_depth=5)

# FIG9: TODO: OK > log y-axis zou nog beter zijn 
        fig9 = plot_streak_run_distribution(streak_seq, max_streak=5, normalize=True)

# FIG10: TODO FIX NEXT CODE;throws an error
#        fig10 = plot_top_streak_timeline_stacked(
#    prices,
#    event_times,
#    streak_seq,
#    start_idx,
#    end_idx,
#    top_n=3,
#    rect_height_px=5
#)

# File "/usr/local/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 4755, in _parse_scatter_color_args
#    colors = mcolors.to_rgba_array(c)
#             ^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/colors.py", line 515, in to_rgba_array
#    rgba = np.array([to_rgba(cc) for cc in c])
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/colors.py", line 515, in <listcomp>
#    rgba = np.array([to_rgba(cc) for cc in c])
#                     ^^^^^^^^^^^
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/colors.py", line 317, in to_rgba
#    rgba = _to_rgba_no_colorcycle(c, alpha)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/colors.py", line 401, in _to_rgba_no_colorcycle
#    raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
#ValueError: Invalid RGBA argument: np.float64(2.0)
#
#The above exception was the direct cause of the following exception:
#
#Traceback (most recent call last):
#  File "/home/i/streak_analyzer.py", line 2502, in <module>
#    main()
#  File "/home/i/streak_analyzer.py", line 2445, in main
#    fig10 = plot_price_and_timedelta_events(np.array(prices), np.array(event_times), np.array(streak_seq))
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/home/i/streak_analyzer.py", line 911, in plot_price_and_timedelta_events
#    sc = ax_p.scatter(
#         ^^^^^^^^^^^^^
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/_api/deprecation.py", line 453, in wrapper
#    return func(*args, **kwargs)
#           ^^^^^^^^^^^^^^^^^^^^^
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/__init__.py", line 1521, in inner
#    return func(
#           ^^^^^
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 4948, in scatter
#    self._parse_scatter_color_args(
#  File "/usr/local/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 4761, in _parse_scatter_color_args
#    raise invalid_shape_exception(c.size, xsize) from err
#ValueError: 'c' argument has 57000 elements, which is inconsistent with 'x' and 'y' with size 102519.
#
#shell returned 1

# FIG11: TODO  kan niet kloppen, de y axis is geen delta waarde, nog streek lengte, het lijkt eerder op absolute price moves
        fig11 = plot_streak_duration_histogram(interarr, streak_seq)

# FIG12: TODO: x-axis geen scientific notatie
        fig12 = plot_cumulative_price_move(prices, streak_seq)

# FIG13: TODO: x-axis geen scientific notatie
#        fig13 = plot_delta_vs_streak(prices, streak_seq)

#- FIG14: TODO: de x-waarden kunnen bijna niet kloppen... er staan waarden in van 100seconden, dat zou meer dan 1.5 minuut in een bepaalde streak zijn ? kan dit ?
#        fig14 = plot_delta_vs_interarrival(prices, streak_seq, interarr)

# FIG15: OK
# fig15 = plot_streak_autocorrelation(streak_seq)

# FIG16: OK
#        fig16 = plot_top_streaks_over_time(event_times, streak_seq)

# FIG16b:  TODO: error all values must be the same ...
#        fig16 = plot_top_streaks_with_trend(event_times, streak_seq, top_percent=5, rolling_window='1D')

# FIG17: OK - TODO: index checken !
# fig17 = plot_leader_laggard(streak_seq)

# FIG18: OK, wel uitleg wat/ho te gebruiken toevoegen
#        fig18 = plot_price_move_density(prices)

# FIG19: TODO: volledig herdoen
#        plot_extremes(pm)

# FIG20: TODO: checken:
# plot_histogram(bins)

# FIG21: TODO: y.2 Tail-only violin (p90+)
#
#Maak een tweede rij violins:
#
#data = delta[delta >= p90]
#log-y
#veel smaller
#Waarom:
#hoofdviolin liegt soms
#tail-violin liegt nooit

# FIG22-yy: TODO: checken als we andere plots vergeten zijn ?

# FIG23: TODO: plot_streak_sequences plot vergeten 
# FIG24: TODO: plot_extreme_streak_origin vergeten ?
# FIG25: TODO: plot_price_and_timedelta_events ?

#        fig25 = plot_top_cumulative_streaks_with_interarrival(event_times, streak_seq, interarr, top_n=5)

# FIG26: Streak Timing Violins
        fig26 = plot_streak_timing_violins(event_times, streak_seq, start_idx, end_idx, max_streak=12)

        plt.show()

        run_parameter_optimization_loop(prices)
        run_data_driven_optimization_loop(prices, pm)

        # THEORETISCHE MAX WINST SIMULATIE
        # Pak de gesuggereerde grid spacing
        suggested_spacing = suggest_global_grid_width(pm, list(pm.keys()))
        if suggested_spacing and suggested_spacing > 0:
            print(f"\n=== Theoretische Max Winst Simulatie (Spacing: {suggested_spacing:.4f}) ===")
            
            # Simuleer met n=3 harvest, 40% afroom
            sim_result = simulate_theoretical_max_profit(
                prices, 
                grid_spacing=suggested_spacing, 
                harvest_n=3, 
                harvest_pct=0.40,
                order_size=1.0
            )
            
            print(f"Totale orders getriggerd : {sim_result['total_orders']}")
            print(f"Totale oogsten (n>=3)   : {sim_result['total_harvests']}")
            print(f"Theoretische Winst      : {sim_result['total_profit_units']:.4f} units")
            print(f"Theoretisch Rendement   : {sim_result['total_profit_pct']:.2f}% van startkapitaal")
            print(f"Gem. winst per order    : {sim_result['avg_profit_per_order']:.4f} units")

if __name__ == "__main__":
    main()


# TODO: y.2 RADAR PLOT 
# ASSEN: tail_mass (prijs) - p90_accel - stability - CV van interarrival (CV = std(Δt) / mean(Δt)), FANO FACTOR / Parabolix index / etc

# TODO: z.2 micro take optimalisatie afleiden (kan enkel maar als we andere parameters specifiëren (is voor laatste)
