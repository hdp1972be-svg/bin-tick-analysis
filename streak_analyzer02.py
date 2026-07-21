#!/usr/local/bin/python3.11

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

def human_fmt(x, _):
    if x == 0:
        return "0"
    if abs(x) >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.6f}".rstrip("0").rstrip(".")

def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def safe_percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else None

def suggested_grid_width(arr):
    if len(arr) < 5:
        return None
    return safe_percentile(arr, 90) - safe_percentile(arr, 50)


def read_csv(path):
    df = pd.read_csv(path)
    close = df["close"].values

    streak_bins = defaultdict(int)
    streak_moves = defaultdict(list)

    streak = 0
    direction = 0
    start_price = close[0]

    for i in range(1, len(close)):
        delta = close[i] - close[i - 1]
        new_dir = np.sign(delta)

        if new_dir == direction and new_dir != 0:
            streak += 1
        else:
            if streak > 0:
                move = abs(close[i - 1] - start_price)
                streak_bins[streak] += 1
                streak_moves[streak].append(move)

            streak = 1
            direction = new_dir
            start_price = close[i - 1]

    return dict(streak_bins), dict(streak_moves)


def compute_regimes(arr):
    return {
        "p90": safe_percentile(arr, 90),
        "p99": safe_percentile(arr, 99),
        "max": float(arr.max())
    }


def draw_split_violin(ax, data, x):
    left = ax.violinplot([data], positions=[x],
                         widths=0.8, showextrema=False)
    for b in left["bodies"]:
        b.set_facecolor("steelblue")
        b.set_alpha(0.6)
        b.set_clip_path(
            plt.Rectangle((x - 1, -1e9), 1, 2e9, transform=ax.transData)
        )

    right = ax.violinplot([data], positions=[x],
                          widths=0.8, showextrema=False)
    for b in right["bodies"]:
        b.set_facecolor("firebrick")
        b.set_alpha(0.6)
        b.set_clip_path(
            plt.Rectangle((x, -1e9), 1, 2e9, transform=ax.transData)
        )


def annotate_stat(ax, x, y, label, side="right"):
    dx = 0.42 if side == "right" else -0.42
    ax.annotate(
        label,
        xy=(x + dx, y),
        xytext=(x + dx * 1.9, y),
        arrowprops=dict(arrowstyle="-", lw=1),
        fontsize=8,
        va="center"
    )


def plot_violins(streak_moves, args, csv_path):
    streaks = sorted(s for s in streak_moves if s <= args.max_streak)
    rows = [streaks[i:i+args.max_violins]
            for i in range(0, len(streaks), args.max_violins)]

    fig, axes = plt.subplots(len(rows), 1,
                             figsize=(18, 4 * len(rows)),
                             sharey=True)

    if len(rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        grid_widths = []

        for i, s in enumerate(row):
            arr = np.array(streak_moves[s])
            draw_split_violin(ax, arr, i + 1)

            stats = {
                "median": np.median(arr),
                "p90": safe_percentile(arr, 90)
            }

            annotate_stat(ax, i + 1, stats["median"], "median")
            annotate_stat(ax, i + 1, stats["p90"], "p90")

            gw = suggested_grid_width(arr)
            if gw:
                grid_widths.append(gw)

            if not args.noregimebg:
                r = compute_regimes(arr)
                ax.axhspan(0, r["p90"], color="steelblue", alpha=0.08)
                ax.axhspan(r["p90"], r["p99"], color="orange", alpha=0.08)
                ax.axhspan(r["p99"], r["max"], color="darkred", alpha=0.12)
                ax.axhline(r["p90"], linestyle="--", alpha=0.6)
                ax.axhline(r["p99"], linestyle=":", alpha=0.6)

        if grid_widths:
            ax.text(
                0.99, 0.94,
                f"Suggested grid width ≈ {np.median(grid_widths):.6f}",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9
            )

        ax.set_xticks(range(1, len(row) + 1))
        ax.set_xticklabels([f"Streak {s}" for s in row])
        ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))

        if not args.nology:
            ax.set_yscale("log")

        ax.grid(alpha=0.2)

    fig.text(0.01, 0.01, f"Source CSV: {csv_path}", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_histogram(streak_bins):
    xs = sorted(streak_bins)
    ys = [streak_bins[x] for x in xs]

    plt.figure(figsize=(10, 4))
    plt.bar(xs, ys, alpha=0.7)

    for x, y in zip(xs, ys):
        plt.text(x, y, f"N={y}", ha="center", va="bottom", fontsize=8)

    plt.xlabel("Streak length")
    plt.ylabel("Count")
    plt.title("Streak length distribution")
    plt.grid(alpha=0.3)
    plt.show()


def export_json(streak_moves, streak_bins, out_path):
    export = {}

    for s, data in streak_moves.items():
        arr = np.array(data)
        r = compute_regimes(arr)

        export[s] = {
            "count": streak_bins.get(s, 0),
            "stats": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "mad": float(mad(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "p90": r["p90"],
                "p99": r["p99"]
            },
            "grid": {
                "suggested_width": suggested_grid_width(arr),
                "zone": [0, r["p90"]]
            },
            "burst": {
                "zone": [r["p90"], r["p99"]]
            },
            "extreme": {
                "zone": [r["p99"], r["max"]]
            }
        }

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV file with 'close' column")
    p.add_argument("--json-out", default="streak_analysis.json")
    p.add_argument("--max-streak", type=int, default=10)
    p.add_argument("--max-violins", type=int, default=5)
    p.add_argument("--nology", action="store_true")
    p.add_argument("--noregimebg", action="store_true")

    args = p.parse_args()

    streak_bins, streak_moves = read_csv(args.csv)

    plot_violins(streak_moves, args, args.csv)
    plot_histogram(streak_bins)
    export_json(streak_moves, streak_bins, args.json_out)

    print(f"✔ JSON exported to {args.json_out}")


if __name__ == "__main__":
    main()


