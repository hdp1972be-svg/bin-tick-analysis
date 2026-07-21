
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# --- CONFIG ---
demo_tick_file = "trading_cloud/tick-data/BTCUSD/csv/BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv"
boundaries = [60, 300, 900]  # seconds: 1min, 5min, 15min
target_bins_per_boundary = 500  # aantal sub-bins per boundary, adaptief

# --- LOAD DATA ---
df = pd.read_csv(demo_tick_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['mid'] = (df['ask'] + df['bid']) / 2
df['tick_dir'] = np.sign(df['mid'].diff().fillna(0))

metrics_dict = {}

for period_sec in boundaries:
    # relative seconds within boundary
    df['relative_sec'] = (
        df['timestamp'].dt.minute*60 +
        df['timestamp'].dt.second +
        df['timestamp'].dt.microsecond/1e6
    ) % period_sec

    # adaptive bin width
    bin_width = period_sec / target_bins_per_boundary
    bins = np.arange(0, period_sec + bin_width, bin_width)

    # histogram for tick count
    tick_counts, edges = np.histogram(df['relative_sec'], bins=bins)
    bin_centers = edges[:-1] + bin_width/2

    # per-bin metrics
    avg_mid = []
    std_mid = []
    vol_bid = []
    vol_ask = []
    entropy_dir = []

    for start, end in zip(edges[:-1], edges[1:]):
        sub = df[(df['relative_sec'] >= start) & (df['relative_sec'] < end)]
        avg_mid.append(sub['mid'].mean() if len(sub) > 0 else np.nan)
        std_mid.append(sub['mid'].std() if len(sub) > 0 else np.nan)
        vol_bid.append(sub['vol_bid'].sum())
        vol_ask.append(sub['vol_ask'].sum())
        if len(sub) > 0:
            hist, _ = np.histogram(sub['tick_dir'], bins=[-1,0,1,2], density=True)
            entropy_dir.append(entropy(hist))
        else:
            entropy_dir.append(np.nan)

    metrics_dict[period_sec] = pd.DataFrame({
        'bin_center': bin_centers,
        'tick_count': tick_counts,
        'avg_mid': avg_mid,
        'std_mid': std_mid,
        'vol_bid_sum': vol_bid,
        'vol_ask_sum': vol_ask,
        'entropy_tick_dir': entropy_dir
    })

for period_sec in boundaries:
    mdf = metrics_dict[period_sec]

    # --- Key values ---
    top_tick_idx = mdf['tick_count'].idxmax()
    top_vol_idx = mdf['std_mid'].idxmax()
    top_mid_idx = mdf['avg_mid'].idxmax()
    top_entropy_idx = mdf['entropy_tick_dir'].idxmax()

    print(f"\n=== Boundary {period_sec//60} min ===")
    print(f"Highest tick count: {mdf['tick_count'][top_tick_idx]:.0f} at bin {mdf['bin_center'][top_tick_idx]:.3f} sec")
    print(f"Highest midprice: {mdf['avg_mid'][top_mid_idx]:.2f} at bin {mdf['bin_center'][top_mid_idx]:.3f} sec")
    print(f"Highest volatility (std_mid): {mdf['std_mid'][top_vol_idx]:.2f} at bin {mdf['bin_center'][top_vol_idx]:.3f} sec")
    print(f"Highest entropy: {mdf['entropy_tick_dir'][top_entropy_idx]:.3f} at bin {mdf['bin_center'][top_entropy_idx]:.3f} sec")

# --- PLOT ---
fig, axes = plt.subplots(len(boundaries), 1, figsize=(14, 6*len(boundaries)), sharex=False)

if len(boundaries) == 1:
    axes = [axes]

for ax, period_sec in zip(axes, boundaries):
    mdf = metrics_dict[period_sec]
    ax.bar(mdf['bin_center'], mdf['tick_count'], width=(period_sec/target_bins_per_boundary)*0.9, alpha=0.6, label='Tick count')
    ax.plot(mdf['bin_center'], mdf['avg_mid'], color='r', label='Avg mid')
    ax.fill_between(mdf['bin_center'],
                    np.array(mdf['avg_mid']) - np.array(mdf['std_mid']),
                    np.array(mdf['avg_mid']) + np.array(mdf['std_mid']),
                    color='r', alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(mdf['bin_center'], mdf['entropy_tick_dir'], color='g', label='Entropy')
    ax.set_title(f'Tick metrics around {period_sec//60} min boundary')
    ax.set_xlabel('Seconds relative to boundary')
    ax.set_ylabel('Tick count / midprice')
    ax2.set_ylabel('Entropy tick direction')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- optional: view metrics numerically ---
print(metrics_dict[60].head(10))
