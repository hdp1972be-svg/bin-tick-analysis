
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.signal import convolve

# --- CONFIG ---
demo_tick_file = "trading_cloud/tick-data/BTCUSD/csv/BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv"
boundaries = [60, 300, 900]  # seconds: 1min, 5min, 15min

# --- LOAD DATA ---
df = pd.read_csv(demo_tick_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['mid'] = (df['ask'] + df['bid']) / 2
df['tick_dir'] = np.sign(df['mid'].diff().fillna(0))

# --- HELPER FUNCTIONS ---
def compute_metrics(df, period_sec):
    seconds_since_boundary = df['timestamp'].dt.second + df['timestamp'].dt.minute*60
    relative = seconds_since_boundary % period_sec
    metrics = []

    for i in range(period_sec):
        mask = (relative >= i) & (relative < i+1)
        subset = df[mask]
        tick_count = len(subset)
        avg_mid = subset['mid'].mean() if tick_count>0 else np.nan
        std_mid = subset['mid'].std() if tick_count>0 else np.nan
        vol_bid_sum = subset['vol_bid'].sum()
        vol_ask_sum = subset['vol_ask'].sum()
        hist, _ = np.histogram(subset['tick_dir'], bins=[-1,0,1,2], density=True)
        ent = entropy(hist) if tick_count>0 else np.nan
        metrics.append({
            'sec': i,
            'tick_count': tick_count,
            'avg_mid': avg_mid,
            'std_mid': std_mid,
            'vol_bid_sum': vol_bid_sum,
            'vol_ask_sum': vol_ask_sum,
            'entropy_tick_dir': ent
        })
    return pd.DataFrame(metrics)

def compute_metrics_fixed(df, period_sec, bin_sec=1.0):
    seconds_since_boundary = df['timestamp'].dt.minute*60 + df['timestamp'].dt.second + df['timestamp'].dt.microsecond/1e6
    relative = seconds_since_boundary % period_sec
    n_bins = int(period_sec / bin_sec)
    bins = np.linspace(0, period_sec, n_bins+1)
    
    df['bin'] = pd.cut(relative, bins, right=False, labels=False)
    
    metrics = []
    for b in range(n_bins):
        subset = df[df['bin'] == b]
        tick_count = len(subset)
        avg_mid = subset['mid'].mean() if tick_count>0 else np.nan
        std_mid = subset['mid'].std() if tick_count>0 else np.nan
        vol_bid_sum = subset['vol_bid'].sum()
        vol_ask_sum = subset['vol_ask'].sum()
        hist, _ = np.histogram(subset['tick_dir'], bins=[-1,0,1,2], density=True)
        ent = entropy(hist) if tick_count>0 else np.nan
        metrics.append({
            'bin': b,
            'tick_count': tick_count,
            'avg_mid': avg_mid,
            'std_mid': std_mid,
            'vol_bid_sum': vol_bid_sum,
            'vol_ask_sum': vol_ask_sum,
            'entropy_tick_dir': ent
        })
    return pd.DataFrame(metrics)

# Compute metrics for each boundary
metrics_dict = {b: compute_metrics_fixed(df,b) for b in boundaries}

# --- PLOT ---
fig, axes = plt.subplots(len(boundaries)+1,1,figsize=(12,8), sharex=True)

for i, b in enumerate(boundaries):
    mdf = metrics_dict[b]
    axes[i].bar(mdf['bin'], mdf['tick_count'], alpha=0.7, label='Tick count')
    axes[i].plot(mdf['avg_mid'], color='r', label='Avg midprice')
    axes[i].fill_between(mdf['bin'], mdf['avg_mid']-mdf['std_mid'], mdf['avg_mid']+mdf['std_mid'], color='r', alpha=0.2)
    axes[i].set_title(f'Boundary {b//60} min')
    axes[i].legend(loc='upper left')

# Show table of metrics for 1-min boundary as example
print(metrics_dict[60].head(10))

plt.tight_layout()
plt.show()
