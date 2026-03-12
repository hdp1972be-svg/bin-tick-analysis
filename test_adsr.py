
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
df = df.sort_values('timestamp')  # ensure chronological order

# Compute midprice
df['mid'] = (df['ask'] + df['bid']) / 2

# --- INTERARRIVAL TIMES ---
df['delta_s'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

# --- EVENT DENSITY AROUND BOUNDARIES ---
def event_density(df, period_sec):
    """Compute number of ticks relative to period boundaries"""
    seconds_since_boundary = df['timestamp'].dt.second + df['timestamp'].dt.minute*60
    relative = seconds_since_boundary % period_sec
    counts, bins = np.histogram(relative, bins=period_sec, range=(0, period_sec))
    return counts, bins[:-1]

density_results = {}
for b in boundaries:
    counts, bins = event_density(df, b)
    density_results[b] = (counts, bins)

# --- ENTROPY CALCULATION (directional) ---
# simple up/down tick
df['tick_dir'] = np.sign(df['mid'].diff().fillna(0))
# discretize for histogram
hist, _ = np.histogram(df['tick_dir'], bins=[-1,0,1,2], density=True)
shannon_entropy = entropy(hist)

# --- SIMPLE ENVELOPE/ADSR USING CONVOLUTION ---
# let's use tick counts as "signal"
sig = density_results[60][0]  # using 1-min as example
# simple smoothing kernel as pseudo-envelope
kernel = np.concatenate([np.linspace(0,1,5), np.ones(50), np.linspace(1,0,5)])
envelope = convolve(sig, kernel, mode='same')

# --- PLOTS ---
fig, axes = plt.subplots(len(boundaries)+2, 1, figsize=(12,8), sharex=True)

for i, b in enumerate(boundaries):
    counts, bins = density_results[b]
    axes[i].bar(bins, counts, width=1, alpha=0.7)
    axes[i].set_title(f'Tick event density around {b//60} min boundary')
    axes[i].set_ylabel('Ticks')

axes[len(boundaries)].plot(sig, label='Tick counts (1-min)')
axes[len(boundaries)].plot(envelope, label='Envelope (ADSR-like)')
axes[len(boundaries)].set_title('ADSR-like envelope on 1-min density')
axes[len(boundaries)].legend()

axes[len(boundaries)+1].bar([-1,0,1], hist, alpha=0.7)
axes[len(boundaries)+1].set_title(f'Shannon Entropy of tick direction: {shannon_entropy:.3f}')

plt.tight_layout()
plt.show()
