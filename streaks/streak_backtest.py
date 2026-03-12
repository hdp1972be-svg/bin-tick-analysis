
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------

GRID_SPACING = 0.05      # y: afstand tussen gridlijnen (in dollars)
GRID_OFFSET  = 0        # x: start afstand vanaf huidige prijs
GRID_LEVELS  = 20        # aantal levels boven/onder prijs

# ---------------------------------------------------------
# LOAD CSV
# ---------------------------------------------------------

df = pd.read_csv(
    "trading_cloud/tick-data/BTCUSD/csv/BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv",
#    "BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv"
    parse_dates=["timestamp"]
)

# midprijs berekenen
df["mid"] = (df["ask"] + df["bid"]) / 2.0

# timestamp in µs resolutie (nodig voor interarrival)
df["ts_us"] = df["timestamp"].astype("int64") // 1000


# ---------------------------------------------------------
# GRID GENERATOR
# ---------------------------------------------------------

def generate_grid(midprice):
    """Return arrays van buy en sell stop-levels."""
    buy_levels  = np.array([midprice - GRID_OFFSET - i*GRID_SPACING for i in range(1, GRID_LEVELS+1)])
    sell_levels = np.array([midprice + GRID_OFFSET + i*GRID_SPACING for i in range(1, GRID_LEVELS+1)])
    return buy_levels, sell_levels


# ---------------------------------------------------------
# SIMPLE MATCHING (alles filled)
# ---------------------------------------------------------

trade_log = []
last_hit_ts = None
volatility_series = []

# initialize grid
mid0 = df["mid"].iloc[0]
buy_grid, sell_grid = generate_grid(mid0)


# ---------------------------------------------------------
# LOOP OVER TICKS
# ---------------------------------------------------------

for idx, row in df.iterrows():
    mid   = row["mid"]
    ts_us = row["ts_us"]

    # check BUY stop hits (prijs valt erdoor)
    hits_buy = buy_grid[mid <= buy_grid]
    for level in hits_buy:
        trade_log.append({
            "timestamp_us": ts_us,
            "side": "BUY",
            "price": float(level),
            "tick_mid": mid
        })

        if last_hit_ts is not None:
            dt = (ts_us - last_hit_ts) / 1e6  # Δt in seconden
            inst_vol = 1.0 / dt if dt > 0 else np.nan
            volatility_series.append(inst_vol)

        last_hit_ts = ts_us

        # regenerate entire grid starting from new midprice
        buy_grid, sell_grid = generate_grid(mid)

    # check SELL stop hits (prijs stijgt erdoor)
    hits_sell = sell_grid[mid >= sell_grid]
    for level in hits_sell:
        trade_log.append({
            "timestamp_us": ts_us,
            "side": "SELL",
            "price": float(level),
            "tick_mid": mid
        })

        if last_hit_ts is not None:
            dt = (ts_us - last_hit_ts) / 1e6
            inst_vol = 1.0 / dt if dt > 0 else np.nan
            volatility_series.append(inst_vol)

        last_hit_ts = ts_us

        # regenerate grid so that we always probe the new zone
        buy_grid, sell_grid = generate_grid(mid)


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

print("Aantal virtuele trades:", len(trade_log))
print("Laatste 10 trades:")
for t in trade_log[-10:]:
    print(t)

print("\nLaatste 20 instant volatiliteitswaardes:")
print(volatility_series[-20:])
