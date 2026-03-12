
# Tick Envelope System Identification App
# ----------------------------------
# This app reads tick-level CSV data, constructs ADSR-like envelopes synced
# to minute boundaries, extracts features, and labels envelopes as leading to
# winning or losing candles.
#
# Philosophy:
# - Candles are human-imposed sampling artifacts
# - Market is a continuous chaotic signal
# - ADSR envelopes model how energy/information flows across boundaries
# - This is system identification applied to market microstructure

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    time_col: str = "timestamp"      # epoch seconds or pandas-parsable
    price_col: str = "ask"
    minute_sec: int = 60
    resample_rule: str = "1min"


# -----------------------------
# Utilities
# -----------------------------

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    return df


def minute_boundaries(times: pd.Series) -> pd.Series:
    return times.dt.floor('1min')


# -----------------------------
# Candle construction
# -----------------------------

def build_candles(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    candles = (
        df
        .set_index(cfg.time_col)
        [cfg.price_col]
        .resample(cfg.resample_rule)
        .ohlc()
        .dropna()
    )
    candles['return'] = candles['close'] - candles['open']
    candles['label'] = np.sign(candles['return'])  # +1 win, -1 loss, 0 flat
    return candles


# -----------------------------
# ADSR Envelope Extraction
# -----------------------------

def extract_adsr_envelopes(df: pd.DataFrame, candles: pd.DataFrame, cfg: Config):
    """
    ADSR definition (synced to minute boundaries):
    - Attack: end of previous candle -> minute boundary
    - Decay: minute boundary -> early part of current candle
    - Sustain: middle of current candle
    - Release: implicit, start of next attack
    """
    envelopes = []

    df = df.copy()
    df['minute'] = minute_boundaries(df[cfg.time_col])

    grouped = df.groupby('minute')
    candle_index = candles.index

    for i in range(1, len(candle_index) - 1):
        prev_min = candle_index[i - 1]
        cur_min = candle_index[i]
        next_min = candle_index[i + 1]

        prev_ticks = grouped.get_group(prev_min) if prev_min in grouped.groups else None
        cur_ticks = grouped.get_group(cur_min) if cur_min in grouped.groups else None

        if prev_ticks is None or cur_ticks is None:
            continue

        # Attack: last price movement of previous candle
        attack = prev_ticks[cfg.price_col].iloc[-1] - prev_ticks[cfg.price_col].iloc[0]

        # Decay: early volatility of current candle (first 20%)
        n_decay = max(1, int(0.2 * len(cur_ticks)))
        decay = cur_ticks[cfg.price_col].iloc[:n_decay].std()

        # Sustain: mid-candle volatility (middle 60%)
        start = int(0.2 * len(cur_ticks))
        end = int(0.8 * len(cur_ticks))
        sustain = cur_ticks[cfg.price_col].iloc[start:end].std()

        # Envelope vector
        env = {
            'minute': cur_min,
            'attack': attack,
            'decay': decay,
            'sustain': sustain,
            'label': candles.loc[cur_min, 'label']
        }
        envelopes.append(env)

    return pd.DataFrame(envelopes)


# -----------------------------
# Feature Space & Matching
# -----------------------------

def normalize_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sigma = out[c].std() + 1e-9
        out[c] = (out[c] - mu) / sigma
    return out


def simple_prototype_match(df: pd.DataFrame, feature_cols: List[str]):
    """
    Create average envelopes for winning vs losing candles
    and compute distance to prototypes
    """
    win_proto = df[df['label'] > 0][feature_cols].mean().values
    loss_proto = df[df['label'] < 0][feature_cols].mean().values

    distances = []
    for _, row in df.iterrows():
        v = row[feature_cols].values
        d_win = np.linalg.norm(v - win_proto)
        d_loss = np.linalg.norm(v - loss_proto)
        distances.append(d_win - d_loss)

    df['proto_score'] = distances
    return df


# -----------------------------
# Main App Logic
# -----------------------------

def run(csv_path: str):
    cfg = Config()

    # Load CSV
    df = pd.read_csv(csv_path)
    print("CSV columns:", df.columns.tolist())
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Controleer op NaT
    print("NaT in timestamp:", df['timestamp'].isna().sum())
    df = ensure_datetime(df, cfg.time_col)
    df = df.sort_values(cfg.time_col)

    # Candles
    candles = build_candles(df, cfg)

    # Envelopes
    env = extract_adsr_envelopes(df, candles, cfg)

    # Normalize & match
    features = ['attack', 'decay', 'sustain']
    env_n = normalize_features(env, features)
    env_scored = simple_prototype_match(env_n, features)

    print("=== Envelope Summary ===")
    print(env_scored.head())

    print("\nPrototype means:")
    print(env.groupby('label')[features].mean())

    return env_scored


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    # Example usage:
    # python app.py ticks.csv
    import sys
    demo_tick_file = "trading_cloud/tick-data/BTCUSD/csv/BTCUSD_2025-04-01_0200GMT_merged_data_corrected.csv"
    run(demo_tick_file)
    sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python app.py <tick_csv>")
    else:
        run(sys.argv[1])
