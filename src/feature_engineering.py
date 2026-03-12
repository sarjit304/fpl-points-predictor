"""
Feature Engineering for FPL Points Prediction

Creates lag and rolling average features for player performance prediction.

Data Leakage Prevention:
- All features use shift() to ensure we only use PAST gameweek data
- For any row at GW N, features are computed from GW N-1 and earlier
- This mirrors real prediction: when predicting GW N, you know results up to GW N-1
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Input/Output paths
INPUT_PATH = "data/processed/cleaned_all_seasons.csv"
OUTPUT_PATH = "data/processed/features_all_seasons.csv"

# Configuration
ROLLING_WINDOW = 5
TIME_DECAY_RATE = 0.1  # Exponential decay for weighted rolling average (0 = no decay)


def add_lag_features(df, columns, lag=1):
    """
    Add lag features for specified columns.

    For each row at gameweek N, the lag1 feature contains
    the value from gameweek N-1 (the previous gameweek).

    Computed per player to handle player-specific history.
    """
    df = df.sort_values(['element', 'season', 'round'])

    for col in columns:
        if col in df.columns:
            df[f'{col}_lag{lag}'] = df.groupby('element')[col].shift(lag)

    return df


def add_rolling_averages(df, columns, window=5, decay_rate=0.0):
    """
    Add rolling average features for specified columns.

    For each row at gameweek N, computes the average of gameweeks
    N-window to N-1 (excludes current gameweek via shift).

    Parameters:
    - window: Number of previous gameweeks to include
    - decay_rate: If > 0, applies exponential decay weighting where
                  recent gameweeks get higher weight: exp(-decay_rate * gw_ago)
    """
    df = df.sort_values(['element', 'season', 'round'])

    for col in columns:
        if col not in df.columns:
            continue

        if decay_rate > 0:
            # Time-weighted rolling average (more weight to recent games)
            df[f'{col}_ma{window}'] = (
                df.groupby('element')[col]
                .apply(lambda x: _weighted_rolling_mean(x, window, decay_rate))
                .reset_index(level=0, drop=True)
            )
        else:
            # Simple rolling average
            df[f'{col}_ma{window}'] = (
                df.groupby('element')[col]
                .shift(1)  # Exclude current gameweek
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    return df


def _weighted_rolling_mean(series, window, decay_rate):
    """
    Compute exponentially weighted rolling mean.

    More recent observations get higher weights:
    weight[i] = exp(-decay_rate * (window - 1 - i))

    Shifted by 1 to exclude current gameweek.
    """
    result = pd.Series(index=series.index, dtype=float)
    values = series.values

    for i in range(len(values)):
        if i == 0:
            result.iloc[i] = np.nan
            continue

        # Get previous values (shift by 1 to exclude current)
        start_idx = max(0, i - window)
        end_idx = i  # Excludes current (i)
        window_values = values[start_idx:end_idx]

        if len(window_values) == 0:
            result.iloc[i] = np.nan
            continue

        # Create weights (most recent = highest weight)
        n = len(window_values)
        weights = np.exp(-decay_rate * np.arange(n - 1, -1, -1))
        weights = weights / weights.sum()  # Normalize

        result.iloc[i] = np.sum(window_values * weights)

    return result


if __name__ == "__main__":
    print("="*60)
    print("Feature Engineering Pipeline")
    print("="*60)

    # Load cleaned dataset
    print("\n📂 Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows, {df['element'].nunique()} unique players")

    # Sort chronologically by player
    df = df.sort_values(['element', 'season', 'round'])

    # Add lag features (previous gameweek values)
    print("\n⏳ Adding lag features (previous gameweek values)...")
    lag_columns = ['total_points', 'expected_goal_involvements', 'minutes']
    for col in tqdm(lag_columns, desc="  Creating lag features", leave=True):
        df = add_lag_features(df, [col], lag=1)

    # Add rolling averages (with optional time decay)
    print(f"\n⏳ Adding {ROLLING_WINDOW}-game rolling averages...")
    if TIME_DECAY_RATE > 0:
        print(f"  Using time-weighted decay (rate={TIME_DECAY_RATE})")
    rolling_columns = ['total_points', 'expected_goal_involvements', 'ict_index', 'bps']
    for col in tqdm(rolling_columns, desc="  Creating rolling averages", leave=True):
        df = add_rolling_averages(df, [col], window=ROLLING_WINDOW, decay_rate=TIME_DECAY_RATE)

    # Create output directory if needed
    os.makedirs("data/processed", exist_ok=True)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "="*60)
    print("Feature engineering complete")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("="*60)
