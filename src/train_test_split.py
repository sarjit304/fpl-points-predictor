"""
Temporal Train/Test Split for FPL Points Prediction

Splits data chronologically to simulate real prediction scenario:
- Train on historical seasons + early current season (GW1-5)
- Test on remaining current season gameweeks (GW6+)

Early cutoff (GW5) ensures:
- New players have 5 games of history for MA5 features
- Large test set for robust evaluation

Data Leakage Prevention:
- Temporal split: train data is strictly BEFORE test data
- Features for test GW N were computed using data up to GW N-1
- At prediction time for GW N, you would know results through GW N-1
- Recency weights applied only to training data
"""

import pandas as pd
import numpy as np
import os

# Input dataset path (reads from feature-engineered data)
INPUT_PATH = "data/processed/features_all_seasons.csv"

# Output paths
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Configuration
CURRENT_SEASON = "2025-26"
CUTOFF_GW = 5  # Fixed cutoff: train on GW1-5, test on GW6+
DECAY_RATE = 0.15  # Exponential decay for sample weights


def get_timeline_value(season, gameweek, season_order):
    """Convert season + gameweek to a single timeline value."""
    return season_order.get(season, 0) * 38 + gameweek


def add_recency_weights(df, decay_rate, cutoff_timeline, season_order):
    """
    Add exponential decay weights based on recency.

    More recent gameweeks receive higher weights.
    Weights are relative to cutoff_timeline (last training gameweek).

    Formula: weight = exp(-decay_rate * gameweeks_ago)
    """
    df = df.copy()

    df['_timeline'] = df.apply(
        lambda row: get_timeline_value(row['season'], row['round'], season_order),
        axis=1
    )

    # Gameweeks ago from the cutoff point
    df['_gw_ago'] = cutoff_timeline - df['_timeline']

    # Exponential decay weight
    df['sample_weight'] = np.exp(-decay_rate * df['_gw_ago'])

    # Normalize so weights sum to number of samples (optional)
    # df['sample_weight'] = df['sample_weight'] / df['sample_weight'].sum() * len(df)

    df = df.drop(columns=['_timeline', '_gw_ago'])

    return df


def temporal_split(df, current_season, cutoff_gw, season_order):
    """
    Split data temporally for next-gameweek prediction.

    Train: All historical seasons + current season up to cutoff_gw
    Test:  Current season GW(cutoff_gw + 1) to GW(max)

    Fixed early cutoff ensures:
    - New players have enough games for MA features
    - Larger test set for more robust evaluation

    Returns:
    - train_df: Training data
    - test_df: Test data
    - max_gw: Maximum gameweek in test set
    - cutoff_timeline: Timeline value of cutoff for weight calculation
    """
    df = df.sort_values(['season', 'round', 'element'])

    # Find the max completed gameweek in current season
    current_df = df[df['season'] == current_season]
    max_gw = int(current_df['round'].max())
    test_start_gw = cutoff_gw + 1

    # Calculate cutoff timeline for weight calculation
    cutoff_timeline = get_timeline_value(current_season, cutoff_gw, season_order)

    print(f"Current season: {current_season}")
    print(f"Max gameweek available: {max_gw}")
    print(f"Train: All historical + {current_season} GW1-{cutoff_gw}")
    print(f"Test: {current_season} GW{test_start_gw}-{max_gw}")

    # Split: test is current season GWs after cutoff
    is_test = (df['season'] == current_season) & (df['round'] > cutoff_gw)

    train_df = df[~is_test].copy()
    test_df = df[is_test].copy()

    return train_df, test_df, max_gw, cutoff_timeline


if __name__ == "__main__":
    
    print("Temporal Train/Test Split")
    
    # Season ordering for timeline calculation
    season_order = {
        '2020-21': 0, '2021-22': 1, '2022-23': 2,
        '2023-24': 3, '2024-25': 4, '2025-26': 5
    }

    # Load feature-engineered dataset
    print("\nLoading feature-engineered dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    # Temporal train/test split
    print(f"\nPerforming temporal split (cutoff at GW{CUTOFF_GW})...")
    train_df, test_df, max_gw, cutoff_timeline = temporal_split(
        df, CURRENT_SEASON, CUTOFF_GW, season_order
    )

    # Add recency weights to training data only
    print(f"\nAdding recency weights to training data (decay_rate={DECAY_RATE})...")
    train_df = add_recency_weights(train_df, DECAY_RATE, cutoff_timeline, season_order)

    # Drop rows with NaN in lag/rolling features
    # (first few GWs per player don't have enough history)
    feature_cols = [c for c in train_df.columns if '_lag' in c or '_ma' in c]
    if feature_cols:
        before_train = len(train_df)
        before_test = len(test_df)
        train_df = train_df.dropna(subset=feature_cols)
        test_df = test_df.dropna(subset=feature_cols)
        print(f"\nDropped rows with insufficient history:")
        print(f"  Train: {before_train} → {len(train_df)} ({before_train - len(train_df)} dropped)")
        print(f"  Test: {before_test} → {len(test_df)} ({before_test - len(test_df)} dropped)")

    # Create output directory if needed
    os.makedirs("data", exist_ok=True)

    # Save files
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Split completed successfully")
    print(f"\nTrain set: {len(train_df)} rows")
    print(f"  Seasons: {sorted(train_df['season'].unique())}")
    print(f"  {CURRENT_SEASON} gameweeks: 1 to {CUTOFF_GW}")
    print(f"\nTest set: {len(test_df)} rows")
    print(f"  Season: {CURRENT_SEASON}")
    print(f"  Gameweeks: {CUTOFF_GW + 1} to {max_gw}")
    print(f"\nFiles saved:")
    print(f"  {TRAIN_PATH}")
    print(f"  {TEST_PATH}")
