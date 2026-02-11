import math
import os
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('data/raw/combined_all_seasons.csv')

# Remove duplicates based on player, season, and round
df = df.drop_duplicates(subset=['element', 'season', 'round'])

# Convert kickoff_time to datetime
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')

# Ensure was_home is integer (0/1)
df['was_home'] = df['was_home'].astype(int)

# Impute missing expected stats and add missingness flags
for col in ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']:
    if col in df.columns:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

# Handle defensive_contribution if present (drop or impute)
if 'defensive_contribution' in df.columns:
    df['defensive_contribution_missing'] = df['defensive_contribution'].isnull().astype(int)
    df['defensive_contribution'] = df['defensive_contribution'].fillna(0)

# Drop rows with missing critical identifiers
df = df.dropna(subset=['element', 'season', 'round'])

# Optional: Reset index
df = df.reset_index(drop=True)

# Create output directory if needed
os.makedirs("data/processed", exist_ok=True)

# Save cleaned data
df.to_csv('data/processed/cleaned_all_seasons.csv', index=False)
print("Cleaned data saved to data/processed/cleaned_all_seasons.csv")