import math
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

print("="*60)
print("Data Cleaning Pipeline")
print("="*60)

# Load raw data
print("\n📂 Loading raw data...")
df = pd.read_csv('data/raw/combined_all_seasons.csv')
print(f"Loaded {len(df)} rows")

# Remove duplicates based on player, season, and round
print("\n⏳ Cleaning data...")
cleaning_steps = [
    ("Removing duplicates", lambda d: d.drop_duplicates(subset=['element', 'season', 'round'])),
    ("Converting kickoff_time", lambda d: d.assign(kickoff_time=pd.to_datetime(d['kickoff_time'], errors='coerce'))),
    ("Converting was_home to int", lambda d: d.assign(was_home=d['was_home'].astype(int))),
]

for step_name, step_func in tqdm(cleaning_steps, desc="  Cleaning steps", leave=True):
    df = step_func(df)

# Impute missing expected stats and add missingness flags
print("\n⏳ Imputing missing values...")
expected_cols = ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']
for col in tqdm(expected_cols, desc="  Expected stats", leave=True):
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
print("\n💾 Saving cleaned data...")
df.to_csv('data/processed/cleaned_all_seasons.csv', index=False)
print(f"✅ Cleaned data saved to data/processed/cleaned_all_seasons.csv")
print(f"   Total rows: {len(df)}")