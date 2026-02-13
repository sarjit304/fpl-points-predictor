import pandas as pd
import glob
import os


def build_combined_dataset(current_season_file, historical_folder, output_file):
    """
    Combine current season with historical data for active players only.

    - Left join: only players active in current season
    - Stacked long format with season identifier
    - Aligned to common column set
    """
    
    # Define the columns we want in the final dataset
    final_columns = [
        'name', 'element', 'season', 'position', 'team',
        'round', 'kickoff_time', 'opponent_team', 'was_home',
        'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'goals_conceded', 'yellow_cards', 'red_cards', 'saves',
        'total_points', 'bps', 'bonus',
        'influence', 'creativity', 'threat', 'ict_index',
        'expected_goals', 'expected_assists',
        'expected_goal_involvements', 'expected_goals_conceded',
        'defensive_contribution'
    ]

    # Load current season data
    print("Loading current season data...")
    current_df = pd.read_csv(current_season_file)
    current_df['season'] = CURRENT_SEASON

    # Get list of active player names
    active_players = set(current_df['name'].unique())
    print(f"Found {len(active_players)} active players in current season")

    # Load and filter historical data
    historical_files = glob.glob(os.path.join(historical_folder, "*.csv"))
    all_dfs = [current_df]

    for file in sorted(historical_files):
        season = os.path.basename(file).replace('.csv', '')
        print(f"Loading {season}...")

        hist_df = pd.read_csv(file)
        hist_df['season'] = season

        # Filter to active players only
        hist_df = hist_df[hist_df['name'].isin(active_players)]
        print(f"  → {len(hist_df)} rows for active players")

        all_dfs.append(hist_df)

    # Combine all dataframes
    print("\nCombining all seasons...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure all final columns exist (fill missing with NaN)
    for col in final_columns:
        if col not in combined_df.columns:
            combined_df[col] = None

    # Select and order columns
    combined_df = combined_df[final_columns].copy()

    # Sort by player name, season, round
    combined_df = combined_df.sort_values(['name', 'season', 'round']).reset_index(drop=True)

    # Save
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined dataset: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Unique players: {combined_df['name'].nunique()}")
    print(f"Seasons: {sorted(combined_df['season'].unique())}")

    return combined_df


if __name__ == "__main__":
    OUTPUT_FILE = "data/raw/combined_all_seasons.csv"
    CURRENT_SEASON = "2025-26"
    CURRENT_SEASON_FILE = "DSCI_611_project/all_gameweeks.csv"
    HISTORICAL_FOLDER = "DSCI_611_project/historical/"

    combined_df = build_combined_dataset(CURRENT_SEASON_FILE, HISTORICAL_FOLDER, OUTPUT_FILE)