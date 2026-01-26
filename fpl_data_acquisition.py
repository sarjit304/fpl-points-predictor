"""
FPL Data Acquisition Pipeline

Fetches player and gameweek data from the Fantasy Premier League API,
plus historical seasons from the vaastav/Fantasy-Premier-League GitHub repo.

Usage:
    python fpl_data_acquisition.py

Functions can also be imported individually:
    from fpl_data_acquisition import download_historical_seasons
"""

import requests
import pandas as pd
import os
import glob
import time

# =============================================================================
# Configuration - Adjust these paths as needed
# =============================================================================
OUTPUT_DIR = "./DSCI_611_project"
GAMEWEEK_DIR = f"{OUTPUT_DIR}/gameweek_data"
HISTORICAL_DIR = f"{OUTPUT_DIR}/historical"
ALL_GAMEWEEKS_FILE = f"{OUTPUT_DIR}/all_gameweeks.csv"

# FPL API endpoints
API_BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{API_BASE}/bootstrap-static/"


def get_all_players():

   
    print("Fetching data from FPL API...")
    response = requests.get(BOOTSTRAP_URL)

    if response.status_code != 200:
        print(f"Error: Failed to fetch data (status code: {response.status_code})")
        return None

    data = response.json()
    print(f"Successfully fetched data from FPL API")

    return data

def process_player_data(data):
  
    # Extract players list
    players = data['elements']
    teams = {team['id']: team['name'] for team in data['teams']}

    # Position mapping
    positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    # Create dataframe
    players_df = pd.DataFrame(players)

    # Add team_name and position columns using the mappings
    players_df['team_name'] = players_df['team'].map(teams)
    players_df['position'] = players_df['element_type'].map(positions)

    # Select key columns
    key_columns = [
        # Player identifiers
        'id',                   
        'web_name',              
        'first_name',
        'second_name',
        'team_name',
        'position',
         # Match Outcomes (Cumulative)
        'minutes',               
        'goals_scored',
        'assists',
        'clean_sheets',
        'goals_conceded',
        'yellow_cards',
        'red_cards',
        'saves',
        # FPL Points
        'total_points', 
        'bps',                   # FPL bonus points
        # FPL game calculated stats
        'bonus',
        'defensive_contribution',                 
        'influence',             
        'creativity',            
        'threat',                
        'ict_index',  
        # Opta calculated statistics           
        'expected_goals',        # xG
        'expected_assists',      # xA
        'expected_goal_involvements',  # xGI
        'expected_goals_conceded'     # xGC
    ]

    players_clean = players_df[key_columns].copy()


    return players_clean

def get_player_gameweek_history(player_id):

    url = f"{API_BASE}/element-summary/{player_id}/"

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: Failed to fetch data for player {player_id} (status code: {response.status_code})")
        return None

    data = response.json()

    # 'history' contains gameweek-by-gameweek data for current season
    history = data.get('history', [])

    if not history:
        print(f"No gameweek history found for player {player_id}")
        return None

    history_df = pd.DataFrame(history)

        # Select key columns
    key_columns = [
        # Player identifiers
        'element',    
        # Match identifiers               
        'round',                    #matchweek
        'kickoff_time',
        'opponent_team',
        'was_home',              
         # Match Outcomes (Cumulative)
        'minutes',               
        'goals_scored',
        'assists',
        'clean_sheets',
        'goals_conceded',
        'yellow_cards',
        'red_cards',
        'saves',
        # FPL Points
        'total_points', 
        'bps',                   # FPL bonus points
        # FPL game calculated stats
        'bonus',
        'defensive_contribution',                 
        'influence',             
        'creativity',            
        'threat',                
        'ict_index',  
        # Opta calculated statistics           
        'expected_goals',        # xG
        'expected_assists',      # xA
        'expected_goal_involvements',  # xGI
        'expected_goals_conceded'     # xGC
    ]

    history_df_clean = history_df[key_columns].copy()

    return history_df_clean

def save_gameweek_csv(players_df, gameweek, output_folder):
    """Fetch all player data for a specific gameweek and save to CSV."""
    os.makedirs(output_folder, exist_ok=True)

    all_gw_data = []

    for i, player_id in enumerate(players_df['id']):
        history_df = get_player_gameweek_history(player_id)

        if history_df is not None:
            gw_data = history_df[history_df['round'] == gameweek]
            all_gw_data.append(gw_data)

        
        if i % 100 == 0:
            print(f"Fetched {i}/{len(players_df)} players for GW{gameweek}")

        time.sleep(0.1)  # Rate limiting

    if not all_gw_data:
        print(f"No data found for gameweek {gameweek}")
        return None
    
    print(f"Fetched {len(all_gw_data)} players for GW{gameweek}")

    gw_df = pd.concat(all_gw_data, ignore_index=True)

    filepath = os.path.join(output_folder, f"gw{gameweek}.csv")
    gw_df.to_csv(filepath, index=False)
    print(f"Saved gw{gameweek}.csv records to {filepath}")

    return gw_df


def get_current_gameweek(data):
    """Get the latest finished gameweek from bootstrap data."""
    events = data.get('events', [])
    for event in reversed(events):
        if event.get('finished', False):
            return event['id']
    return 0


def download_missing_gameweeks(players_df, data, output_folder):
    """Check folder for existing CSVs and download any missing gameweeks."""
    os.makedirs(output_folder, exist_ok=True)

    # Get current gameweek from API
    current_gw = get_current_gameweek(data)
    print(f"Current gameweek: {current_gw}")

    # Find existing gameweek CSVs
    existing_files = glob.glob(os.path.join(output_folder, "gw*.csv"))
    existing_gws = set()
    for f in existing_files:
        # Extract gameweek number from filename (e.g., "gw12.csv" -> 12)
        filename = os.path.basename(f)
        gw_num = int(filename.replace("gw", "").replace(".csv", ""))
        existing_gws.add(gw_num)

    print(f"Found existing gameweeks: {sorted(existing_gws)}")

    # Find missing gameweeks
    all_gws = set(range(1, current_gw + 1))
    missing_gws = sorted(all_gws - existing_gws)

    if not missing_gws:
        print("All gameweeks up to date!")
        return

    print(f"Missing gameweeks: {missing_gws}")

    # Download missing gameweeks
    for gw in missing_gws:
        print(f"\nDownloading GW{gw}...")
        save_gameweek_csv(players_df, gw, output_folder)


def merge_gameweek_csvs(folder_path, output_file):
    """Merge all gameweek CSVs into one combined CSV."""
    csv_files = glob.glob(os.path.join(folder_path, "gw*.csv"))

    if not csv_files:
        print(f"No gameweek CSVs found in {folder_path}")
        return None

    all_dfs = []
    for file in sorted(csv_files):
        df = pd.read_csv(file)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(['element', 'round']).reset_index(drop=True)

    combined_df.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} gameweeks → {output_file} ({len(combined_df)} rows)")

    return combined_df


def download_historical_seasons(output_folder, seasons=None):
    """Download historical season data from GitHub repo."""
    if seasons is None:
        seasons = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

    os.makedirs(output_folder, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

    for season in seasons:
        filepath = os.path.join(output_folder, f"{season}.csv")

        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"{season}.csv already exists, skipping")
            continue

        url = f"{base_url}/{season}/gws/merged_gw.csv"
        print(f"Downloading {season}...")

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Failed to download {season} (status code: {response.status_code})")
            continue

        # Save CSV
        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(f"Saved {filepath}")

    print("\nHistorical download complete!")


if __name__ == "__main__":
    # Fetch player data & cumulative performance
    print("\nFetching Player Data")
    data = get_all_players()

    # Process player data
    print("\nProcessing player data...")
    players_df = process_player_data(data)
    print(f"Processed {len(players_df)} players")

    # Create player-match performance table
    print("\nFetching Player Match Performance")
    download_missing_gameweeks(players_df, data, output_folder=GAMEWEEK_DIR)

    # Merge all gameweeks into one CSV
    print("\nMerging gameweek CSVs...")
    all_gameweeks_df = merge_gameweek_csvs(
        folder_path=GAMEWEEK_DIR,
        output_file=ALL_GAMEWEEKS_FILE
    )

    # Download historical seasons from GitHub
    print("\nDownloading historical seasons...")
    download_historical_seasons(output_folder=HISTORICAL_DIR)