## Overview

This dataset contains Fantasy Premier League (FPL) player performance data spanning multiple seasons. It combines current season (2025-26) gameweek-level statistics with historical data dating back to the 2020-21 season. The dataset is designed for machine learning and statistical analysis of player performance, enabling prediction models and trend analysis.

**Key Features:**
- Gameweek-by-gameweek player statistics for active players
- Match context (home/away, opponent, kickoff time)
- Traditional statistics (goals, assists, clean sheets)
- Advanced metrics (xG, xA, xGI, xGC, ICT index)
- FPL-specific data (points, bonus, BPS)
- Historical coverage for longitudinal analysis

**Dataset Scope:**
- **Current Season:** All available gameweeks for 2025-26 season
- **Historical Seasons:** 2020-21, 2021-22, 2022-23, 2023-24, 2024-25
- **Player Coverage:** Only players active in the current season (with their historical records where available)

---

## Data Sources

### Primary Source: FPL Official API
- **Base URL:** `https://fantasy.premierleague.com/api`
- **Endpoints Used:**
  - `bootstrap-static/` - Player metadata, team mappings, season overview
  - `element-summary/{player_id}/` - Individual player gameweek history
- **Update Frequency:** Real-time (updated after each gameweek completes)
- **Access:** Public, no authentication required
- **Data Format:** JSON

### Secondary Source: vaastav/Fantasy-Premier-League GitHub Repository
- **Repository:** https://github.com/vaastav/Fantasy-Premier-League
- **Data Path:** `data/{season}/gws/merged_gw.csv`
- **Coverage:** Historical seasons (2020-21 through 2024-25)
- **Update Frequency:** Season-end finalized data
- **Access:** Public GitHub repository
- **Data Format:** CSV

### Third-Party Data Providers
- **Opta Sports:** Expected statistics (xG, xA, xGI, xGC) provided via FPL API
- **FPL Algorithm:** Calculated metrics (ICT index, influence, creativity, threat)

---

## Data Collection Pipeline

### Pipeline Architecture

The data acquisition follows a four-stage process:

#### Stage 1: Current Season Player Data Extraction
1. Fetch bootstrap data from FPL API (`bootstrap-static/`)
2. Extract player list with metadata (names, teams, positions)
3. Map team IDs to team names and position codes to abbreviations
4. Generate player dataframe with cumulative season statistics

#### Stage 2: Gameweek-Level Data Collection
1. For each active player, fetch individual gameweek history (`element-summary/{player_id}/`)
2. Extract match-level performance for each completed gameweek
3. Add player metadata (name, position, team) to each record
4. Implement rate limiting (0.1 second delay between requests)
5. Save individual gameweek CSVs to `gameweek_data/` folder

#### Stage 3: Incremental Update Logic
1. Scan `gameweek_data/` folder for existing CSV files
2. Identify most recent completed gameweek from API
3. Download only missing gameweeks (incremental updates)
4. Merge all gameweek CSVs into single `all_gameweeks.csv` file

#### Stage 4: Historical Data Integration
1. Download historical season files from GitHub repository
2. Filter historical data to include only players active in current season
3. Standardize column names and data types across all seasons
4. Combine current season with historical data
5. Generate final `combined_all_seasons.csv` with season identifier

### Rate Limiting and API Etiquette
- **Delay:** 0.1 seconds between player requests
- **Batch Processing:** 100-player progress updates
- **Error Handling:** HTTP status code validation with retry logic
- **Incremental Updates:** Avoid re-downloading existing gameweeks

---

## Directory Structure

```
DSCI_611_project/
│
├── gameweek_data/                    # Individual gameweek CSV files
│   ├── gw1.csv                       # Gameweek 1 data for all players
│   ├── gw2.csv                       # Gameweek 2 data for all players
│   ├── gw3.csv
│   └── ...                           # Up to current completed gameweek
│
├── historical/                       # Historical season data from GitHub
│   ├── 2020-21.csv                   # Complete 2020-21 season data
│   ├── 2021-22.csv                   # Complete 2021-22 season data
│   ├── 2022-23.csv                   # Complete 2022-23 season data
│   ├── 2023-24.csv                   # Complete 2023-24 season data
│   └── 2024-25.csv                   # Complete 2024-25 season data
│
├── all_gameweeks.csv                 # Merged current season (all GWs)
└── combined_all_seasons.csv          # Final dataset (current + historical)
```

**File Descriptions:**

- **`gameweek_data/gw{N}.csv`:** Individual gameweek files containing all player performances for that specific gameweek. Used for incremental updates and verification.

- **`historical/{season}.csv`:** Complete season data from GitHub repository, filtered to include only players active in the current season.

- **`all_gameweeks.csv`:** Concatenated current season data across all completed gameweeks. Sorted by player ID and gameweek number.

- **`combined_all_seasons.csv`:** Final production dataset combining current season with historical data. Includes season identifier column for temporal analysis.

---

## Data Dictionary

### Player Identifiers

| Field Name | Data Type | Description | Example | Source |
|------------|-----------|-------------|---------|--------|
| `name` | String | Player full name (first + last) | "Erling Haaland" | FPL API |
| `element` | Integer | Unique player ID (persistent across seasons) | 354 | FPL API |
| `position` | String | Player position abbreviation | "FWD", "MID", "DEF", "GK" | FPL API (mapped) |
| `team` | String | Current team name | "Man City" | FPL API (mapped) |
| `season` | String | Season identifier | "2025-26", "2024-25" | Generated |

### Match Context

| Field Name | Data Type | Description | Example | Source |
|------------|-----------|-------------|---------|--------|
| `round` | Integer | Gameweek number (1-38) | 15 | FPL API |
| `kickoff_time` | Datetime | Match kickoff timestamp (ISO 8601) | "2026-01-15T20:00:00Z" | FPL API |
| `opponent_team` | Integer | Opponent team ID | 3 | FPL API |
| `was_home` | Boolean | True if home match, False if away | True | FPL API |

### Match Performance Statistics

| Field Name | Data Type | Description | Range/Unit | Source |
|------------|-----------|-------------|------------|--------|
| `minutes` | Integer | Minutes played in match | 0-120 | FPL API |
| `goals_scored` | Integer | Goals scored | 0+ | FPL API |
| `assists` | Integer | Assists provided | 0+ | FPL API |
| `clean_sheets` | Integer | Clean sheet (1 if applicable, 0 otherwise) | 0-1 | FPL API |
| `goals_conceded` | Integer | Goals conceded (relevant for DEF/GK) | 0+ | FPL API |
| `yellow_cards` | Integer | Yellow cards received | 0-2 | FPL API |
| `red_cards` | Integer | Red cards received | 0-1 | FPL API |
| `saves` | Integer | Saves made (primarily GK) | 0+ | FPL API |

### FPL Points and Bonus System

| Field Name | Data Type | Description | Range/Unit | Source |
|------------|-----------|-------------|------------|--------|
| `total_points` | Integer | Total FPL points earned in gameweek | -3 to 30+ | FPL API |
| `bps` | Integer | Bonus Points System score (raw) | 0-100+ | FPL API |
| `bonus` | Integer | Actual bonus points awarded (0-3) | 0-3 | FPL API |

**Note on Bonus Points:** `bps` is the raw Bonus Points System score calculated from player actions. The top 3 BPS scorers in each match receive `bonus` points (3, 2, and 1 respectively).

### FPL Calculated Metrics (ICT Index)

| Field Name | Data Type | Description | Range/Unit | Source |
|------------|-----------|-------------|------------|--------|
| `influence` | Float | Player's impact on match result | 0-150+ | FPL Algorithm |
| `creativity` | Float | Chance creation and attacking contribution | 0-150+ | FPL Algorithm |
| `threat` | Float | Likelihood to score (shots, position) | 0-200+ | FPL Algorithm |
| `ict_index` | Float | Combined ICT metric (Influence + Creativity + Threat) / 10 | 0-30+ | FPL Algorithm |

**ICT Index Components:**
- **Influence:** Evaluates impact on match result (winning goals, key events)
- **Creativity:** Measures chance creation through passes, crosses, and assists
- **Threat:** Quantifies scoring likelihood based on shots and positioning
- **ICT Index:** Combined metric normalized to single value

### Expected Statistics (Opta Data)

| Field Name | Data Type | Description | Range/Unit | Source |
|------------|-----------|-------------|------------|--------|
| `expected_goals` | Float | Expected goals (xG) | 0.00-3.00+ | Opta via FPL |
| `expected_assists` | Float | Expected assists (xA) | 0.00-2.00+ | Opta via FPL |
| `expected_goal_involvements` | Float | xG + xA combined metric | 0.00-4.00+ | Opta via FPL |
| `expected_goals_conceded` | Float | Expected goals conceded (xGC) for defenders | 0.00-5.00+ | Opta via FPL |

**About Expected Statistics:**
- **xG (Expected Goals):** Probability-based metric evaluating shot quality (0.0 = impossible, 1.0 = certain goal)
- **xA (Expected Assists):** Probability that a pass leads to a goal
- **xGI (Expected Goal Involvements):** Combined attacking contribution (xG + xA)
- **xGC (Expected Goals Conceded):** Defensive metric for likelihood of conceding

### Additional Metrics

| Field Name | Data Type | Description | Range/Unit | Source |
|------------|-----------|-------------|------------|--------|
| `defensive_contribution` | Float | Defensive actions metric (tackles, blocks, clearances) | 0.00-100+ | FPL API |

**Data Availability Notes:**
- `defensive_contribution` introduced in recent seasons; may be NULL in historical data (2020-2023)
- All expected statistics (xG, xA, xGI, xGC) available from 2020-21 onwards
- Historical data may have slight schema variations (handled by standardization)

---

## Data Formats and Schemas

### File Format Specifications

**Format:** CSV (Comma-Separated Values)
**Encoding:** UTF-8
**Line Endings:** Unix-style (LF)
**Header Row:** Included in all files
**Delimiter:** Comma (,)
**Quote Character:** Double quote (") for fields containing commas

### Data Type Specifications

**Integer Fields:** Whole numbers without decimals
- Examples: `element`, `round`, `minutes`, `goals_scored`, `total_points`

**Float Fields:** Decimal numbers (up to 2 decimal places for most metrics)
- Examples: `expected_goals`, `influence`, `ict_index`

**String Fields:** Text values (UTF-8 encoded)
- Examples: `name`, `position`, `team`, `season`

**Boolean Fields:** True/False values
- `was_home`: True (home match) or False (away match)

**Datetime Fields:** ISO 8601 format
- `kickoff_time`: "YYYY-MM-DDTHH:MM:SSZ" (UTC timezone)

### Missing Value Handling

**NULL Values:** Represented as empty cells in CSV
**Zero vs. NULL:**
- `0` indicates measured zero (e.g., 0 goals scored)
- `NULL` indicates data not available or not applicable

**Common NULL Scenarios:**
- `defensive_contribution`: NULL in seasons before 2023-24
- `saves`: NULL for outfield players (not goalkeepers)
- Future gameweeks: All values NULL until match completion

---

## Data Quality and Validation

### Known Data Issues

1. **Defensive Contribution Field**
   - **Issue:** Field introduced mid-2023; NULL in earlier seasons
   - **Affected Seasons:** 2020-21 through 2022-23
   - **Recommendation:** Exclude from models requiring complete historical data

2. **Player Name Variations**
   - **Issue:** Some players have name changes (transfers, corrections)
   - **Example:** Player transferred mid-season may have duplicate entries under different team names
   - **Mitigation:** Use `element` (player ID) as primary identifier, not name

3. **Historical Filtering**
   - **Issue:** Historical data filtered to active players only
   - **Impact:** Players who retired or left the league are excluded
   - **Use Case:** Analysis limited to currently active players' career trajectories

4. **Incomplete Gameweeks**
   - **Issue:** Current gameweek may be partially complete (some matches finished, others pending)
   - **Recommendation:** Filter to fully completed gameweeks for analysis

### Data Validation Checks

**Automated Validation (implemented in pipeline):**
- HTTP status code verification (200 = success)
- JSON parsing validation
- Column existence checks before CSV generation
- Sort order verification (player ID → gameweek)

**Recommended Manual Validation:**
```python
import pandas as pd

# Load dataset
df = pd.read_csv('combined_all_seasons.csv')

# Check for missing critical fields
print("Missing player names:", df['name'].isnull().sum())
print("Missing gameweeks:", df['round'].isnull().sum())

# Validate data ranges
print("Minutes range:", df['minutes'].min(), "-", df['minutes'].max())
print("Valid positions:", df['position'].unique())

# Check for duplicates
duplicates = df[df.duplicated(subset=['element', 'season', 'round'], keep=False)]
print(f"Duplicate records: {len(duplicates)}")
```

---

## Usage Instructions

### Initial Setup

1. **Install Dependencies**
```bash
pip install requests pandas
```

2. **Configure Paths (Optional)**
Edit the configuration section in `fpl_data_acquisition.py`:
```python
OUTPUT_DIR = "./DSCI_611_project"  # Change if needed
GAMEWEEK_DIR = f"{OUTPUT_DIR}/gameweek_data"
HISTORICAL_DIR = f"{OUTPUT_DIR}/historical"
```

### Running the Pipeline

**Full Pipeline Execution:**
```bash
python fpl_data_acquisition.py
```

This will:
1. Fetch current season player data
2. Download missing gameweeks (incremental)
3. Merge gameweeks into `all_gameweeks.csv`
4. Download historical seasons (if not present)
5. Build combined dataset → `combined_all_seasons.csv`

**Expected Runtime:**
- Initial run: 15-20 minutes (all gameweeks + historical)
- Incremental update: 2-5 minutes (new gameweeks only)

### Using Individual Functions

```python
from fpl_data_acquisition import (
    get_all_players,
    process_player_data,
    download_historical_seasons,
    build_combined_dataset
)

# Fetch only current player data
data = get_all_players()
players_df = process_player_data(data)

# Download specific historical seasons
download_historical_seasons(
    output_folder='./historical',
    seasons=['2023-24', '2024-25']
)

# Build custom combined dataset
build_combined_dataset(
    current_season_file='all_gameweeks.csv',
    historical_folder='./historical',
    output_file='custom_combined.csv'
)
```

### Loading Data for Analysis

**Load Current Season Only:**
```python
import pandas as pd

current_df = pd.read_csv('DSCI_611_project/all_gameweeks.csv')
print(f"Rows: {len(current_df)}, Players: {current_df['element'].nunique()}")
```

**Load Combined Multi-Season Dataset:**
```python
combined_df = pd.read_csv('DSCI_611_project/combined_all_seasons.csv')

# Filter to specific season
season_2024 = combined_df[combined_df['season'] == '2024-25']

# Filter to specific player
haaland = combined_df[combined_df['name'] == 'Erling Haaland']

# Filter to high-scoring gameweeks
big_hauls = combined_df[combined_df['total_points'] >= 15]
```

---

## Update and Maintenance

### Regular Updates

**Recommended Update Frequency:** Weekly (after gameweek completion)

**Update Process:**
1. Run pipeline script: `python fpl_data_acquisition.py`
2. Script automatically detects and downloads only new gameweeks
3. Re-merges `all_gameweeks.csv` and `combined_all_seasons.csv`

**Update Timing:**
- FPL API updates ~2 hours after final match of gameweek
- Best practice: Run script Monday morning after weekend matches complete

### Incremental Update Logic

The pipeline intelligently handles incremental updates:

```python
# Script checks existing gameweek files
existing_gws = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example

# API reports current completed gameweek = 12
current_gw = 12

# Script downloads only missing gameweeks
missing_gws = [11, 12]  # Automatically calculated
```

**Benefits:**
- Faster update times (minutes vs. hours)
- Reduced API load (respectful data collection)
- Preserves existing validated data

### Historical Data Maintenance

**One-Time Download:** Historical seasons downloaded once and cached locally

**Re-download Only If:**
- Files deleted or corrupted
- New historical season added to GitHub repository (e.g., end of 2025-26 season)

**Manual Historical Update:**
```python
from fpl_data_acquisition import download_historical_seasons

# Download specific season
download_historical_seasons(
    output_folder='./DSCI_611_project/historical',
    seasons=['2025-26']  # Add newly completed season
)
```

### Troubleshooting

**Issue: "Error: Failed to fetch data (status code: 429)"**
- **Cause:** Rate limiting from FPL API
- **Solution:** Increase sleep delay in script: `time.sleep(0.2)`

**Issue: "No gameweek history found for player X"**
- **Cause:** Player hasn't played yet this season (new signing, injured)
- **Solution:** Expected behavior; script continues with other players

**Issue: Historical download fails**
- **Cause:** GitHub repository unavailable or URL changed
- **Solution:** Check repository status: https://github.com/vaastav/Fantasy-Premier-League

**Issue: Duplicate records in combined dataset**
- **Cause:** Multiple pipeline runs without cleanup
- **Solution:** Delete `combined_all_seasons.csv` and re-run pipeline

---

## Example Use Cases

### 1. Player Performance Prediction Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('DSCI_611_project/combined_all_seasons.csv')

# Feature engineering
df['points_per_minute'] = df['total_points'] / df['minutes']
df['xG_per_90'] = (df['expected_goals'] / df['minutes']) * 90

# Filter to players with meaningful minutes
df_model = df[df['minutes'] >= 60]

# Train-test split (use historical seasons for training)
train = df_model[df_model['season'] != '2025-26']
test = df_model[df_model['season'] == '2025-26']

# Predict next gameweek points
X_train = train[['expected_goals', 'expected_assists', 'minutes', 'ict_index']]
y_train = train['total_points']
```

### 2. Trend Analysis: xG vs. Actual Goals

```python
# Aggregate by player
player_stats = df.groupby('name').agg({
    'expected_goals': 'sum',
    'goals_scored': 'sum',
    'minutes': 'sum'
}).reset_index()

# Calculate over/underperformance
player_stats['xG_diff'] = player_stats['goals_scored'] - player_stats['expected_goals']
player_stats['goals_per_90'] = (player_stats['goals_scored'] / player_stats['minutes']) * 90

# Find overperformers
overperformers = player_stats.nlargest(10, 'xG_diff')
```

### 3. Home vs. Away Performance

```python
# Compare home/away stats
home_away = df.groupby(['name', 'was_home']).agg({
    'total_points': 'mean',
    'goals_scored': 'mean',
    'assists': 'mean'
}).reset_index()

# Pivot for easy comparison
comparison = home_away.pivot(index='name', columns='was_home', values='total_points')
comparison.columns = ['Away_Points', 'Home_Points']
comparison['Home_Advantage'] = comparison['Home_Points'] - comparison['Away_Points']
```

### 4. Position-Specific Analysis

```python
# Average points by position and gameweek
position_trends = df.groupby(['position', 'round']).agg({
    'total_points': 'mean',
    'expected_goal_involvements': 'mean'
}).reset_index()

# Identify best value positions
position_value = df.groupby('position')['total_points'].describe()
```

---

## Data Attribution and Licensing

### Data Sources

**Fantasy Premier League API**
- Provider: Premier League / Fantasy Premier League Official
- URL: https://fantasy.premierleague.com/api
- License: Public API (terms of use apply)
- Attribution: Data provided by the official Fantasy Premier League platform

**Historical Data Repository**
- Maintainer: vaastav (GitHub user)
- Repository: https://github.com/vaastav/Fantasy-Premier-League
- License: Open source (check repository for specific license)
- Attribution: Historical data aggregated and maintained by vaastav

**Opta Sports Data**
- Provider: Opta Sports (Stats Perform)
- Delivery: Via FPL API
- Attribution: Expected statistics (xG, xA, xGI, xGC) powered by Opta

### Usage Terms

**Commercial Use:**
- FPL API data subject to Premier League terms of service
- Check official FPL website for commercial usage restrictions
- Recommend using data for personal, educational, or research purposes

**Redistribution:**
- Cite original sources when publishing analyses
- Do not claim ownership of underlying data
- Follow academic citation practices for research papers

**Recommended Citation:**
```
Fantasy Premier League player performance data (2020-2026).
Data sources: FPL Official API (fantasy.premierleague.com) 
and vaastav/Fantasy-Premier-League GitHub repository.
Expected statistics provided by Opta Sports.
```

---

## Technical Specifications

### System Requirements

**Python Version:** 3.7 or higher

**Required Packages:**
- `requests` >= 2.25.0
- `pandas` >= 1.2.0

**Disk Space:**
- Gameweek data: ~5 MB per gameweek (~200 MB for full season)
- Historical data: ~30 MB per season (~150 MB for 5 seasons)
- Combined dataset: ~250-300 MB total

**Network Requirements:**
- Stable internet connection
- ~100 MB download for initial setup
- ~10 MB per weekly update

### Performance Metrics

**API Request Volume:**
- Bootstrap endpoint: 1 request per pipeline run
- Player history: ~650 requests (one per active player)
- Total API calls per full run: ~651 requests

**Execution Time Estimates:**
- Bootstrap data fetch: 2-3 seconds
- Single gameweek download: 5-8 minutes (all players)
- Historical download: 1-2 minutes (one-time)
- Full pipeline (initial): 15-20 minutes
- Incremental update: 2-5 minutes

**Rate Limiting:**
- Current: 0.1 second delay between requests
- Daily API limit: Unknown (no official limit published)
- Recommended: Maximum 2-3 full runs per day

---

## Contact and Support

### Questions or Issues

For questions about the data pipeline or dataset:

**Pipeline Issues:**
- Check GitHub repository: [Your repo URL if applicable]
- Review error messages in console output
- Verify API endpoint availability: https://fantasy.premierleague.com/api/bootstrap-static/

**Data Quality Issues:**
- Report upstream to FPL API support
- Check vaastav repository for historical data issues: https://github.com/vaastav/Fantasy-Premier-League/issues

**Feature Requests:**
- [Your contact information or GitHub issues page]

### Additional Resources

**FPL API Documentation:**
- Community documentation: https://www.oliverlooney.com/blogs/FPL-APIs-Explained
- API explorer: https://github.com/jeppe-smith/fpl-api

**Fantasy Premier League:**
- Official website: https://fantasy.premierleague.com
- Rules and scoring: https://fantasy.premierleague.com/help

**Expected Statistics (Opta):**
- xG explanation: https://theanalyst.com/na/2021/07/what-are-expected-goals-xg/
- Advanced metrics guide: [Search "Opta xG methodology"]

---

## Changelog

### Version 1.0 (January 2026)
- Initial pipeline implementation
- Current season (2025-26) data collection
- Historical integration (2020-21 through 2024-25)
- Combined dataset generation
- Incremental update support

### Future Enhancements

**Planned Features:**
- Automated weekly scheduling (cron job)
- Data validation reporting
- Team-level aggregation datasets
- Fixture difficulty integration
- Price change tracking

**Under Consideration:**
- Database storage (SQLite/PostgreSQL)
- Dashboard visualization (Streamlit/Dash)
- API wrapper for easier data access
- Docker containerization

---

## Appendix

### Complete Column Reference (Alphabetical)

| Column | Type | Description |
|--------|------|-------------|
| `assists` | int | Assists provided in match |
| `bonus` | int | Bonus points awarded (0-3) |
| `bps` | int | Bonus Points System raw score |
| `clean_sheets` | int | Clean sheet indicator (0 or 1) |
| `creativity` | float | FPL creativity metric |
| `defensive_contribution` | float | Defensive actions score |
| `element` | int | Unique player ID |
| `expected_assists` | float | Expected assists (xA) |
| `expected_goal_involvements` | float | xG + xA combined |
| `expected_goals` | float | Expected goals (xG) |
| `expected_goals_conceded` | float | Expected goals conceded (xGC) |
| `goals_conceded` | int | Goals conceded (DEF/GK) |
| `goals_scored` | int | Goals scored in match |
| `ict_index` | float | ICT index combined metric |
| `influence` | float | FPL influence metric |
| `kickoff_time` | datetime | Match kickoff timestamp (UTC) |
| `minutes` | int | Minutes played (0-120) |
| `name` | str | Player full name |
| `opponent_team` | int | Opponent team ID |
| `position` | str | Position (GK/DEF/MID/FWD) |
| `red_cards` | int | Red cards received |
| `round` | int | Gameweek number (1-38) |
| `saves` | int | Saves made (primarily GK) |
| `season` | str | Season identifier (YYYY-YY) |
| `team` | str | Player's team name |
| `threat` | float | FPL threat metric |
| `total_points` | int | Total FPL points earned |
| `was_home` | bool | Home match indicator |
| `yellow_cards` | int | Yellow cards received |

### Sample Data Preview

```csv
name,element,season,position,team,round,minutes,goals_scored,assists,total_points
Erling Haaland,354,2025-26,FWD,Man City,1,90,2,1,13
Mohamed Salah,283,2025-26,MID,Liverpool,1,90,1,2,12
Bukayo Saka,392,2025-26,MID,Arsenal,1,87,0,1,6
Erling Haaland,354,2024-25,FWD,Man City,1,90,1,0,7
Mohamed Salah,283,2024-25,MID,Liverpool,1,90,2,1,15
```

### Position Codes

| Code | Full Name | FPL Category |
|------|-----------|--------------|
| GK | Goalkeeper | Goalkeeper |
| DEF | Defender | Defender |
| MID | Midfielder | Midfielder |
| FWD | Forward | Forward |

### Team ID Mapping (2025-26 Season)

| Team ID | Team Name | Short Code |
|---------|-----------|------------|
| 1 | Arsenal | ARS |
| 2 | Aston Villa | AVL |
| 3 | Bournemouth | BOU |
| 4 | Brentford | BRE |
| 5 | Brighton | BHA |
| 6 | Chelsea | CHE |
| 7 | Crystal Palace | CRY |
| 8 | Everton | EVE |
| 9 | Fulham | FUL |
| 10 | Ipswich | IPS |
| 11 | Leicester | LEI |
| 12 | Liverpool | LIV |
| 13 | Man City | MCI |
| 14 | Man Utd | MUN |
| 15 | Newcastle | NEW |
| 16 | Nott'm Forest | NFO |
| 17 | Southampton | SOU |
| 18 | Spurs | TOT |
| 19 | West Ham | WHU |
| 20 | Wolves | WOL |

*Note: Team IDs may vary slightly in historical seasons due to promotions/relegations.*

---

**Document Version:** 1.0  
**Last Updated:** January 27, 2026  
**Maintained By:** [Your Name]  
**Pipeline Version:** 1.0
