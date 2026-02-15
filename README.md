# FPL Points Predictor

A machine learning project to predict Fantasy Premier League (FPL) player points for upcoming gameweeks.

## Team Members

- Andy Pike
- Mubeen Yaqub
- Sarjit Kaswala
- Sumedha Na

## Project Goal

Fantasy Premier League (FPL) is the official fantasy game for the English Premier League. The objective of the game is to predict which soccer players will score the most points weekly based on the game's rules. Our goal is to build a model that automatically forecasts the points each player will earn in FPL for the upcoming gameweek.

### Target Variable & Features

**Target Variable:** FPL points in the upcoming gameweek

FPL assigns players points based on real-life actions in games, such as scoring goals, playing a full 90 minutes, and passing a threshold for defensive actions.

**Features:**
- Historical player performance (FPL points per gameweek)
- Underlying real-life actions (goals scored, assists, clean sheets, etc.)
- Advanced metrics from Opta (xG, xA, xGI) - measures of chance quality
- Match context (home/away, opponent)
- Time-series features (moving averages to capture performance trends)

### Success Metrics

We will evaluate model performance using Mean Absolute Error (MAE) across gameweeks. The project succeeds if we build a complete data integration, model training, and analysis pipeline that can be retrained weekly as new gameweek data becomes available.

## Data Sources

1. **FPL Official API** (`https://fantasy.premierleague.com/api`)
   - Current season player data (gameweek-by-gameweek statistics)
   - Player metadata, team mappings, position information
   - Opta-powered expected statistics (xG, xA, xGI, xGC)

2. **vaastav/Fantasy-Premier-League GitHub Repository** (`https://github.com/vaastav/Fantasy-Premier-League`)
   - Historical season data (2020-21 through 2024-25)
   - Aggregated and validated performance records

See [Data.md](Data.md) for comprehensive data documentation including schema, collection pipeline, and usage examples.

## Project Setup

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/sarjit304/fpl-points-predictor.git
cd fpl-points-predictor

# Install dependencies
uv sync

# Pull DVC-tracked data files
dvc pull
```

### Running the Data Pipeline

```bash
# Fetch current season data and build combined dataset
python fpl_data_acquisition.py
```

This will:
1. Fetch player data from FPL API
2. Download any missing gameweeks (incremental updates)
3. Download historical season data (if not cached)
4. Generate `combined_all_seasons.csv` with all seasons merged

## Project Structure

```
fpl-points-predictor/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── EDA.py
│
├── src/
│   ├── data_acquisition.py
│   ├── data_cleaning.py
│   ├── train_test_split.py
│   ├── train.py
│
├── dvc.yaml
├── dvc.lock
├── pyproject.toml
├── uv.lock
├── README.md

```
---

## Deliverable 2 - Exploration

We made the following improvements to modularize the workflow and integrate it cleanly with DVC:

### 1. Project Restructuring
- Moved all Python scripts into a dedicated `src/` directory.
- This improves clarity and readability.


### 2. Data Combination Stage  
**File:** `src/combine_data.py`

- Combines multiple raw datasets into a single unified dataset.
- Originally part of data acquisition, but separated into its own script.
- Added as a dedicated stage in the DVC pipeline.


### 3. Data Cleaning Stage  
**File:** `src/clean_data.py`

- Takes the combined dataset as input.
- Performs preprocessing and cleaning for downstream tasks.
- Outputs a cleaned dataset used for splitting and analysis.


### 4. Train–Test Split Stage  
**File:** `src/train_test_split.py`

- Splits cleaned data into:
  - 70% Training
  - 30% Testing
- Outputs separate CSV files.
- Registered as a DVC stage for reproducibility.


### 5. Exploratory Data Analysis (EDA)  
**File:** `eda.ipynb & eda.py`

- Used for understanding distributions, trends, and feature behavior.

To improve Git tracking, we used **Ruff** to convert the notebook into a `.py` file, making diffs cleaner and easier to review.

---

## Recommended File Review Order

To understand the workflow:

1. `dvc.yaml`
2. `src/combine_data.py`
3. `src/clean_data.py`
4. `src/train_test_split.py`
5. `notebooks/eda.ipynb` (or converted `.py` version)

This update ensures a modular, reproducible, and DVC-managed ML data pipeline.


## License

This project uses data from the official Fantasy Premier League API and the vaastav/Fantasy-Premier-League GitHub repository. See [Data.md](Data.md) for attribution and usage terms.
