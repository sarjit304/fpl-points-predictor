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
- Time-series features (lag and moving averages to capture performance trends)
- Planned: match context (fixture difficulty, home/away) — not yet included

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
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (for DVC remote access)
- **Mac only:** `brew install libomp` (required for XGBoost)

### Installation

```bash
# Clone the repository
git clone https://github.com/sarjit304/fpl-points-predictor.git
cd fpl-points-predictor

# Install dependencies
uv sync
```

### Google Cloud Authentication

DVC-tracked data and models are stored in a GCS bucket (`gs://fpl-points-predictor`). You need to authenticate before running `dvc pull`:

```bash
gcloud auth application-default login
```

You will also need to be granted access to the bucket — contact a team member if you haven't been added yet.

### Fetching the Latest Data

This step pulls the most recent gameweek from the FPL API and updates the raw data. It feeds into the DVC pipeline but runs outside of it:

```bash
python fpl_data_acquisition.py
```

This will:
1. Fetch player data from the FPL API
2. Download any missing gameweeks (incremental updates)
3. Download historical season data (if not cached)
4. Update `DSCI_611_project/all_gameweeks.csv`

### Running the Full Pipeline

Pull the raw input data from GCS, then let DVC reproduce all pipeline outputs (cleaned data, features, models, metrics) from scratch:

```bash
# Pull raw input data from GCS
dvc pull

# Reproduce all pipeline stages
dvc repro

# View metrics
dvc metrics show
```

Only the raw input data (`DSCI_611_project/all_gameweeks.csv` and historical seasons) is stored in GCS. All other files — cleaned data, features, train/test splits, models, and metrics — are generated locally by `dvc repro`. This ensures the pipeline is fully reproducible from source.

### Publishing Changes

After making changes, push source code to GitHub and raw data to GCS:

```bash
# Push source code, dvc.yaml, dvc.lock, and other tracked files
git push

# Push only the raw input data to GCS (pipeline outputs are reproduced, not stored)
dvc push DSCI_611_project/all_gameweeks.csv.dvc
```

## Project Structure

```
fpl-points-predictor/
│
├── data/
│   ├── raw/
│   │   └── combined_all_seasons.csv      # DVC-tracked
│   ├── processed/
│   │   ├── cleaned_all_seasons.csv       # DVC-tracked
│   │   └── features_all_seasons.csv      # DVC-tracked
│   ├── train.csv                         # DVC-tracked
│   ├── test.csv                          # DVC-tracked
│   ├── predictions.csv                   # Baseline predictions (DVC-tracked)
│   └── predictions_tuned.csv             # Tuned predictions (DVC-tracked)
│
├── models/                              # DVC-tracked
│   ├── ridge_model.joblib
│   ├── rf_model.joblib
│   ├── gb_model.joblib
│   ├── ridge_model_tuned.joblib
│   ├── rf_model_tuned.joblib
│   ├── gb_model_tuned.joblib
│   └── tuned_params.json
│
├── metrics/
│   └── results.json                     # DVC-tracked metric
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── EDA.py
│   └── evaluation.ipynb
│
├── src/
│   ├── combine_data.py
│   ├── Data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_test_split.py
│   ├── train.py
│   └── evaluate.py
│
├── dvc.yaml
├── dvc.lock
├── pyproject.toml
├── uv.lock
└── README.md
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

## Recommended File Review Order

To understand the full workflow, read files in pipeline order:

1. `dvc.yaml` — pipeline definition (start here to see the big picture)
2. `src/combine_data.py` — merges raw data sources
3. `src/Data_cleaning.py` — cleans and standardises the combined dataset
4. `src/feature_engineering.py` — creates lag and moving average features
5. `src/train_test_split.py` — temporal train/test split with recency weights
6. `src/tune.py` — performs CV-based hyperparameter tuning and saves best params
7. `src/train.py` — trains baseline and tuned Ridge, Random Forest, and XGBoost models
8. `src/evaluate.py` — computes and saves baseline vs tuned evaluation metrics
9. `notebooks/evaluation.ipynb` — analysis, visualisations, and conclusions
10. `notebooks/EDA.ipynb` (or `.py`) — exploratory data analysis

This update ensures a modular, reproducible, and DVC-managed ML data pipeline.

---

## Deliverable 3 - Model Training & Evaluation

### New Pipeline Stages

We extended the DVC pipeline with model training and evaluation:

#### 1. Feature Engineering Stage
**File:** `src/feature_engineering.py`

Creates predictive features from cleaned data:
- **Lag features**: Previous gameweek values (`total_points_lag1`, `minutes_lag1`, etc.)
- **Rolling averages**: 5-game moving averages (`total_points_ma5`, `ict_index_ma5`, etc.)
- All features use `shift()` to prevent data leakage

#### 2. Train/Test Split Stage
**File:** `src/train_test_split.py`

Temporal split to simulate real-world prediction:
- **Train**: Historical seasons (2020-25) + GW1-5 of 2025-26
- **Test**: GW6+ of 2025-26 season
- Recency weighting via exponential decay (recent games weighted higher)

#### 3. Model Training Stage
**File:** `src/train.py`

Trains three models and evaluates on test set:
- **Ridge Regression**: Linear baseline with regularization
- **Random Forest**: Ensemble of decision trees (best performer)
- **Gradient Boosting**: Sequential ensemble method

Outputs:
- Trained models serialised with `joblib` and saved to `models/` (`ridge_model.joblib`, `rf_model.joblib`, `gb_model.joblib`) — DVC-tracked, not stored in git
- Predictions saved to `data/predictions.csv`
- Metrics saved to `metrics/results.json` (tracked by git via `cache: false`)

### Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Random Forest | 1.03 | 1.98 | 0.31 |
| Gradient Boosting | 1.05 | 2.02 | 0.28 |
| Ridge Regression | 1.08 | 1.98 | 0.31 |

### Evaluation Notebook

**File:** `notebooks/evaluation.ipynb`

Contains:
- Evaluation setup explanation (temporal split rationale)
- Metrics explanation (why MAE, RMSE, R²)
- Visualizations (model comparison, residuals, feature importance)
- Conclusions and observations

### Running the Full Pipeline

```bash
# Pull data and run entire pipeline
dvc pull
dvc repro

# View metrics
dvc metrics show

# Force re-run everything
dvc repro -f
```

### Key Files

| File | Description |
|------|-------------|
| `src/feature_engineering.py` | Creates lag/rolling features |
| `src/train_test_split.py` | Temporal train/test split |
| `src/train.py` | Model training and evaluation |
| `src/evaluate.py` | Computes and saves metrics |
| `notebooks/evaluation.ipynb` | Evaluation notebook with visualizations |
| `models/ridge_model.joblib` | Serialised Ridge Regression model (DVC-tracked) |
| `models/rf_model.joblib` | Serialised Random Forest model (DVC-tracked) |
| `models/gb_model.joblib` | Serialised Gradient Boosting model (DVC-tracked) |
| `metrics/results.json` | MAE, RMSE, R² for all models (tracked by git) |
| `data/predictions.csv` | Player predictions ranked by expected points |

---
## Deliverable 4 - Tuning & Experiments

### What was added

- Added `src/tune.py` to run hyperparameter tuning with `RandomizedSearchCV` + `TimeSeriesSplit` and optimize for CV MAE.
- Added DVC stage `tune` that writes tuned parameters to `models/tuned_params.json`.
- Updated `src/train.py` with `--mode baseline|tuned` and loading from `models/tuned_params.json`.
- Added DVC stage `train-tuned` to save tuned artifacts separately:
  - `models/ridge_model_tuned.joblib`
  - `models/rf_model_tuned.joblib`
  - `models/gb_model_tuned.joblib`
  - `data/predictions_tuned.csv`
- Updated `src/evaluate.py` to report both baseline and tuned metrics in one file (`metrics/results.json`) and include deltas (`tuned_minus_baseline`).
- Updated `notebooks/evaluation.ipynb` to load and compare baseline vs tuned outputs.

### Tuning results summary (test set)

| Model | Baseline MAE | Tuned MAE | MAE Delta (Tuned - Baseline) |
|------|------|------|------|
| Ridge Regression | 1.0772 | 1.0772 | 0.0000 |
| Random Forest | 1.0326 | 1.0181 | -0.0145 |
| XGBoost | 1.0566 | 1.0459 | -0.0107 |

Interpretation:
- Tuning improved MAE for both Random Forest and XGBoost.
- Tuned XGBoost also improved RMSE (2.0037 -> 1.9858) and R² (0.2894 -> 0.3020).
- Ridge was kept as the linear baseline and therefore unchanged.

### Reproducing Deliverable 4

```bash
dvc pull
dvc repro -f
dvc metrics show
```

---

## License

This project uses data from the official Fantasy Premier League API and the vaastav/Fantasy-Premier-League GitHub repository. See [Data.md](Data.md) for attribution and usage terms.
