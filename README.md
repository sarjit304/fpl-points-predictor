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
│   ├── predictions_baseline.csv          # DVC-tracked (baseline model predictions)
│   └── predictions_tuned.csv             # DVC-tracked (tuned model predictions)
│
├── models/                               # DVC-tracked
│   ├── tuned_params.json                 # Optimized hyperparameters from tuning
│   ├── baseline/                         # Baseline models
│   │   ├── ridge_model.joblib
│   │   ├── rf_model.joblib
│   │   └── gb_model.joblib
│   └── tuned/                            # Tuned models
│       ├── ridge_model.joblib
│       ├── rf_model.joblib
│       └── gb_model.joblib
│
├── metrics/
│   ├── results.json                      # DVC-tracked metric (legacy)
│   ├── baseline_metrics.json             # Baseline model metrics
│   ├── tuned_metrics.json                # Tuned model metrics
│   └── comparison_metrics.json           # Baseline vs tuned comparison
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
│   ├── tune.py                           # NEW: Hyperparameter tuning
│   ├── train.py                          # UPDATED: Supports baseline/tuned modes
│   └── evaluate.py                       # UPDATED: Compares baseline vs tuned
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
6. `src/train.py` — trains Ridge, Random Forest, and XGBoost models
7. `src/evaluate.py` — computes and saves evaluation metrics
8. `notebooks/evaluation.ipynb` — analysis, visualisations, and conclusions
9. `notebooks/EDA.ipynb` (or `.py`) — exploratory data analysis

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

## Deliverable 4 - Hyperparameter Tuning

### Overview

We implemented hyperparameter tuning to improve model performance, comparing tuned models against our original baseline configurations.

### Experiments Conducted

#### Experiment 1: Decay Rate Tuning
Tests different decay rates for sample weights to find optimal recency weighting.

- **Rationale**: Player form changes over time. Recent games may be more predictive than games from months ago.
- **Formula**: `weight = exp(-decay_rate × gameweeks_ago)`
- **Tested values**: 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30

#### Experiment 2: Log-Transformed Target
Tests whether predicting `log(total_points + 1)` instead of raw points improves performance.

- **Rationale**: FPL points are right-skewed (most players score 2-4 points, some score 15+). Log transform compresses the range, potentially reducing outlier influence.
- **Implementation**: Uses `log1p` (natural log of 1+x) which handles zeros gracefully.

#### Experiment 3: Model Hyperparameter Grid Search
Uses GridSearchCV with TimeSeriesSplit cross-validation to find optimal hyperparameters.

**Ridge Regression:**
- `alpha`: [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

**Random Forest:**
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, 15, 20]
- `min_samples_split`: [5, 10, 20]
- `min_samples_leaf`: [3, 5, 10]

**XGBoost:**
- `n_estimators`: [50, 100, 200]
- `max_depth`: [3, 6, 9]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `subsample`: [0.7, 0.8, 0.9]
- `colsample_bytree`: [0.7, 0.8, 0.9]

### Evaluation Metric: MAE

We use **Mean Absolute Error (MAE)** as our primary metric because:
- **Interpretable**: MAE of 1.0 means predictions are off by 1 FPL point on average
- **Robust to outliers**: Unlike RMSE, doesn't over-penalize big hauls (15+ points)
- **Matches our goal**: We want accurate point predictions for team selection

### Cross-Validation: TimeSeriesSplit

We use **TimeSeriesSplit** (5 folds) because:
- Respects temporal ordering of FPL gameweek data
- Prevents data leakage from future gameweeks
- More realistic evaluation than random splits for time-series prediction

### New Pipeline Stages

#### 1. Tuning Stage
**File:** `src/tune.py`

Runs all three experiments and saves optimized parameters:
- Tests decay rates for sample weights
- Tests log-transformed target prediction
- Performs grid search for model hyperparameters
- Outputs: `models/tuned_params.json`

#### 2. Updated Training Stage
**File:** `src/train.py`

Now supports two modes via `--mode` argument:
- `baseline`: Trains with original hardcoded parameters
- `tuned`: Trains with optimized parameters from `tuned_params.json`
- `both`: Trains both configurations (default)

Outputs are saved to separate directories:
- Baseline models: `models/baseline/`
- Tuned models: `models/tuned/`
- Baseline predictions: `data/predictions_baseline.csv`
- Tuned predictions: `data/predictions_tuned.csv`

#### 3. Updated Evaluation Stage
**File:** `src/evaluate.py`

Evaluates both baseline and tuned models and generates comparison metrics:
- `metrics/baseline_metrics.json`: Metrics for baseline models
- `metrics/tuned_metrics.json`: Metrics for tuned models
- `metrics/comparison_metrics.json`: Side-by-side comparison with improvement percentages

### Running the Pipeline

```bash
# Pull data and run entire pipeline including tuning
dvc pull
dvc repro

# View all metrics
dvc metrics show

# View comparison between baseline and tuned
cat metrics/comparison_metrics.json

# Force re-run everything
dvc repro -f
```

### Key Files

| File | Description |
|------|-------------|
| `src/tune.py` | Hyperparameter tuning script |
| `src/train.py` | Updated training script (baseline + tuned modes) |
| `src/evaluate.py` | Updated evaluation with comparison |
| `models/tuned_params.json` | Optimized hyperparameters |
| `models/baseline/` | Baseline models directory |
| `models/tuned/` | Tuned models directory |
| `data/predictions_baseline.csv` | Predictions from baseline models |
| `data/predictions_tuned.csv` | Predictions from tuned models |
| `metrics/baseline_metrics.json` | Baseline model metrics |
| `metrics/tuned_metrics.json` | Tuned model metrics |
| `metrics/comparison_metrics.json` | Baseline vs tuned comparison |

---

## License

This project uses data from the official Fantasy Premier League API and the vaastav/Fantasy-Premier-League GitHub repository. See [Data.md](Data.md) for attribution and usage terms.
