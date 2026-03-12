"""
Hyperparameter Tuning Script for FPL Points Prediction

Experiments:
1. Decay rate tuning (sample weights)
2. Log-transformed target prediction
3. Model hyperparameter grid search

Uses TimeSeriesSplit cross-validation to respect temporal ordering.
Saves optimized parameters to JSON for use in training.

Metric Justification:
- MAE (Mean Absolute Error) is used because:
  - Interpretable: MAE of 1.0 means predictions are off by 1 FPL point on average
  - Robust to outliers: Unlike RMSE, doesn't over-penalize big hauls (15+ pts)
  - Matches our goal: We want accurate point predictions for team selection

Cross-Validation Justification:
- TimeSeriesSplit is used because:
  - Respects temporal ordering of FPL gameweek data
  - Prevents data leakage from future gameweeks
  - More realistic than random splits for time-series prediction
"""

import pandas as pd
import numpy as np
import json
import os
import yaml
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# =============================================================================
# Load Configuration from params.yaml
# =============================================================================

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

PARAMS = load_params()

# Extract paths and settings from params.yaml
TRAIN_PATH = PARAMS['paths']['train_data']
TUNED_PARAMS_PATH = PARAMS['paths']['tuned_params']

# Combine feature columns from params.yaml
FEATURE_COLUMNS = (
    PARAMS['features']['lag_features'] + 
    PARAMS['features']['moving_average_features']
)

TARGET = PARAMS['features']['target']

# Cross-validation settings
CV_SPLITS = PARAMS['cross_validation']['n_splits']


# =============================================================================
# Helper function to convert numpy types to Python native types
# =============================================================================

def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# =============================================================================
# EXPERIMENT 1: Decay Rate Tuning
# =============================================================================

def tune_decay_rate(train_df, cv):
    """
    Test different decay rates for sample weights.
    
    The decay rate controls how much weight recent games get vs older games.
    Formula: weight = exp(-decay_rate × gameweeks_ago)
    
    - decay=0.0: All games weighted equally
    - decay=0.15: Current default - games 5 weeks ago get ~47% weight
    - decay=0.30: Aggressive - games 5 weeks ago get ~22% weight
    
    We test which decay rate minimizes MAE on cross-validation.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Decay Rate Tuning")
    print("="*60)
    print("\nRationale: Player form changes over time. Recent games may be")
    print("more predictive than games from months ago. The decay parameter")
    print("controls this recency weighting.\n")
    
    decay_rates = PARAMS['sample_weight']['decay_rates_to_test']
    results = {}
    
    X = train_df[FEATURE_COLUMNS].fillna(0)
    y = train_df[TARGET]
    
    # We simulate different decay rates by recalculating sample weights
    # Using row index as timeline proxy (later rows = more recent)
    train_df = train_df.reset_index(drop=True)
    max_idx = len(train_df)
    
    for decay in tqdm(decay_rates, desc="Testing decay rates", leave=False):
        # Recalculate weights based on position in dataframe
        if decay == 0.0:
            weights = np.ones(len(train_df))
        else:
            # More recent rows (higher index) get higher weight
            recency = np.arange(len(train_df)) / max_idx  # 0 to 1
            weights = np.exp(-decay * (1 - recency) * 50)  # Scale factor
        
        # Cross-validation with these weights
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            w_train = weights[train_idx]
            
            model = Ridge(alpha=PARAMS['baseline']['ridge']['alpha'])
            model.fit(X_train_cv, y_train_cv, sample_weight=w_train)
            y_pred = model.predict(X_val_cv)
            mae = mean_absolute_error(y_val_cv, y_pred)
            scores.append(mae)
        
        mean_mae = np.mean(scores)
        std_mae = np.std(scores)
        results[decay] = {'mae': mean_mae, 'std': std_mae}
        tqdm.write(f"  decay={decay:.2f}: MAE={mean_mae:.4f} (+/- {std_mae:.4f})")
    
    # Find best decay rate
    best_decay = min(results, key=lambda x: results[x]['mae'])
    print(f"\n  ✓ Best decay rate: {best_decay}")
    
    return {
        'best_decay_rate': best_decay,
        'all_results': {str(k): {'mae': round(v['mae'], 4), 'std': round(v['std'], 4)} 
                        for k, v in results.items()}
    }


# =============================================================================
# EXPERIMENT 2: Log-Transformed Target
# =============================================================================

def tune_log_transform(train_df, cv):
    """
    Test whether predicting log(total_points + 1) improves performance.
    
    Rationale:
    - FPL points are right-skewed (most players score 2-4, some score 15+)
    - Log transform compresses the range, potentially reducing outlier influence
    - We use log1p (natural log of 1+x) because it handles zeros gracefully
    
    We compare MAE on the ORIGINAL scale (after inverse transform).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Log-Transformed Target")
    print("="*60)
    print("\nRationale: FPL points are right-skewed. A player scoring 15 points")
    print("is an outlier that may dominate the loss function. Log transform")
    print("compresses high values, potentially improving predictions for")
    print("typical gameweeks (2-6 points).\n")
    
    X = train_df[FEATURE_COLUMNS].fillna(0)
    y = train_df[TARGET]
    y_log = np.log1p(y.clip(lower=0))  # log(1 + points)
    
    sample_weight = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None
    
    results = {}
    
    for target_type, y_target in tqdm([('original', y), ('log_transformed', y_log)], 
                                       desc="Testing log transform", leave=False):
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv = y_target.iloc[train_idx]
            y_val_actual = y.iloc[val_idx]  # Always evaluate on original scale
            
            w_train = sample_weight[train_idx] if sample_weight is not None else None
            
            model = Ridge(alpha=PARAMS['baseline']['ridge']['alpha'])
            model.fit(X_train_cv, y_train_cv, sample_weight=w_train)
            y_pred = model.predict(X_val_cv)
            
            # If log-transformed, inverse transform predictions
            if target_type == 'log_transformed':
                y_pred = np.expm1(y_pred).clip(0)  # exp(pred) - 1, clip negatives
            
            mae = mean_absolute_error(y_val_actual, y_pred)
            scores.append(mae)
        
        mean_mae = np.mean(scores)
        std_mae = np.std(scores)
        results[target_type] = {'mae': mean_mae, 'std': std_mae}
        tqdm.write(f"  {target_type}: MAE={mean_mae:.4f} (+/- {std_mae:.4f})")
    
    # Determine if log transform helps
    use_log = results['log_transformed']['mae'] < results['original']['mae']
    improvement = results['original']['mae'] - results['log_transformed']['mae']
    
    print(f"\n  {'✓' if use_log else '✗'} Log transform {'IMPROVES' if use_log else 'does NOT improve'} MAE")
    print(f"  Difference: {improvement:.4f} points")
    
    return {
        'use_log_transform': use_log,
        'improvement': round(improvement, 4),
        'original_mae': round(results['original']['mae'], 4),
        'log_mae': round(results['log_transformed']['mae'], 4)
    }


# =============================================================================
# EXPERIMENT 3: Model Hyperparameter Tuning
# =============================================================================

def tune_ridge(X, y, sample_weight, cv):
    """Tune Ridge regression alpha parameter."""
    print("\n--- Tuning Ridge Regression ---")
    print("Alpha controls L2 regularization strength.")
    print("Higher alpha = simpler model, lower alpha = more complex.\n")
    
    # Get search space from params.yaml
    param_grid = {'alpha': PARAMS['tuning']['ridge']['alpha']}
    
    grid_search = GridSearchCV(
        Ridge(),
        param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X, y, sample_weight=sample_weight)
    
    print(f"  Best alpha: {grid_search.best_params_['alpha']}")
    print(f"  Best CV MAE: {-grid_search.best_score_:.4f}")
    
    # Show all results
    results_df = pd.DataFrame(grid_search.cv_results_)
    for _, row in results_df.iterrows():
        alpha = row['param_alpha']
        score = -row['mean_test_score']
        marker = " <-- best" if alpha == grid_search.best_params_['alpha'] else ""
        print(f"    alpha={alpha}: MAE={score:.4f}{marker}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_mae': round(-grid_search.best_score_, 4)
    }


def tune_random_forest(X, y, sample_weight, cv):
    """Tune Random Forest hyperparameters."""
    print("\n--- Tuning Random Forest ---")
    print("Key parameters: n_estimators (# trees), max_depth (tree depth),")
    print("min_samples_split/leaf (controls overfitting).\n")
    
    # Get search space from params.yaml
    rf_params = PARAMS['tuning']['random_forest']
    param_grid = {
        'n_estimators': rf_params['n_estimators'],
        'max_depth': rf_params['max_depth'],
        'min_samples_split': rf_params['min_samples_split'],
        'min_samples_leaf': rf_params['min_samples_leaf']
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y, sample_weight=sample_weight)
    
    print(f"\n  Best params: {grid_search.best_params_}")
    print(f"  Best CV MAE: {-grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_mae': round(-grid_search.best_score_, 4)
    }


def tune_xgboost(X, y, sample_weight, cv):
    """Tune XGBoost hyperparameters."""
    print("\n--- Tuning XGBoost ---")
    print("Key parameters: learning_rate (step size), max_depth,")
    print("subsample/colsample (regularization via subsampling).\n")
    
    # Get search space from params.yaml
    xgb_params = PARAMS['tuning']['xgboost']
    param_grid = {
        'n_estimators': xgb_params['n_estimators'],
        'max_depth': xgb_params['max_depth'],
        'learning_rate': xgb_params['learning_rate'],
        'subsample': xgb_params['subsample'],
        'colsample_bytree': xgb_params['colsample_bytree']
    }
    
    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y, sample_weight=sample_weight)
    
    print(f"\n  Best params: {grid_search.best_params_}")
    print(f"  Best CV MAE: {-grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_mae': round(-grid_search.best_score_, 4)
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    print("="*70)
    print("FPL POINTS PREDICTOR - HYPERPARAMETER TUNING")
    print("="*70)
    
    # Get CV configuration from params.yaml
    cv_splits = PARAMS['cross_validation']['n_splits']
    
    print("\n📊 EVALUATION METRIC: Mean Absolute Error (MAE)")
    print("   Justification:")
    print("   - Interpretable: MAE of 1.0 = predictions off by 1 point on average")
    print("   - Robust to outliers: Doesn't over-penalize big hauls (15+ pts)")
    print("   - Matches our goal: Accurate point predictions for team selection")
    
    print(f"\n📅 CROSS-VALIDATION: TimeSeriesSplit ({cv_splits} folds)")
    print("   Justification:")
    print("   - Respects temporal ordering of FPL gameweek data")
    print("   - Prevents data leakage from future gameweeks")
    print("   - More realistic evaluation than random splits")
    
    # Load data
    print("\n" + "-"*70)
    print("Loading training data...")
    train_path = PARAMS['paths']['train_data']
    train_df = pd.read_csv(train_path)
    print(f"Training samples: {len(train_df)}")
    
    X = train_df[FEATURE_COLUMNS].fillna(0)
    y = train_df[TARGET]
    sample_weight = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None
    
    cv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Store all results
    tuned_params = {
        'tuning_metadata': {
            'cv_method': 'TimeSeriesSplit',
            'cv_folds': cv_splits,
            'scoring_metric': 'neg_mean_absolute_error',
            'n_train_samples': len(train_df),
            'params_yaml': 'params.yaml'
        }
    }
    
    # ==========================================================================
    # Run experiments with progress bar
    # ==========================================================================
    
    experiments = [
        ("1/3 Decay Rate Tuning", lambda: tune_decay_rate(train_df, cv)),
        ("2/3 Log Transform Test", lambda: tune_log_transform(train_df, cv)),
    ]
    
    results_list = []
    pbar = tqdm(experiments, desc="🔧 Running experiments", unit="exp")
    for exp_name, exp_func in pbar:
        pbar.set_postfix_str(exp_name)
        results_list.append(exp_func())
    
    decay_results = results_list[0]
    log_results = results_list[1]
    
    tuned_params['decay_experiment'] = decay_results
    tuned_params['log_transform_experiment'] = log_results
    
    # Experiment 3: Model hyperparameters
    print("\n" + "="*60)
    print("EXPERIMENT 3: Model Hyperparameter Tuning")
    print("="*60)
    
    # If log transform helps, use it for hyperparameter tuning
    if log_results['use_log_transform']:
        print("\n⚠️  Using log-transformed target for hyperparameter tuning")
        y_tune = np.log1p(y.clip(lower=0))
    else:
        y_tune = y
    
    # Tune models with progress bar
    model_tuning = [
        ("Ridge", lambda: tune_ridge(X, y_tune, sample_weight, cv)),
        ("Random Forest", lambda: tune_random_forest(X, y_tune, sample_weight, cv)),
        ("XGBoost", lambda: tune_xgboost(X, y_tune, sample_weight, cv)),
    ]
    
    model_results = {}
    pbar = tqdm(model_tuning, desc="🔧 Tuning models", unit="model")
    for model_name, tune_func in pbar:
        pbar.set_postfix_str(model_name)
        model_results[model_name] = tune_func()
    
    ridge_results = model_results["Ridge"]
    rf_results = model_results["Random Forest"]
    xgb_results = model_results["XGBoost"]
    
    tuned_params['ridge'] = ridge_results['best_params']
    tuned_params['ridge_cv_mae'] = ridge_results['best_cv_mae']
    
    tuned_params['random_forest'] = rf_results['best_params']
    tuned_params['random_forest_cv_mae'] = rf_results['best_cv_mae']
    
    tuned_params['xgboost'] = xgb_results['best_params']
    tuned_params['xgboost_cv_mae'] = xgb_results['best_cv_mae']
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    
    tuned_params_path = PARAMS['paths']['tuned_params']
    os.makedirs(os.path.dirname(tuned_params_path), exist_ok=True)
    
    # Convert all numpy types to native Python types for JSON serialization
    tuned_params_clean = convert_to_native(tuned_params)
    
    with open(tuned_params_path, 'w') as f:
        json.dump(tuned_params_clean, f, indent=2)
    
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {tuned_params_path}")
    
    print("\n📋 SUMMARY OF BEST PARAMETERS:")
    print(f"   Decay rate: {decay_results['best_decay_rate']}")
    print(f"   Use log transform: {log_results['use_log_transform']}")
    print(f"   Ridge: {ridge_results['best_params']}")
    print(f"   Random Forest: {rf_results['best_params']}")
    print(f"   XGBoost: {xgb_results['best_params']}")


if __name__ == "__main__":
    main()
