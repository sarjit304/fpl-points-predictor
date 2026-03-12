"""
Model Training for FPL Points Prediction

Trains three models:
- Ridge Regression: Linear baseline with L2 regularization
- Random Forest: Ensemble of decision trees
- XGBoost: Gradient boosting ensemble

Supports two modes:
- Baseline: Uses hardcoded default parameters
- Tuned: Loads optimized parameters from tune.py output

Outputs are saved to separate directories for comparison.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import yaml
import argparse
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# =============================================================================
# Load Configuration from params.yaml
# =============================================================================

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

PARAMS = load_params()

# Extract paths from params.yaml
TRAIN_PATH = PARAMS['paths']['train_data']
TEST_PATH = PARAMS['paths']['test_data']
TUNED_PARAMS_PATH = PARAMS['paths']['tuned_params']

# Separate output directories for baseline vs tuned
BASELINE_MODEL_DIR = PARAMS['paths']['baseline_model_dir']
TUNED_MODEL_DIR = PARAMS['paths']['tuned_model_dir']
BASELINE_PREDICTIONS_PATH = PARAMS['paths']['baseline_predictions']
TUNED_PREDICTIONS_PATH = PARAMS['paths']['tuned_predictions']

# Feature columns from params.yaml
FEATURE_COLUMNS = (
    PARAMS['features']['lag_features'] + 
    PARAMS['features']['moving_average_features']
)

TARGET = PARAMS['features']['target']

# Baseline parameters from params.yaml
BASELINE_PARAMS = {
    'ridge': PARAMS['baseline']['ridge'],
    'random_forest': PARAMS['baseline']['random_forest'],
    'xgboost': PARAMS['baseline']['xgboost'],
    'use_log_transform': PARAMS['baseline']['use_log_transform']
}


def load_tuned_params():
    """Load tuned parameters from JSON file."""
    if os.path.exists(TUNED_PARAMS_PATH):
        with open(TUNED_PARAMS_PATH, 'r') as f:
            return json.load(f)
    return None


def train_ridge(X_train, y_train, sample_weight=None, params=None):
    params = params or BASELINE_PARAMS['ridge']
    model = Ridge(alpha=params.get('alpha', 1.0))
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_random_forest(X_train, y_train, sample_weight=None, params=None):
    params = params or BASELINE_PARAMS['random_forest']
    model = RandomForestRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 10),
        min_samples_split=params.get('min_samples_split', 10),
        min_samples_leaf=params.get('min_samples_leaf', 5),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_xgboost(X_train, y_train, sample_weight=None, params=None):
    params = params or BASELINE_PARAMS['xgboost']
    model = xgb.XGBRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_and_predict(X_train, y_train, X_test, test_df, sample_weight, 
                      params, model_dir, predictions_path, use_log_transform=False):
    """
    Train all models and save predictions.
    
    Args:
        use_log_transform: If True, train on log(1+points) and inverse transform predictions
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Apply log transform if specified
    if use_log_transform:
        print("  Using log-transformed target: log(1 + total_points)")
        y_train_model = np.log1p(y_train.clip(lower=0))
    else:
        y_train_model = y_train
    
    # Train models with progress bar
    models_to_train = [
        ("Ridge Regression", lambda: train_ridge(X_train, y_train_model, sample_weight, params.get('ridge'))),
        ("Random Forest", lambda: train_random_forest(X_train, y_train_model, sample_weight, params.get('random_forest'))),
        ("XGBoost", lambda: train_xgboost(X_train, y_train_model, sample_weight, params.get('xgboost'))),
    ]
    
    trained_models = {}
    pbar = tqdm(models_to_train, desc="  🏋️ Training models", unit="model", leave=True)
    for model_name, train_func in pbar:
        pbar.set_postfix_str(model_name)
        trained_models[model_name] = train_func()
    
    ridge_model = trained_models["Ridge Regression"]
    rf_model = trained_models["Random Forest"]
    gb_model = trained_models["XGBoost"]
    
    # Save models
    print("  💾 Saving models...")
    joblib.dump(ridge_model, f"{model_dir}/ridge_model.joblib")
    joblib.dump(rf_model, f"{model_dir}/rf_model.joblib")
    joblib.dump(gb_model, f"{model_dir}/gb_model.joblib")
    print(f"  Models saved to {model_dir}/")
    
    # Generate predictions
    predictions_df = test_df[['name', 'element', 'round', 'position', 'team', TARGET]].copy()
    
    pred_ridge = ridge_model.predict(X_test)
    pred_rf = rf_model.predict(X_test)
    pred_gb = gb_model.predict(X_test)
    
    # Inverse transform if using log target
    if use_log_transform:
        print("  Inverse transforming predictions: exp(pred) - 1")
        predictions_df['pred_ridge'] = np.expm1(pred_ridge).clip(0)
        predictions_df['pred_rf'] = np.expm1(pred_rf).clip(0)
        predictions_df['pred_gb'] = np.expm1(pred_gb).clip(0)
    else:
        predictions_df['pred_ridge'] = pred_ridge
        predictions_df['pred_rf'] = pred_rf
        predictions_df['pred_gb'] = pred_gb
    
    predictions_df.to_csv(predictions_path, index=False)
    print(f"  Predictions saved to {predictions_path}")
    
    return ridge_model, rf_model, gb_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FPL prediction models")
    parser.add_argument('--mode', type=str, default='both',
                        choices=['baseline', 'tuned', 'both'],
                        help="Which model configuration to train")
    args = parser.parse_args()
    
    print("="*70)
    print("FPL POINTS PREDICTOR - MODEL TRAINING")
    print("="*70)
    
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df[TARGET]

    sample_weight = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None

    # =========================================================================
    # Train BASELINE models
    # =========================================================================
    if args.mode in ['baseline', 'both']:
        print("\n" + "-"*70)
        print("BASELINE MODELS (original parameters)")
        print("-"*70)
        print(f"  Ridge alpha: {BASELINE_PARAMS['ridge']['alpha']}")
        print(f"  RF n_estimators: {BASELINE_PARAMS['random_forest']['n_estimators']}, "
              f"max_depth: {BASELINE_PARAMS['random_forest']['max_depth']}")
        print(f"  XGB learning_rate: {BASELINE_PARAMS['xgboost']['learning_rate']}, "
              f"max_depth: {BASELINE_PARAMS['xgboost']['max_depth']}")
        print(f"  Log transform: {BASELINE_PARAMS['use_log_transform']}")
        
        train_and_predict(
            X_train, y_train, X_test, test_df, sample_weight,
            BASELINE_PARAMS, BASELINE_MODEL_DIR, BASELINE_PREDICTIONS_PATH,
            use_log_transform=BASELINE_PARAMS['use_log_transform']
        )

    # =========================================================================
    # Train TUNED models
    # =========================================================================
    if args.mode in ['tuned', 'both']:
        print("\n" + "-"*70)
        print("TUNED MODELS (optimized parameters)")
        print("-"*70)
        
        tuned_params = load_tuned_params()
        
        if tuned_params is None:
            print("  ⚠️  No tuned parameters found!")
            print(f"  Run 'uv run python src/tune.py' first to generate {TUNED_PARAMS_PATH}")
        else:
            use_log = tuned_params.get('log_transform_experiment', {}).get('use_log_transform', False)
            
            print(f"  Ridge alpha: {tuned_params.get('ridge', {}).get('alpha', 'N/A')}")
            print(f"  RF params: {tuned_params.get('random_forest', {})}")
            print(f"  XGB params: {tuned_params.get('xgboost', {})}")
            print(f"  Log transform: {use_log}")
            
            train_and_predict(
                X_train, y_train, X_test, test_df, sample_weight,
                tuned_params, TUNED_MODEL_DIR, TUNED_PREDICTIONS_PATH,
                use_log_transform=use_log
            )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
