import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_DIR = "models"
TUNED_PARAMS_PATH = "models/tuned_params.json"
PARAMS_PATH = "params.json"

FEATURE_COLUMNS = [
    'total_points_lag1',
    'expected_goal_involvements_lag1',
    'minutes_lag1',
    'total_points_ma5',
    'expected_goal_involvements_ma5',
    'ict_index_ma5',
    'bps_ma5',
]

TARGET = 'total_points'


def train_ridge(X_train, y_train, sample_weight=None, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_random_forest(X_train, y_train, sample_weight=None, params=None):
    params = params or {}
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 10),
        min_samples_split=params.get("min_samples_split", 10),
        min_samples_leaf=params.get("min_samples_leaf", 5),
        max_features=params.get("max_features", "sqrt"),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_xgboost(X_train, y_train, sample_weight=None, params=None):
    params = params or {}
    model = xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def load_tuned_params(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return raw.get("random_forest", {}).get("best_params", {}), raw.get("xgboost", {}).get("best_params", {})


def load_params_config(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "tuned"], default="baseline")
    parser.add_argument("--tuned-params-path", default=TUNED_PARAMS_PATH)
    parser.add_argument("--params-path", default=PARAMS_PATH)
    args = parser.parse_args()

    is_tuned = args.mode == "tuned"
    predictions_path = "data/predictions_tuned.csv" if is_tuned else "data/predictions.csv"
    ridge_model_path = f"{MODEL_DIR}/ridge_model_tuned.joblib" if is_tuned else f"{MODEL_DIR}/ridge_model.joblib"
    rf_model_path = f"{MODEL_DIR}/rf_model_tuned.joblib" if is_tuned else f"{MODEL_DIR}/rf_model.joblib"
    gb_model_path = f"{MODEL_DIR}/gb_model_tuned.joblib" if is_tuned else f"{MODEL_DIR}/gb_model.joblib"

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    params_config = load_params_config(args.params_path)
    feature_columns = params_config.get("features", FEATURE_COLUMNS)
    baseline_params = params_config.get("baseline", {})

    X_train = train_df[feature_columns].fillna(0)
    X_test = test_df[feature_columns].fillna(0)
    y_train = train_df[TARGET]

    sample_weight = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None

    ridge_alpha = baseline_params.get("ridge", {}).get("alpha", 1.0)
    rf_params = baseline_params.get("random_forest", {})
    gb_params = baseline_params.get("xgboost", {})
    if is_tuned:
        print(f"Loading tuned params from {args.tuned_params_path}...")
        rf_params, gb_params = load_tuned_params(args.tuned_params_path)

    # Train models
    print("Training Ridge Regression...")
    ridge_model = train_ridge(X_train, y_train, sample_weight, alpha=ridge_alpha)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, sample_weight, params=rf_params)

    print("Training XGBoost...")
    gb_model = train_xgboost(X_train, y_train, sample_weight, params=gb_params)

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ridge_model, ridge_model_path)
    joblib.dump(rf_model, rf_model_path)
    joblib.dump(gb_model, gb_model_path)
    print(f"Models saved to {MODEL_DIR}/")

    # Save predictions
    predictions_df = test_df[['name', 'element', 'round', 'position', 'team', TARGET]].copy()
    predictions_df['pred_ridge'] = ridge_model.predict(X_test)
    predictions_df['pred_rf'] = rf_model.predict(X_test)
    predictions_df['pred_gb'] = gb_model.predict(X_test)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
