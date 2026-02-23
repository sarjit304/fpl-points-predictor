import pandas as pd
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_DIR = "models"
PREDICTIONS_PATH = "data/predictions.csv"

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


def train_ridge(X_train, y_train, sample_weight=None):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_random_forest(X_train, y_train, sample_weight=None):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_xgboost(X_train, y_train, sample_weight=None):
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


if __name__ == "__main__":
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df[TARGET]

    sample_weight = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None

    # Train models
    print("Training Ridge Regression...")
    ridge_model = train_ridge(X_train, y_train, sample_weight)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, sample_weight)

    print("Training XGBoost...")
    gb_model = train_xgboost(X_train, y_train, sample_weight)

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ridge_model, f"{MODEL_DIR}/ridge_model.joblib")
    joblib.dump(rf_model, f"{MODEL_DIR}/rf_model.joblib")
    joblib.dump(gb_model, f"{MODEL_DIR}/gb_model.joblib")
    print(f"Models saved to {MODEL_DIR}/")

    # Save predictions
    predictions_df = test_df[['name', 'element', 'round', 'position', 'team', TARGET]].copy()
    predictions_df['pred_ridge'] = ridge_model.predict(X_test)
    predictions_df['pred_rf'] = rf_model.predict(X_test)
    predictions_df['pred_gb'] = gb_model.predict(X_test)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
