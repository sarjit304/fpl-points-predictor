import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb

TRAIN_PATH = "data/train.csv"
TUNED_PARAMS_PATH = "models/tuned_params.json"
PARAMS_PATH = "params.json"

FEATURE_COLUMNS = [
    "total_points_lag1",
    "expected_goal_involvements_lag1",
    "minutes_lag1",
    "total_points_ma5",
    "expected_goal_involvements_ma5",
    "ict_index_ma5",
    "bps_ma5",
]

TARGET = "total_points"


def sort_temporal(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["season", "round", "kickoff_time"] if c in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols).reset_index(drop=True)


def load_params_config(path):
    with open(path, "r") as f:
        return json.load(f)


def run_search(model, param_distributions, X, y, sample_weight, cv_folds=3, n_iter=20):
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    if sample_weight is not None:
        search.fit(X, y, sample_weight=sample_weight)
    else:
        search.fit(X, y)
    return search


if __name__ == "__main__":
    print("Loading training data for tuning...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = sort_temporal(train_df)

    params_config = load_params_config(PARAMS_PATH)
    feature_columns = params_config.get("features", FEATURE_COLUMNS)
    tuning_config = params_config.get("tuning", {})

    cv_folds = tuning_config.get("cv_folds", 3)
    n_iter = tuning_config.get("n_iter", 20)
    rf_search_space = tuning_config.get(
        "random_forest_search_space",
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [6, 8, 10, 14, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", 0.6, 0.8, 1.0],
        },
    )
    xgb_search_space = tuning_config.get(
        "xgboost_search_space",
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
    )

    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df[TARGET]
    sample_weight = train_df["sample_weight"] if "sample_weight" in train_df.columns else None

    print("Tuning Random Forest...")
    rf_search = run_search(
        model=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=rf_search_space,
        X=X_train,
        y=y_train,
        sample_weight=sample_weight,
        cv_folds=cv_folds,
        n_iter=n_iter,
    )

    print("Tuning XGBoost...")
    gb_search = run_search(
        model=xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
        param_distributions=xgb_search_space,
        X=X_train,
        y=y_train,
        sample_weight=sample_weight,
        cv_folds=cv_folds,
        n_iter=n_iter,
    )

    tuned_params = {
        "metric": "cv_mean_absolute_error",
        "cv_folds": cv_folds,
        "n_iter": n_iter,
        "random_forest": {
            "best_score_mae": round(-rf_search.best_score_, 6),
            "best_params": rf_search.best_params_,
        },
        "xgboost": {
            "best_score_mae": round(-gb_search.best_score_, 6),
            "best_params": gb_search.best_params_,
        },
    }

    os.makedirs(os.path.dirname(TUNED_PARAMS_PATH), exist_ok=True)
    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(tuned_params, f, indent=2)

    print(f"Tuned parameters saved to {TUNED_PARAMS_PATH}")
