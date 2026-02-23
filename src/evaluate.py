import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_PATH = "data/test.csv"
PREDICTIONS_PATH = "data/predictions.csv"
METRICS_PATH = "metrics/results.json"

TARGET = 'total_points'


def compute_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    y_true = predictions_df[TARGET]

    metrics = {
        'naive_lag1':        compute_metrics(y_true, test_df['total_points_lag1'].fillna(0), "Naive (Lag1)"),
        'naive_ma5':         compute_metrics(y_true, test_df['total_points_ma5'].fillna(0),  "Naive (MA5)"),
        'ridge_regression':  compute_metrics(y_true, predictions_df['pred_ridge'],            "Ridge Regression"),
        'random_forest':     compute_metrics(y_true, predictions_df['pred_rf'],               "Random Forest"),
        'gradient_boosting': compute_metrics(y_true, predictions_df['pred_gb'],               "XGBoost"),
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")
