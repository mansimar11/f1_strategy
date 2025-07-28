# train.py
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import mlflow
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from config import DATABASE_URL
from track_config import TRACK_CHARACTERISTICS

# --- QUICK TEST CONFIGURATIONS ---
N_TRIALS = 4
TRACK = "monza"
TARGET_COLUMN = "lap_time"

# --- Load Data ---
engine = create_engine(DATABASE_URL)
df = pd.read_sql_table(TRACK, con=engine)

# --- Preprocessing ---
X = df.drop(columns=["lap_time", "driver", "group_id"])
y = df[TARGET_COLUMN]
groups = df["group_id"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Objective for Optuna ---
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 300),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "loss_function": "RMSE",
        "verbose": 0,
    }

    model = CatBoostRegressor(**params)
    gkf = GroupKFold(n_splits=3)

    scores = []
    for train_idx, valid_idx in gkf.split(X_scaled, y, groups):
        X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        scores.append(rmse)

    return np.mean(scores)

# --- MLflow Tracking ---
mlflow.set_experiment("f1_strategy_sim")

with mlflow.start_run():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    mlflow.log_params(best_params)

    final_model = CatBoostRegressor(**best_params, loss_function="RMSE", verbose=0)
    final_model.fit(X_scaled, y)

    # Save model locally
    joblib.dump(final_model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print(f"Best RMSE: {study.best_value:.4f}")
