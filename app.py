<<<<<<< HEAD
# app.py
import streamlit as st
from PIL import Image
from pathlib import Path
import os

st.set_page_config(
    page_title="F1 Strategy Simulator | Home",
    page_icon="🏎️",
    layout="wide"
)

st.title("F1 Scientific Strategy Simulator 🏎️")
st.markdown("### _The Definitive Tool for Motorsport Aficionados_")

if not os.path.exists("model.pkl"):
    st.warning("⚠️ Model not found. The app may not function. Please run the training pipeline.")

try:
    image_path = Path(__file__).parent / "image.png"
    image = Image.open(image_path)
    st.image(image, caption="An iconic moment: Alonso, Vettel, and Hamilton perform donuts after the 2018 Abu Dhabi Grand Prix.", use_container_width=True)
except FileNotFoundError:
    st.error("❌ Error: 'image.png' not found.")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.header("About This Project")
        st.write("This application uses an XGBoost model, trained on real historical race data via the FastF1 API, to provide high-fidelity F1 strategy simulations. Select a tool from the sidebar to begin planning and predicting race outcomes.")
with col2:
    with st.container(border=True):
        st.header("Scientific Principles Modelled")
        st.write("- 🏎️ **Tyre Degradation**\n- ⛽ **Fuel Load** (Implicit)\n- ✨ **Track Evolution** (Implicit)\n- 🧬 **Circuit DNA**\n- 🧑‍🚀 **Driver Factor**")
st.info("*If you no longer go for a gap that exists, you are no longer a racing driver.* - Ayrton Senna", icon="🏁")
=======
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
>>>>>>> 6654cc450a78f302f5fe365ff121aa506498f2a0
