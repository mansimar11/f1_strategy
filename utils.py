import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load Model Artifacts ---
def load_model_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

# --- Predict Lap Times for a Stint ---
def predict_stint_lap_times(stint_info, model, scaler, model_columns):
    compound = stint_info.get('compound', 'UNKNOWN')
    num_laps = int(stint_info.get('num_laps', 0))
    start_lap = int(stint_info.get('start_lap', 1))
    tyre_start_age = int(stint_info.get('tyre_start_age', 0))

    prediction_data = []
    for i in range(num_laps):
        lap_features = {
            'LapNumber': start_lap + i,
            'TyreLife': tyre_start_age + i + 1,
            f'Compound_{compound}': 1
        }
        prediction_data.append(lap_features)

    X_pred = pd.DataFrame(prediction_data)
    for col in model_columns:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[model_columns]
    X_scaled = scaler.transform(X_pred)
    predicted_lap_times = model.predict(X_scaled)

    return pd.DataFrame({'Lap': X_pred['LapNumber'].astype(int), 'LapTime': predicted_lap_times})

# --- Plot Strategy ---
def plot_strategy(strategy, total_race_laps):
    TYRE_VISUALS = {
        "SOFT": {"color": "#FF3333", "text": "black"},
        "MEDIUM": {"color": "#FFF200", "text": "black"},
        "HARD": {"color": "#F0F0F0", "text": "black"},
        "INTERMEDIATE": {"color": "#44D47E", "text": "black"},
        "WET": {"color": "#339EFF", "text": "black"}
    }

    fig, ax = plt.subplots(figsize=(15, 2), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    current_lap = 1
    for i, stint in enumerate(strategy):
        compound = stint.get('compound', 'UNKNOWN').upper()
        num_laps = int(stint.get('laps', 0))
        stint_end_lap = current_lap + num_laps
        visual_info = TYRE_VISUALS.get(compound, {"color": "grey", "text": "white"})

        ax.barh(0, num_laps, left=current_lap, color=visual_info["color"], edgecolor='black', height=0.6)
        ax.text(current_lap + num_laps / 2, 0, compound, color=visual_info["text"],
                ha='center', va='center', weight='bold', fontsize=12)

        if i < len(strategy) - 1:
            pit_lap = stint_end_lap - 1
            ax.axvline(pit_lap + 0.5, color="white", linestyle="--", lw=1.5, alpha=0.8)
            ax.text(pit_lap + 0.5, 0.45, f"Pit\nLap {pit_lap}", color="cyan", fontsize=9, ha='center', va='bottom')

        current_lap = stint_end_lap

    ax.set_xlim(0, total_race_laps + 1)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks(range(0, total_race_laps + 1, 10))
    ax.tick_params(axis='x', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel("Lap Number", color="white")
    plt.tight_layout()

    return fig
