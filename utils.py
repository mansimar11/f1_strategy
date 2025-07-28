# utils.py
import streamlit as st
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from track_config import TRACK_CHARACTERISTICS
from PIL import Image
from io import BytesIO
from datetime import datetime
import requests

# --- Data Loading ---
@st.cache_resource
def load_best_model():
    try:
        # Looks for the reliable model experiment
        runs = mlflow.search_runs(
            experiment_names=["F1 Reliable XGBoost Model"], 
            filter_string="tags.best_overall_model = 'true'"
        )
        if runs.empty: return None
        return mlflow.sklearn.load_model(f"runs:/{runs.iloc[0].run_id}/model")
    except Exception as e:
        st.error(f"Error loading model from MLflow: {e}")
        return None

# --- Image Loading ---
@st.cache_data
def load_icon(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGBA")
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {url}: {e}")
        return None

TYRE_URLS = {"SOFT": "https://i.imgur.com/H025L6E.png", "MEDIUM": "https://i.imgur.com/4lJd9T0.png", "HARD": "https://i.imgur.com/T0b4x2b.png"}
TYRE_ICONS = {compound: load_icon(url) for compound, url in TYRE_URLS.items()}

# --- Prediction Function ---
def predict_stint(model, stint_params):
    if model is None: return pd.DataFrame({'LapTime': []})
    num_laps = int(stint_params['num_laps'])
    start_lap = int(stint_params['start_lap'])
    tyre_start_age = int(stint_params['tyre_start_age'])
    total_laps = int(stint_params['total_laps'])
    if num_laps <= 0: return pd.DataFrame({'LapTime': []})
    
    race_laps = range(start_lap, start_lap + num_laps)
    tyre_laps = range(tyre_start_age + 1, tyre_start_age + num_laps + 1)
    
    df = pd.DataFrame()
    df['TyreLife'] = tyre_laps; df['Compound'] = stint_params['compound']
    df['track_temp'] = stint_params['track_temp']; df['abrasiveness'] = stint_params['abrasiveness']
    df['downforce'] = stint_params['downforce']; df['grip_type'] = stint_params['grip_type']
    df['fuel_correction'] = (total_laps - pd.Series(race_laps)) * 0.03
    df['track_evolution'] = (pd.Series(race_laps) - 1) * -0.015
    df['tyre_load'] = df['TyreLife'] * df['downforce']
    df['TyreLife'] *= stint_params['driver_factor']
    df['tyre_load'] *= stint_params['driver_factor']
    
    predicted_times = model.predict(df)
    return pd.DataFrame({'LapTime': predicted_times})

# --- Visualization Function ---
def plot_strategy_visual(strategy, title, total_laps):
    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "whitesmoke"}
    text_colors = {"SOFT": "white", "MEDIUM": "black", "HARD": "black"}
    
    start_lap = 0
    for i, stint in enumerate(strategy):
        compound, laps = stint['compound'], int(stint['laps'])
        ax.barh(0, laps, left=start_lap, color=colors[compound], edgecolor=colors[compound], height=0.5)
        if i > 0: ax.text(start_lap, 0.55, f'Lap {start_lap}', color='white', ha='center', va='bottom', fontsize=10)
        ax.text(start_lap + laps / 2, -0.55, compound, color=text_colors[compound], ha='center', va='top', weight='bold', fontsize=11)

        icon = TYRE_ICONS.get(compound)
        if icon:
            imagebox = OffsetImage(icon, zoom=0.4)
            ab = AnnotationBbox(imagebox, (start_lap + laps / 2, 0), frameon=False, zorder=5)
            ax.add_artist(ab)
        start_lap += laps

    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[c], label=c) for c in ["SOFT", "MEDIUM", "HARD"]]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.5), ncol=3, frameon=False, labelcolor='white', handletextpad=0.5, columnspacing=1)

    ax.set_xlim(0, total_laps); ax.set_ylim(-1.2, 1.2); ax.set_yticks([])
    ax.set_xlabel("Race Lap", color='white'); ax.set_title(title, color='white', fontsize=14, weight='bold')
    ax.tick_params(axis='x', colors='white')
    for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_visible(False)
    ax.tick_params(axis='x', length=0)
    fig.tight_layout(pad=1.5)
    return fig