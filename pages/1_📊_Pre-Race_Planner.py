import streamlit as st
import pandas as pd
from track_config import TRACK_CHARACTERISTICS
from utils import load_model_artifacts, predict_stint_lap_times, plot_strategy

st.set_page_config(page_title="Pre-Race Planner", layout="wide")
st.title("📊 Pre-Race Planner (ML-Driven)")

# Load model
model, scaler, model_columns = load_model_artifacts()

st.info("Using best available model: XGBoost")

available_tracks = list(TRACK_CHARACTERISTICS.keys())
selected_race = st.selectbox("Select a Grand Prix:", available_tracks)
track_info = TRACK_CHARACTERISTICS.get(selected_race, {})
total_laps = track_info.get("laps", 58)

with st.expander("Adjust Simulation Parameters"):
    pit_time_loss = st.slider("Pit Lane Time Loss (s)", 15.0, 35.0, 22.0, 0.5)
    track_temp = st.slider("Track Temperature (°C)", 10, 60, int(track_info.get("track_temp", 30)))
    abrasiveness = st.slider("Track Abrasiveness (1-5)", 1, 5, int(track_info.get("abrasiveness", 3)))
    driver_factor = st.slider("Driver Factor (Speed Multiplier)", 0.95, 1.05, 1.0, 0.001)

predict_model = lambda stint: predict_stint_lap_times(stint, model, scaler, model_columns)

if st.button("Run Strategy Simulations"):
    pit_m_h = round(total_laps * 0.45)
    pit_s_h = round(total_laps * 0.33)
    pit1_smm = round(total_laps * 0.28)
    pit2_smm = round(total_laps * 0.62)

    strategies = {
        "1-Stop (Medium -> Hard)": [{'compound': 'MEDIUM', 'laps': pit_m_h}, {'compound': 'HARD', 'laps': total_laps - pit_m_h}],
        "1-Stop (Soft -> Hard)": [{'compound': 'SOFT', 'laps': pit_s_h}, {'compound': 'HARD', 'laps': total_laps - pit_s_h}],
        "2-Stop (Soft -> Medium -> Medium)": [
            {'compound': 'SOFT', 'laps': pit1_smm},
            {'compound': 'MEDIUM', 'laps': pit2_smm - pit1_smm},
            {'compound': 'MEDIUM', 'laps': total_laps - pit2_smm}
        ]
    }

    results = []
    for name, strat in strategies.items():
        num_stops = len(strat) - 1
        total_time = num_stops * pit_time_loss
        start_lap = 1
        plan = []

        for stint in strat:
            stint_params = {
                'compound': stint['compound'],
                'start_lap': start_lap,
                'num_laps': stint['laps'],
                'tyre_start_age': 0
            }
            pred_df = predict_model(stint_params)
            if not pred_df.empty:
                total_time += pred_df['LapTime'].sum()
            plan.append({'compound': stint['compound'], 'laps': stint['laps']})
            start_lap += stint['laps']

        results.append({'name': name, 'plan': plan, 'time': total_time})

    sorted_res = sorted(results, key=lambda x: x['time'])
    for i, res in enumerate(sorted_res):
        prefix = "🏆 Quickest:" if i == 0 else f"{i+1}."
        title = f"{prefix} {res['name']} ({pd.to_datetime(res['time'], unit='s').strftime('%H:%M:%S')})"
        st.pyplot(plot_strategy(res['plan'], total_laps))
        st.write(f"**Strategy Plan:** {', '.join([f'{stint['compound']} for {stint['laps']} laps' for stint in res['plan']])}")
