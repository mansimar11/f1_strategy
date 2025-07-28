# pages/3_⏱️_In-Race_Advisor.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from track_config import TRACK_CHARACTERISTICS
from utils import load_best_model, predict_stint

st.set_page_config(page_title="In-Race Advisor", page_icon="⏱️", layout="wide")
model = load_best_model()

st.title("⏱️ In-Race Advisor")
if model is None:
    st.error("Model not loaded. Please train a model (train.py) and ensure the MLflow UI server is running.")
else:
    st.info("Simulate whether pitting or staying out is the faster option over the next 10 laps.")
    track_info = TRACK_CHARACTERISTICS.get("Hungarian Grand Prix", {})
    total_laps = track_info.get('laps', 70)
    
    col1, col2, col3, col4 = st.columns(4)
    current_lap = col1.number_input("Current Lap:", 1, total_laps, 20)
    current_compound = col2.selectbox("Current Tyre:", ('SOFT', 'MEDIUM', 'HARD'))
    tyre_age = col3.number_input("Current Tyre Age:", 1, 60, 15)
    pit_time_loss = col4.number_input("Pit Time Loss (s):", 18.0, 30.0, 22.0)

    base_params = {**track_info, 'total_laps': total_laps, 'driver_factor': 1.0}
    
    p_stay = {**base_params, 'compound': current_compound, 'start_lap': current_lap, 'num_laps': 10, 'tyre_start_age': tyre_age}
    stay_out_stint = predict_stint(model, p_stay)
    stay_out_stint['CumulativeTime'] = stay_out_stint['LapTime'].cumsum()
    
    p_pit = {**base_params, 'compound': 'SOFT', 'start_lap': current_lap, 'num_laps': 10, 'tyre_start_age': 0}
    pit_now_stint = predict_stint(model, p_pit)
    pit_now_stint['CumulativeTime'] = pit_now_stint['LapTime'].cumsum() + pit_time_loss
    
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), stay_out_stint['CumulativeTime'], label=f'Stay Out (Old {current_compound})', marker='o')
    ax.plot(range(1, 11), pit_now_stint['CumulativeTime'], label='Pit Now (New SOFT)', marker='x', linestyle='--')
    ax.set_title("Cumulative Time Comparison: Pit vs. Stay Out")
    ax.set_xlabel("Laps From Now")
    ax.set_ylabel("Total Time Elapsed (s)")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)
    
    crossover = next((i for i, (s, p) in enumerate(zip(stay_out_stint['CumulativeTime'], pit_now_stint['CumulativeTime'])) if p < s), None)
    if crossover is not None:
        st.warning(f"**PIT NOW!** Pitting becomes the faster option in **{crossover + 1} laps.**")
    else:
        st.success("**STAY OUT.** It is currently faster to remain on your current tyres.")