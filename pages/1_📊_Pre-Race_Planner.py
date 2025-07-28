# pages/1_📊_Pre-Race_Planner.py
import streamlit as st
import pandas as pd
from datetime import datetime
from track_config import TRACK_CHARACTERISTICS
from utils import load_best_model, predict_stint, plot_strategy_visual

st.set_page_config(page_title="Pre-Race Planner", page_icon="📊", layout="wide")
model = load_best_model()

st.title("📊 Pre-Race Planner")
if model is None:
    st.error("Model not loaded. Please run train.py and ensure the MLflow UI server is running.")
else:
    today = pd.to_datetime(datetime.now().date())
    future_races_list = [r for r, d in sorted(TRACK_CHARACTERISTICS.items(), key=lambda i: i[1]['date']) if pd.to_datetime(d['date']) >= today]
    
    selected_race = st.selectbox("Select an Upcoming Grand Prix:", future_races_list)
    track_info = TRACK_CHARACTERISTICS.get(selected_race, {})
    total_laps = track_info.get('laps', 58)
    
    st.info(f"**{selected_race}:** {total_laps} Laps | **Downforce:** {track_info.get('downforce', 'N/A')}/5 | **Abrasiveness:** {track_info.get('abrasiveness', 'N/A')}/5")
    
    with st.expander("Adjust Race Conditions"):
        col1, col2, col3, col4 = st.columns(4)
        weather = col1.selectbox("Expected Weather:", ["Dry", "Wet (Fail-safe)"])
        track_temp = col2.slider("Track Temp (°C)", 10, 50, track_info.get('track_temp', 30))
        pit_time_loss = col3.slider("Pit Lane Time Loss (s)", 18.0, 30.0, 22.0)

        driver_categories = {
            "Elite Tyre Management": 0.95,
            "Average Tyre Management": 1.00,
            "Aggressive / Rookie": 1.05
        }
        selected_driver_cat = col4.selectbox("Driver Tyre Management Style", options=list(driver_categories.keys()))
        driver_factor = driver_categories[selected_driver_cat]

    if weather == "Wet (Fail-safe)":
        st.warning("🚨 **Wet Weather Fail-Safe:** Tyre degradation models are invalid in wet conditions. The optimal strategy will depend on Intermediate/Wet tyres, which are not part of this simulation.")
    else:
        base_params = {**track_info, 'total_laps': total_laps, 'track_temp': track_temp, 'tyre_start_age': 0, 'driver_factor': driver_factor}
        
        # (Simulation logic is unchanged)
        one_stop_pit = 28; strat_1 = [{'compound': 'MEDIUM', 'laps': one_stop_pit}, {'compound': 'HARD', 'laps': total_laps - one_stop_pit}]
        two_stop_pit1, two_stop_pit2 = 18, 38; strat_2 = [{'compound': 'SOFT', 'laps': two_stop_pit1}, {'compound': 'MEDIUM', 'laps': two_stop_pit2 - two_stop_pit1}, {'compound': 'SOFT', 'laps': total_laps - two_stop_pit2}]
        time1 = sum(predict_stint(model, {**base_params, 'compound': s['compound'], 'start_lap': sum(strat_1[j]['laps'] for j in range(i)) + 1, 'num_laps': s['laps']})['LapTime'].sum() for i, s in enumerate(strat_1)) + pit_time_loss
        time2 = sum(predict_stint(model, {**base_params, 'compound': s['compound'], 'start_lap': sum(strat_2[j]['laps'] for j in range(i)) + 1, 'num_laps': s['laps']})['LapTime'].sum() for i, s in enumerate(strat_2)) + (2 * pit_time_loss)

        st.subheader("Strategy Comparison")
        if time1 < time2:
            st.pyplot(plot_strategy_visual(strat_1, f"🏆 Quickest: One-Stop ({pd.to_datetime(time1, unit='s').strftime('%H:%M:%S')})", total_laps))
            st.pyplot(plot_strategy_visual(strat_2, f"Slower: Two-Stop ({pd.to_datetime(time2, unit='s').strftime('%H:%M:%S')})", total_laps))
        else:
            st.pyplot(plot_strategy_visual(strat_2, f"🏆 Quickest: Two-Stop ({pd.to_datetime(time2, unit='s').strftime('%H:%M:%S')})", total_laps))
            st.pyplot(plot_strategy_visual(strat_1, f"Slower: One-Stop ({pd.to_datetime(time1, unit='s').strftime('%H:%M:%S')})", total_laps))