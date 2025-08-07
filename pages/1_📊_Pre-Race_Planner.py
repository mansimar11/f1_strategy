<<<<<<< HEAD
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
=======
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
>>>>>>> 6654cc450a78f302f5fe365ff121aa506498f2a0
