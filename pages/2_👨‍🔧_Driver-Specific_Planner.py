# pages/2_👨‍🔧_Driver-Specific_Planner.py
import streamlit as st
import pandas as pd
from track_config import TRACK_CHARACTERISTICS
from utils import load_best_model, predict_stint, plot_strategy_visual

st.set_page_config(page_title="Driver Planner", page_icon="👨‍🔧", layout="wide")
model = load_best_model()

# Initialize session state for storing tyre sets
if 'driver_data' not in st.session_state:
    st.session_state.driver_data = {
        "Default Driver": {
            "tyres": pd.DataFrame([
                {"Compound": "SOFT", "Type": "New", "Available": True, "Used Laps": 0},
                {"Compound": "MEDIUM", "Type": "New", "Available": True, "Used Laps": 0},
                {"Compound": "MEDIUM", "Type": "New", "Available": True, "Used Laps": 0},
                {"Compound": "HARD", "Type": "New", "Available": True, "Used Laps": 0},
                {"Compound": "HARD", "Type": "New", "Available": True, "Used Laps": 0},
            ]), "strategy_result": None
        }
    }
if 'active_driver' not in st.session_state:
    st.session_state.active_driver = "Default Driver"

st.title("👨‍🔧 Driver-Specific Planner")
if model is None:
    st.error("Model not loaded. Please train a model (train.py) and ensure the MLflow UI server is running.")
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.active_driver = st.selectbox("Select Driver to Plan For:", options=list(st.session_state.driver_data.keys()))
    with col2:
        new_driver_name = st.text_input("Or Add a New Driver:", "")
        if st.button("Add Driver") and new_driver_name:
            if new_driver_name not in st.session_state.driver_data:
                st.session_state.driver_data[new_driver_name] = {
                    "tyres": pd.DataFrame([{"Compound": "SOFT", "Type": "New", "Available": True, "Used Laps": 0}]),
                    "strategy_result": None
                }
                st.session_state.active_driver = new_driver_name
                st.rerun()

    st.subheader(f"Available Tyre Sets for {st.session_state.active_driver}")
    edited_df = st.data_editor(
        st.session_state.driver_data[st.session_state.active_driver]["tyres"],
        num_rows="dynamic",
        column_config={
            "Compound": st.column_config.SelectboxColumn("Compound", options=["SOFT", "MEDIUM", "HARD"], required=True),
            "Type": st.column_config.SelectboxColumn("Type", options=["New", "Used"], required=True),
            "Available": st.column_config.CheckboxColumn("Available?", default=True),
            "Used Laps": st.column_config.NumberColumn("Used Laps", min_value=0, max_value=50, step=1, required=True),
        },
        key=f"editor_{st.session_state.active_driver}"
    )
    st.session_state.driver_data[st.session_state.active_driver]["tyres"] = edited_df

    if st.button(f"Calculate Best Strategy for {st.session_state.active_driver}"):
        with st.spinner("Simulating..."):
            available_tyres = edited_df[edited_df["Available"] == True].to_dict('records')
            hard_sets = sorted([t for t in available_tyres if t['Compound'] == 'HARD'], key=lambda x: x['Used Laps'])
            medium_sets = sorted([t for t in available_tyres if t['Compound'] == 'MEDIUM'], key=lambda x: x['Used Laps'])
            if not medium_sets or not hard_sets:
                st.error("A Medium->Hard one-stop strategy is not possible with the selected tyres.")
                st.session_state.driver_data[st.session_state.active_driver]["strategy_result"] = None
            else:
                medium_set = medium_sets[0]; hard_set = hard_sets[0]
                track_info = TRACK_CHARACTERISTICS.get("Hungarian Grand Prix", {})
                total_laps = track_info.get('laps', 70); pit_lap = 25; pit_time_loss = 22.0
                base_params = {**track_info, 'total_laps': total_laps, 'driver_factor': 1.0}
                p1 = {**base_params, 'compound': 'MEDIUM', 'start_lap': 1, 'num_laps': pit_lap, 'tyre_start_age': medium_set['Used Laps']}
                p2 = {**base_params, 'compound': 'HARD', 'start_lap': pit_lap + 1, 'num_laps': total_laps - pit_lap, 'tyre_start_age': hard_set['Used Laps']}
                total_time = predict_stint(model, p1)['LapTime'].sum() + predict_stint(model, p2)['LapTime'].sum() + pit_time_loss
                result = {"strategy_viz": [{'compound': 'MEDIUM', 'laps': pit_lap}, {'compound': 'HARD', 'laps': total_laps - pit_lap}],
                          "title": f"M({medium_set['Used Laps']}L) -> H({hard_set['Used Laps']}L) | Total Time: {pd.to_datetime(total_time, unit='s').strftime('%H:%M:%S')}",
                          "total_laps": total_laps}
                st.session_state.driver_data[st.session_state.active_driver]["strategy_result"] = result
    
    st.subheader("Calculated Strategies")
    any_results = any(data["strategy_result"] for data in st.session_state.driver_data.values())
    if not any_results:
        st.info("No strategies calculated yet.")
    else:
        for driver, data in st.session_state.driver_data.items():
            if data["strategy_result"]:
                with st.expander(f"View Strategy for {driver}", expanded=(driver == st.session_state.active_driver)):
                    result = data["strategy_result"]
                    st.pyplot(plot_strategy_visual(result["strategy_viz"], result["title"], result["total_laps"]))