import streamlit as st
import pandas as pd
from track_config import TRACK_CHARACTERISTICS
from utils import (
    load_model_artifacts,
    predict_stint_lap_times,
    plot_strategy,
)

st.set_page_config(page_title="Driver Planner", page_icon="👨‍🔧", layout="wide")

# Load the single available model
model, scaler, model_columns = load_model_artifacts()

available_tracks = list(TRACK_CHARACTERISTICS.keys())
selected_race = st.selectbox("Track (GP):", available_tracks)
track_info = TRACK_CHARACTERISTICS.get(selected_race, {})
total_laps = track_info.get("laps", 58)

with st.expander("Tyre & Simulation Customization"):
    pit_time_loss = st.slider("Pit Lane Time Loss (s)", 18, 30, 22)
    track_temp = st.slider("Track Temperature (°C)", 10, 60, int(track_info.get("track_temp", 30)))
    abrasiveness = st.slider("Track Abrasiveness", 1, 5, int(track_info.get("abrasiveness", 3)))
    driver_factor = st.slider("Driver Factor", 0.98, 1.03, 1.0, 0.001)
    user_params = {"track_temp": track_temp, "abrasiveness": abrasiveness, "driver_factor": driver_factor}

if 'driver_data' not in st.session_state:
    st.session_state.driver_data = {
        "Default Driver": {
            "tyres": pd.DataFrame([
                {"Compound": "SOFT", "Available": True, "Used Laps": 0},
                {"Compound": "MEDIUM", "Available": True, "Used Laps": 0},
                {"Compound": "HARD", "Available": True, "Used Laps": 0}
            ]),
            "strategy_result": None
        }
    }

if 'active_driver' not in st.session_state:
    st.session_state.active_driver = "Default Driver"

st.subheader(f"Tyre Sets for {st.session_state.active_driver}")
edited_df = st.data_editor(
    st.session_state.driver_data[st.session_state.active_driver]["tyres"],
    num_rows="dynamic",
    column_config={
        "Compound": st.column_config.SelectboxColumn("Compound", options=["SOFT", "MEDIUM", "HARD"], required=True),
        "Available": st.column_config.CheckboxColumn("Available?", default=True),
        "Used Laps": st.column_config.NumberColumn("Used Laps", min_value=0, max_value=50, step=1, required=True),
    },
    key=f"editor_{st.session_state.active_driver}"
)
st.session_state.driver_data[st.session_state.active_driver]["tyres"] = edited_df

def valid_stops_seq(comps, total_laps):
    strategies = []
    if len(comps) < 2: return strategies
    for c1 in comps:
        for c2 in comps:
            if c1 == c2: continue
            for pit_lap in range(10, total_laps-10, 3):
                if pit_lap > 0 and (total_laps - pit_lap) > 0:
                    strategies.append([
                        {"compound": c1, "num_laps": pit_lap},
                        {"compound": c2, "num_laps": total_laps-pit_lap}
                    ])
    if len(comps) > 2:
        for c1 in comps:
            for c2 in comps:
                for c3 in comps:
                    if len({c1, c2, c3})<2: continue
                    for pit1 in range(8, total_laps-12, 4):
                        for pit2 in range(pit1+6, total_laps-6, 4):
                            if pit2-pit1 > 4 and total_laps-pit2 > 4:
                                strategies.append([
                                    {"compound": c1, "num_laps": pit1},
                                    {"compound": c2, "num_laps": pit2-pit1},
                                    {"compound": c3, "num_laps": total_laps-pit2}
                                ])
    return strategies

predict_model = lambda stint: predict_stint_lap_times(stint, model, scaler, model_columns)

if st.button("Find Best Strategy"):
    avail_tyres = [t for t in edited_df.to_dict("records") if t["Available"]]
    compounds = list(sorted({t["Compound"] for t in avail_tyres}))
    if len(compounds) < 2:
        st.warning("Add at least two available tyre compounds!")
    else:
        best, min_time = None, float('inf')
        for plan in valid_stops_seq(compounds, total_laps):
            total_time, start_lap = (len(plan)-1)*pit_time_loss, 1
            viz_plan = []
            valid = True
            for stint in plan:
                matched = [ty for ty in avail_tyres if ty["Compound"] == stint["compound"]]
                tyre = min(matched, key=lambda x: x["Used Laps"]) if matched else {"Used Laps": 0}
                stint_params = {
                    "compound": stint["compound"],
                    "start_lap": start_lap,
                    "num_laps": stint["num_laps"],
                    "tyre_start_age": tyre["Used Laps"]
                }
                df = predict_model(stint_params)
                if df.empty or df["LapTime"].sum() < 0:
                    valid = False
                    break
                total_time += df["LapTime"].sum()
                viz_plan.append({"compound": stint["compound"], "laps": stint["num_laps"]})
                start_lap += stint["num_laps"]
            if valid and total_time < min_time:
                best, min_time = {"plan": viz_plan, "time": total_time}, total_time

        if best:
            st.pyplot(plot_strategy(
                best["plan"],
                total_race_laps=total_laps
            ))
            st.write(f"🏆 **ML Best Strategy Time:** {pd.to_datetime(best['time'], unit='s').strftime('%H:%M:%S')}")
        else:
            st.info("No viable strategy found.")
        st.session_state.driver_data[st.session_state.active_driver]["strategy_result"] = best
