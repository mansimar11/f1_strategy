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