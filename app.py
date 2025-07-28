# app.py
import streamlit as st
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="F1 Strategy Simulator | Home",
    page_icon="🏎️",
    layout="wide"
)

# --- Header ---
st.title("F1 Scientific Strategy Simulator 🏎️")
st.markdown("### _The Definitive Tool for Motorsport Aficionados_")

# --- BANNER IMAGE WITH WIDER CROPPING ---
script_dir = Path(__file__).parent
image_path = script_dir / "image.png"

try:
    image = Image.open(image_path)
    
    width, height = image.size
    aspect_ratio = width / height
    
    # Desired aspect ratio (21:9 for a very wide banner)
    desired_aspect_ratio = 21 / 9
    
    if aspect_ratio > desired_aspect_ratio:
        new_width = int(desired_aspect_ratio * height)
        offset = (width - new_width) / 2
        cropped_image = image.crop((offset, 0, width - offset, height))
    else:
        new_height = int(width / desired_aspect_ratio)
        offset = (height - new_height) / 2
        cropped_image = image.crop((0, offset, width, height - offset))

    # --- THIS IS THE CORRECTED LINE ---
    st.image(
        cropped_image, 
        caption="An iconic moment: Alonso, Vettel, and Hamilton perform donuts after the 2018 Abu Dhabi Grand Prix.",
        use_container_width=True # Replaced the deprecated parameter
    )
    # ------------------------------------

except FileNotFoundError:
    st.error(f"Error: 'image.png' not found. Please make sure the image is in the same folder as app.py.")

# --- Two-Column Layout for Text ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.header("About This Project")
        st.write("""
        This application uses a sophisticated XGBoost model, trained on historical race data, to provide high-fidelity F1 strategy simulations. 
        Select a tool from the sidebar to begin planning and predicting race outcomes like a true strategist on the pit wall.
        """)

with col2:
    with st.container(border=True):
        st.header("Scientific Principles Modelled")
        st.write("""
        - 🏎️ **Tyre Degradation:** Predicts lap time drop-off.
        - ⛽ **Fuel Load:** Simulates the car getting lighter and faster.
        - ✨ **Track Evolution:** Models the track "rubbering in".
        - 🧬 **Circuit DNA:** Considers downforce, abrasiveness, etc.
        - 🧑‍🚀 **Driver Factor:** Simulates a driver's unique skill.
        """)

st.info("*If you no longer go for a gap that exists, you are no longer a racing driver.* - Ayrton Senna", icon="🏁")