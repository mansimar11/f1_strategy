# pages/4_📈_Data_Analytics.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Data Analytics", page_icon="📈", layout="wide")
st.title("Data Analytics Dashboard 📈")
st.info("This is the dedicated space for your embedded Power BI dashboard.")

html_code = """
<iframe title="Your Report Title" width="100%" height="600" src="YOUR_POWER_BI_LINK_HERE" frameborder="0" allowFullScreen="true"></iframe>
"""
# components.html(html_code, height=600)
st.warning("No Power BI embed code has been provided yet. Edit this file to add your dashboard.")