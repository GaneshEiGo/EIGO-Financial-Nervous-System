import streamlit as st
import numpy as np
import plotly.graph_objects as go
from eigo_engine import run_engine, digital_twin

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="EIGO",
    layout="wide"
)

# ---------------------------------
# GLOBAL STYLE (APPLE / GOOGLE CLEAN)
# ---------------------------------

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 4rem;
    padding-bottom: 4rem;
}

h1 {
    font-weight: 600;
    letter-spacing: -1px;
}

.metric-card {
    background: #f9f9f9;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
