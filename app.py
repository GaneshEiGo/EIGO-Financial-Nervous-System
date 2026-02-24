import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# GLOBAL LAYOUT HELPERS
# ------------------------------

def section(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

def two_panel(left_func, right_func, ratio=(2,1)):
    """
    Render two panels side by side.
    left_func and right_func are functions that render content.
    """
    col_left, col_right = st.columns(ratio)
    with col_left:
        left_func()
    with col_right:
        right_func()

# ------------------------------
# GLOBAL CSS
# ------------------------------

st.markdown("""
<style>

/* RESET DEFAULT UI */
#MainMenu, footer, header {visibility:hidden;}

/* FONT SYSTEM */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* BACKGROUND */
body {
    background: #fcfcfd;
}

/* SECTIONS */
.section-title {
    font-size:42px;
    font-weight:600;
    text-align:center;
    margin:80px 0 40px 0;
}

/* METRIC CARD */
.metric-card {
    background:rgba(255,255,255,0.75);
    padding:26px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 20px 80px rgba(0,0,0,0.06);
}

/* SLIDER */
.stSlider > div > div {
    margin-top:8px;
    margin-bottom:8px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# HERO
# ------------------------------

st.markdown("<h1 style='text-align:center; font-size:68px;'>EIGO</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#6e6e73;'>Financial Nervous System</h3>", unsafe_allow_html=True)

# ------------------------------
# CONTROL (Live Instability)
# ------------------------------

section("System Control Interface")

instability_value = st.slider("Macro Instability Level", 0.0, 1.0, 0.3, 0.01)

# ------------------------------
# PLACEHOLDER SECTIONS
# ------------------------------

section("1) Instability Reactor")
st.write("⏳ Reactor will display here.")

section("2) Forward Structural Evolution")
st.write("⏳ Evolution chart will display here.")

section("3) Risk Metrics Grid")
st.write("⏳ Metrics grid will display here.")

section("4) Contagion Network")
st.write("⏳ Network will display here.")

section("5) Tail Risk")
st.write("⏳ Tail risk chart will display here.")

# ------------------------------
# FOOTER
# ------------------------------

st.markdown("<div style='text-align:center; margin-top:80px; color:#8e8e93;'>EIGO — Build by you • No Money • Institutional First</div>", unsafe_allow_html=True)
