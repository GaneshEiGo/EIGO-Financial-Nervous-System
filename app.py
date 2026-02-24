import streamlit as st
import numpy as np
import plotly.graph_objects as go
from eigo_engine import run_engine, digital_twin

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="EIGO",
    layout="centered",
)

# ------------------------------------------------
# CLEAN CSS (Minimal Modern Style)
# ------------------------------------------------

st.markdown("""
    <style>
        body {
            background-color: white;
        }
        .main {
            padding-top: 3rem;
        }
        h1 {
            font-weight: 600;
            letter-spacing: -1px;
        }
        .stMetric {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown("<h1 style='text-align:center;'>EIGO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: grey;'>Financial Nervous System</p>", unsafe_allow_html=True)

st.markdown("")

# ------------------------------------------------
# CONTROLS
# ------------------------------------------------

shock = st.slider(
    "Fragility Shock",
    0.0, 1.0, 0.0, 0.05
)

st.markdown("")

# ------------------------------------------------
# ENGINE EXECUTION
# ------------------------------------------------

if st.button("Run System"):

    with st.spinner("Analyzing global financial structure..."):
        results = run_engine()
        adjusted_instability = min(1.0, results["instability"] + shock)
        twin = digital_twin(adjusted_instability)

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    col1.metric("Instability", adjusted_instability)
    col2.metric("Regime", results["regime"])
    col3.metric("Survival (90%)", twin["survival_90"])

    st.markdown("---")
    # ------------------------------------------------
    # PREMIUM CAPITAL FLOW VISUAL
    # ------------------------------------------------

    distribution = twin["distribution"]
    sorted_values = np.sort(distribution)

    # Create smooth curve
    x_vals = np.linspace(0, len(sorted_values) - 1, 400)
    y_vals = np.interp(x_vals, np.arange(len(sorted_values)), sorted_values)

    # Normalize for soft visual scaling
    y_normalized = (y_vals - min(y_vals)) / (max(y_vals) - min(y_vals))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_normalized,
            mode="lines",
            line=dict(width=3),
            fill="tozeroy"
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # INTERPRETATION PANEL
    # ------------------------------------------------

    if adjusted_instability < 0.3:
        st.markdown("<p style='text-align:center; color:green;'>System stability strong.</p>", unsafe_allow_html=True)
    elif adjusted_instability < 0.6:
        st.markdown("<p style='text-align:center; color:orange;'>System under moderate stress.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center; color:red;'>High systemic instability detected.</p>", unsafe_allow_html=True)
