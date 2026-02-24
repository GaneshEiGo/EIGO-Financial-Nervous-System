import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from eigo_engine import run_engine, digital_twin

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="EIGO",
    layout="centered"
)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("EIGO")
st.caption("Financial Nervous System — Probabilistic Risk Engine")

st.markdown("")

# ------------------------------------------------
# CONTROLS
# ------------------------------------------------

shock = st.slider(
    "Fragility Shock Level",
    0.0, 1.0, 0.0, 0.05
)

st.markdown("")

# ------------------------------------------------
# RUN ENGINE
# ------------------------------------------------

if st.button("Run Analysis"):

    with st.spinner("Analyzing global financial system..."):

        results = run_engine()

        adjusted_instability = min(1.0, results["instability"] + shock)

        twin = digital_twin(adjusted_instability)

    # ------------------------------------------------
    # KPI ROW
    # ------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Instability Index", adjusted_instability)
    col2.metric("Regime", results["regime"])
    col3.metric("90% Survival", twin["survival_90"])

    st.markdown("")

    # ------------------------------------------------
    # MODERN INTERACTIVE DISTRIBUTION
    # ------------------------------------------------

    distribution = twin["distribution"]
    mean_value = np.mean(distribution)
    initial_capital = 1_000_000
    threshold_90 = 0.9 * initial_capital

    kde = gaussian_kde(distribution)
    x_range = np.linspace(min(distribution), max(distribution), 500)
    density = kde(x_range)

    fig = go.Figure()

    # Smooth density curve
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=density,
            mode="lines",
            line=dict(width=3),
            name="Distribution"
        )
    )

    # Vertical markers
    fig.add_vline(
        x=initial_capital,
        line_width=2,
        line_dash="dash",
        annotation_text="Initial Capital",
        annotation_position="top"
    )

    fig.add_vline(
        x=threshold_90,
        line_width=2,
        line_dash="dot",
        annotation_text="90% Threshold",
        annotation_position="top"
    )

    fig.add_vline(
        x=mean_value,
        line_width=3,
        annotation_text="Mean Outcome",
        annotation_position="top"
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        showlegend=False
    )

    fig.update_yaxes(visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # LIVE INTERPRETATION
    # ------------------------------------------------

    if adjusted_instability < 0.3:
        st.success("System stability is strong.")
    elif adjusted_instability < 0.6:
        st.warning("Moderate systemic stress detected.")
    else:
        st.error("High systemic instability.")
