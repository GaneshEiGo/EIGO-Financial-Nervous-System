import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from eigo_engine import run_engine, digital_twin

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide"
)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("EIGO")
st.caption("Financial Nervous System — Probabilistic Risk Engine")

st.markdown("")

# ------------------------------------------------
# USER CONTROLS
# ------------------------------------------------

colA, colB = st.columns(2)

with colA:
    shock = st.slider(
        "Fragility Shock Level",
        0.0, 1.0, 0.0, 0.05
    )

with colB:
    sim_days = st.slider(
        "Simulation Horizon (Days)",
        30, 180, 90, 30
    )

st.markdown("---")

# ------------------------------------------------
# RUN ENGINE
# ------------------------------------------------

if st.button("Run Analysis"):

    with st.spinner("Analyzing global financial system..."):

        results = run_engine()

        adjusted_instability = min(1.0, results["instability"] + shock)

        twin = digital_twin(adjusted_instability)

    st.success("Analysis Complete")

    # ------------------------------------------------
    # METRICS SECTION
    # ------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Instability Index", adjusted_instability)
    col2.metric("Regime", results["regime"])
    col3.metric("90% Survival", twin["survival_90"])
    col4.metric("80% Survival", twin["survival_80"])

    st.markdown("")

    # ------------------------------------------------
    # DISTRIBUTION GRAPH (UPGRADED)
    # ------------------------------------------------

    distribution = twin["distribution"]
    mean_value = np.mean(distribution)
    initial_capital = 1_000_000
    threshold_90 = 0.9 * initial_capital
    threshold_80 = 0.8 * initial_capital

    # Google-style color logic
    if adjusted_instability < 0.3:
        primary_color = "#34A853"  # Google green
    elif adjusted_instability < 0.6:
        primary_color = "#FBBC05"  # Google yellow
    else:
        primary_color = "#EA4335"  # Google red

    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram
    ax.hist(distribution, bins=50, density=True, alpha=0.3, color=primary_color)

    # Smooth density curve
    kde = gaussian_kde(distribution)
    x_range = np.linspace(min(distribution), max(distribution), 500)
    ax.plot(x_range, kde(x_range), color=primary_color, linewidth=3)

    # Reference lines
    ax.axvline(initial_capital, linestyle="--", linewidth=2, color="#4285F4", label="Initial Capital")
    ax.axvline(threshold_90, linestyle="--", linewidth=2, color="#FBBC05", label="90% Threshold")
    ax.axvline(threshold_80, linestyle="--", linewidth=2, color="#EA4335", label="80% Threshold")
    ax.axvline(mean_value, linestyle="-", linewidth=3, color=primary_color, label="Mean Outcome")

    ax.set_xlabel("Final Capital")
    ax.set_ylabel("Probability Density")
    ax.set_title("Digital Twin Capital Distribution")

    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)

    # ------------------------------------------------
    # LIVE INTERPRETATION PANEL
    # ------------------------------------------------

    st.markdown("---")

    if adjusted_instability < 0.3:
        st.success("System is stable. Risk levels are structurally low.")
    elif adjusted_instability < 0.6:
        st.warning("Moderate systemic stress detected. Monitor portfolio risk exposure.")
    else:
        st.error("High systemic instability. Capital preservation strategies advised.")
