import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from eigo_engine import run_engine, digital_twin

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------

st.title("EIGO — Financial Nervous System")
st.write("A probabilistic financial instability engine.")

st.markdown("---")

# -----------------------------
# USER CONTROLS
# -----------------------------

shock = st.slider(
    "Fragility Shock Level",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

simulation_days = st.slider(
    "Simulation Horizon (Days)",
    min_value=30,
    max_value=180,
    value=90,
    step=30
)

st.markdown("---")

# -----------------------------
# RUN ENGINE
# -----------------------------

if st.button("Run EIGO Engine"):

    with st.spinner("Analyzing global financial system..."):

        results = run_engine()

        # Adjust instability with shock
        adjusted_instability = min(1.0, results["instability"] + shock)

        # Re-run digital twin with new instability
        twin = digital_twin(adjusted_instability)

    st.success("Analysis Complete")

    # -----------------------------
    # METRICS
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Global Instability Index", adjusted_instability)
    col2.metric("Current Regime", results["regime"])
    col3.metric("90% Capital Survival", twin["survival_90"])

    st.metric("80% Capital Survival", twin["survival_80"])

    st.markdown("---")

    # -----------------------------
    # DISTRIBUTION PLOT
    # -----------------------------

    st.subheader("Digital Twin Capital Distribution")

    fig, ax = plt.subplots()
    ax.hist(twin["distribution"], bins=40)
    ax.set_xlabel("Final Capital")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
