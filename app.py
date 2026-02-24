import streamlit as st
import matplotlib.pyplot as plt
from eigo_engine import run_engine

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("EIGO — Financial Nervous System")
st.write("A probabilistic financial instability engine.")

st.markdown("---")

# -----------------------------
# RUN ENGINE BUTTON
# -----------------------------

if st.button("Run EIGO Engine"):

    with st.spinner("Analyzing global financial system..."):

        results = run_engine()

    st.success("Analysis Complete")

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Global Instability Index", results["instability"])
    col2.metric("Current Regime", results["regime"])
    col3.metric("90% Capital Survival", results["survival_90"])

    st.markdown("---")

    st.subheader("Digital Twin Capital Distribution")

    fig, ax = plt.subplots()
    ax.hist(results["distribution"], bins=40)
    ax.set_xlabel("Final Capital")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)