import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# GLOBAL STYLE (SINGLE INJECTION)
# ======================================================

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 0;
    color: #111111;
}

body {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f6f8fb 40%,
        #edf1f7 100%
    );
}

.block-container {
    max-width: 1400px;
    margin: auto;
    padding-top: 80px;
    padding-bottom: 80px;
    padding-left: 6%;
    padding-right: 6%;
}

.section {
    margin-top: 140px;
    margin-bottom: 140px;
}

.section-title {
    font-size: 36px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 70px;
    letter-spacing: -1px;
}

.hero-title {
    font-size: 72px;
    font-weight: 600;
    letter-spacing: -2.5px;
    text-align: center;
}

.hero-subtitle {
    font-size: 24px;
    color: #6e6e73;
    text-align: center;
    margin-top: 10px;
}

.metric-card {
    background: rgba(255,255,255,0.75);
    border-radius: 24px;
    padding: 30px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 30px 80px rgba(0,0,0,0.05);
    transition: all 0.4s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 40px 90px rgba(0,0,0,0.08);
}

div.stButton > button {
    background: #111111;
    color: white;
    border-radius: 999px;
    padding: 14px 40px;
    border: none;
}

div.stButton > button:hover {
    background: #333333;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO
# ======================================================

st.markdown("<div class='hero-title'>EIGO</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Financial Nervous System</div>", unsafe_allow_html=True)

# ======================================================
# CONTROL SECTION
# ======================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Control Interface</div>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])

with col1:
    shock = st.slider("Macro Instability Amplifier", 0.0, 1.0, 0.0, 0.05)

with col2:
    run_system = st.button("Initialize System")

st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# MAIN EXECUTION BLOCK (ONLY ONE)
# ======================================================

if run_system:

    instability_value = min(0.3 + shock, 1.0)

    # ======================================================
    # INSTABILITY REACTOR
    # ======================================================

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Instability Reactor</div>", unsafe_allow_html=True)

    reactor_size = 220 + 200 * instability_value
    glow_strength = 30 + instability_value * 80

    st.markdown(f"""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        margin-top:40px;
        margin-bottom:40px;
    ">
        <div style="
            width:{reactor_size}px;
            height:{reactor_size}px;
            border-radius:50%;
            background: radial-gradient(circle at center,
                rgba(255,255,255,0.95) 0%,
                rgba(0,0,0,0.75) 100%);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:42px;
            font-weight:600;
            color:#111111;
            box-shadow: 0 0 {glow_strength}px rgba(0,0,0,0.15);
        ">
            {round(instability_value,3)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # FORWARD EVOLUTION (CLEAN)
    # ======================================================

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Forward Structural Evolution</div>", unsafe_allow_html=True)

    projection_steps = 200
    simulations = 800
    base_capital = 1_000_000

    volatility = 0.006 + instability_value * 0.1
    drift = 0.0002

    paths = []

    for _ in range(simulations):
        returns = np.random.normal(drift, volatility, projection_steps)
        path = base_capital * np.cumprod(1 + returns)
        paths.append(path)

    paths = np.array(paths)

    median_path = np.median(paths, axis=0)
    q05 = np.quantile(paths, 0.05, axis=0)
    q95 = np.quantile(paths, 0.95, axis=0)

    time_axis = np.arange(projection_steps)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=q95,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=q05,
        fill='tonexty',
        fillcolor='rgba(0,0,0,0.1)',
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=median_path,
        mode='lines',
        line=dict(width=4, color='black'),
        showlegend=False
    ))

    fig.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
