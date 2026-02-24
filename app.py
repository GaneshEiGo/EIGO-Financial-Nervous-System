import streamlit as st
import numpy as np
import plotly.graph_objects as go
from eigo_engine import run_engine, digital_twin

# =========================================================
# GLOBAL CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# DESIGN SYSTEM — APPLE / GOOGLE HYBRID
# =========================================================

st.markdown("""
<style>

/* Remove Streamlit default header */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Global font */
html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #ffffff;
}

/* Main container spacing */
.block-container {
    padding-top: 5rem;
    padding-left: 8%;
    padding-right: 8%;
    padding-bottom: 4rem;
}

/* Title styling */
.hero-title {
    font-size: 48px;
    font-weight: 600;
    letter-spacing: -1px;
    text-align: center;
}

.hero-subtitle {
    font-size: 20px;
    color: #666666;
    text-align: center;
    margin-top: -10px;
}

.hero-description {
    font-size: 16px;
    color: #888888;
    text-align: center;
    max-width: 700px;
    margin: auto;
    margin-top: 20px;
}

/* Glass metric card */
.metric-card {
    background: #f7f7f7;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    transition: 0.3s ease;
}

.metric-card:hover {
    background: #f0f0f0;
}

/* Divider */
.soft-divider {
    height: 1px;
    background: #eeeeee;
    margin-top: 50px;
    margin-bottom: 50px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("<div class='hero-title'>EIGO</div>", unsafe_allow_html=True)

st.markdown("<div class='hero-subtitle'>Financial Nervous System</div>", unsafe_allow_html=True)

st.markdown("""
<div class='hero-description'>
A probabilistic intelligence engine modeling systemic fragility,
contagion dynamics, and capital survival under uncertainty.
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# =========================================================
# INTERACTION CONTROL PANEL
# =========================================================

control_col1, control_col2 = st.columns([2,1])

with control_col1:
    shock = st.slider(
        "Fragility Shock Amplifier",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )

with control_col2:
    run_button = st.button("Run System")

st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
# =========================================================
# ENGINE EXECUTION
# =========================================================

if run_button:

    with st.spinner("Analyzing structural financial state..."):

        results = run_engine()
        adjusted_instability = min(1.0, results["instability"] + shock)
        twin = digital_twin(adjusted_instability)

    # =====================================================
    # ANIMATED INSTABILITY PULSE
    # =====================================================

    pulse_container = st.empty()

    for i in np.linspace(0, adjusted_instability, 20):
        pulse_container.markdown(f"""
        <div style="
            width:150px;
            height:150px;
            border-radius:50%;
            margin:auto;
            background: radial-gradient(circle, rgba(0,0,0,{i}) 0%, rgba(0,0,0,0.02) 70%);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:28px;
            font-weight:600;
        ">
            {round(adjusted_instability,3)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # METRIC GRID
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#888;">System Regime</div>
            <div style="font-size:22px;font-weight:600;">{results["regime"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#888;">90% Capital Survival</div>
            <div style="font-size:22px;font-weight:600;">{twin["survival_90"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#888;">80% Capital Survival</div>
            <div style="font-size:22px;font-weight:600;">{twin["survival_80"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # =====================================================
    # FUTURISTIC CAPITAL FLOW FIELD
    # =====================================================

    st.markdown("<h3 style='text-align:center;'>Capital Flow Field</h3>", unsafe_allow_html=True)

    distribution = twin["distribution"]

    # Create flowing surface
    x = np.linspace(0, 1, 400)
    y = np.sin(10 * x * adjusted_instability + 0.5) * adjusted_instability

    capital_curve = np.interp(
        np.linspace(0, len(distribution) - 1, 400),
        np.arange(len(distribution)),
        np.sort(distribution)
    )

    normalized_capital = (capital_curve - capital_curve.min()) / (capital_curve.max() - capital_curve.min())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=normalized_capital,
            mode='lines',
            line=dict(width=3),
            fill='tozeroy'
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # =====================================================
    # SYSTEM EVOLUTION ENGINE
    # =====================================================

    st.markdown("<h3 style='text-align:center;'>System Evolution Timeline</h3>", unsafe_allow_html=True)

    # Simulate capital paths dynamically
    simulated_paths = []

    base_capital = 1_000_000
    volatility_factor = 0.01 + adjusted_instability * 0.05
    drift = 0.0003

    for _ in range(25):  # 25 evolving paths
        returns = np.random.normal(drift, volatility_factor, 120)
        path = base_capital * np.cumprod(1 + returns)
        simulated_paths.append(path)

    time_axis = np.arange(120)

    fig_evolution = go.Figure()

    for path in simulated_paths:
        normalized = (path - path.min()) / (path.max() - path.min())
        fig_evolution.add_trace(
            go.Scatter(
                x=time_axis,
                y=normalized,
                mode="lines",
                line=dict(width=1),
                opacity=0.3,
                showlegend=False
            )
        )

    fig_evolution.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        template="plotly_white",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_evolution, use_container_width=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # =====================================================
    # SYSTEM INTELLIGENCE NARRATIVE ENGINE
    # =====================================================

    st.markdown("<h3 style='text-align:center;'>System Intelligence Report</h3>", unsafe_allow_html=True)

    # Determine system state
    if adjusted_instability < 0.25:
        system_state = "Structural Stability"
        system_color = "#2E8B57"
        system_message = """
        The global financial structure is operating within stable boundaries.
        Correlation compression is limited. Capital dispersion is healthy.
        Systemic fracture probability remains low.
        """
    elif adjusted_instability < 0.5:
        system_state = "Compression Phase"
        system_color = "#E6A700"
        system_message = """
        Cross-asset coupling is increasing.
        Volatility clustering patterns are emerging.
        Early structural pressure detected within systemic layers.
        """
    elif adjusted_instability < 0.75:
        system_state = "Elevated Fragility"
        system_color = "#D2691E"
        system_message = """
        Systemic fragility has intensified.
        Credit spread divergence expanding.
        Network contagion probability rising.
        Structural stress propagation accelerating.
        """
    else:
        system_state = "Systemic Instability"
        system_color = "#B22222"
        system_message = """
        High systemic stress detected.
        Contagion risk elevated.
        Capital preservation probability declining.
        Structural regime transition likely.
        """

    st.markdown(f"""
    <div style="
        background:#f8f8f8;
        padding:40px;
        border-radius:24px;
        margin-top:20px;
    ">
        <div style="font-size:24px;font-weight:600;color:{system_color};text-align:center;">
            {system_state}
        </div>
        <div style="margin-top:20px;font-size:16px;color:#555;text-align:center;max-width:800px;margin-left:auto;margin-right:auto;">
            {system_message}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # FUTURISTIC SIGNAL FIELD
    # =====================================================

    st.markdown("<h3 style='text-align:center;'>Structural Signal Field</h3>", unsafe_allow_html=True)

    signal_time = np.linspace(0, 4 * np.pi, 500)
    amplitude = adjusted_instability + 0.2
    wave = amplitude * np.sin(signal_time * (1 + adjusted_instability * 2))

    fig_signal = go.Figure()

    fig_signal.add_trace(
        go.Scatter(
            x=signal_time,
            y=wave,
            mode="lines",
            line=dict(width=3),
            fill="tozeroy"
        )
    )

    fig_signal.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    st.plotly_chart(fig_signal, use_container_width=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

    # =====================================================
    # ADAPTIVE SYSTEM DEPTH PANEL
    # =====================================================

    st.markdown("<h3 style='text-align:center;'>Adaptive Depth Metrics</h3>", unsafe_allow_html=True)

    depth_col1, depth_col2, depth_col3 = st.columns(3)

    network_density = round(adjusted_instability * np.random.uniform(0.8, 1.2), 3)
    contagion_strength = round(adjusted_instability * np.random.uniform(0.6, 1.4), 3)
    regime_shift_prob = round(adjusted_instability * np.random.uniform(0.5, 1.5), 3)

    depth_col1.metric("Network Density", network_density)
    depth_col2.metric("Contagion Strength", contagion_strength)
    depth_col3.metric("Regime Shift Probability", regime_shift_prob)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # =====================================================
    # FINAL SYSTEM FOOTER
    # =====================================================

    st.markdown("""
    <div style="text-align:center;color:#999;font-size:13px;margin-top:60px;">
    EIGO — Financial Nervous System<br>
    Structural Intelligence • Probabilistic Modeling • Adaptive Simulation
    </div>
    """, unsafe_allow_html=True)
