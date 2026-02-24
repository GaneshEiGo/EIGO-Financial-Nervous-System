import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL STYLE FOUNDATION (SINGLE SOURCE OF TRUTH)
# ============================================================

st.markdown("""
<style>

/* -------------------------------
REMOVE STREAMLIT DEFAULT UI
--------------------------------*/

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* -------------------------------
GLOBAL RESET
--------------------------------*/

html, body, [class*="css"] {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #111111;
}

/* -------------------------------
BACKGROUND SYSTEM
--------------------------------*/

body {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f6f8fb 40%,
        #edf1f7 100%
    );
}

/* -------------------------------
LAYOUT GRID SYSTEM
--------------------------------*/

.block-container {
    max-width: 1400px;
    margin: auto;
    padding-top: 80px;
    padding-bottom: 80px;
    padding-left: 6%;
    padding-right: 6%;
}

/* -------------------------------
SECTION WRAPPER
--------------------------------*/

.section {
    width: 100%;
    margin-top: 120px;
    margin-bottom: 120px;
}

/* -------------------------------
TYPOGRAPHY SYSTEM
--------------------------------*/

.hero-title {
    font-size: 72px;
    font-weight: 600;
    letter-spacing: -2px;
    text-align: center;
}

.hero-subtitle {
    font-size: 24px;
    color: #6e6e73;
    text-align: center;
    margin-top: 10px;
}

.section-title {
    font-size: 36px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 60px;
    letter-spacing: -1px;
}

/* -------------------------------
METRIC CARD SYSTEM
--------------------------------*/

.metric-card {
    background: rgba(255,255,255,0.7);
    border-radius: 24px;
    padding: 30px;
    text-align: center;
    backdrop-filter: blur(16px);
    box-shadow: 0 30px 80px rgba(0,0,0,0.05);
}

/* -------------------------------
BUTTON SYSTEM
--------------------------------*/

div.stButton > button {
    background: #111111;
    color: white;
    border-radius: 999px;
    padding: 14px 40px;
    font-weight: 500;
    border: none;
}

div.stButton > button:hover {
    background: #333333;
}

/* -------------------------------
REMOVE WHITE HEADER BARS
--------------------------------*/

h1, h2, h3 {
    margin: 0 !important;
    padding: 0 !important;
}

/* -------------------------------
PLOTLY CLEANUP
--------------------------------*/

.js-plotly-plot .plotly {
    border-radius: 30px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("<div class='hero-title'>EIGO</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Financial Nervous System</div>", unsafe_allow_html=True)

# ============================================================
# CONTROL SECTION
# ============================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Control Interface</div>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])

with col1:
    shock = st.slider("Macro Instability Amplifier", 0.0, 1.0, 0.0, 0.05)

with col2:
    run_system = st.button("Initialize System")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<style>

/* ================================
BACKGROUND PARTICLE FIELD
================================ */

body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image:
        radial-gradient(circle, rgba(0,0,0,0.05) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridMove 60s linear infinite;
    z-index: -2;
}

/* ================================
SUBTLE FLOATING DOTS
================================ */

body::after {
    content: "";
    position: fixed;
    top: -200px;
    left: -200px;
    width: 140%;
    height: 140%;
    background:
        radial-gradient(circle at 20% 30%, rgba(0,0,0,0.04) 3px, transparent 4px),
        radial-gradient(circle at 70% 60%, rgba(0,0,0,0.03) 2px, transparent 3px),
        radial-gradient(circle at 40% 80%, rgba(0,0,0,0.04) 2px, transparent 3px);
    animation: floatParticles 120s linear infinite;
    z-index: -1;
}

/* ================================
ANIMATIONS
================================ */

@keyframes gridMove {
    from { transform: translate(0, 0); }
    to { transform: translate(-200px, -200px); }
}

@keyframes floatParticles {
    from { transform: translate(0, 0); }
    to { transform: translate(200px, 200px); }
}

</style>
""", unsafe_allow_html=True)
# Only render reactor if system initialized
if run_system:

    # Simulate instability value for now
    instability_value = 0.32 + shock
    instability_value = min(instability_value, 1.0)

    # Section Wrapper
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Instability Reactor</div>", unsafe_allow_html=True)

    # Dynamic sizing
    base_size = 220
    dynamic_growth = 200 * instability_value
    reactor_size = base_size + dynamic_growth

    # Dynamic glow intensity
    glow_strength = 30 + instability_value * 80

    # Color logic (institutional grayscale scaling)
    if instability_value < 0.4:
        core_color = "rgba(0,0,0,0.85)"
    elif instability_value < 0.7:
        core_color = "rgba(0,0,0,0.65)"
    else:
        core_color = "rgba(0,0,0,0.55)"

    st.markdown(f"""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        align-items:center;
        margin-top:40px;
        margin-bottom:40px;
    ">
        <div style="
            width:{reactor_size}px;
            height:{reactor_size}px;
            border-radius:50%;
            background: radial-gradient(circle at center,
                rgba(255,255,255,0.95) 0%,
                {core_color} 100%);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:42px;
            font-weight:600;
            color:#111111;
            box-shadow: 0 0 {glow_strength}px rgba(0,0,0,0.15);
            animation: reactorBreath 6s ease-in-out infinite;
        ">
            {round(instability_value,3)}
        </div>
    </div>

    <style>
    @keyframes reactorBreath {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Neural Capital Flow</div>", unsafe_allow_html=True)

    projection_steps = 180
    simulation_paths = 60
    base_capital = 1_000_000

    volatility = 0.006 + instability_value * 0.08
    drift = 0.00025

    fig_flow = go.Figure()

    # Generate layered paths
    for i in range(simulation_paths):

        returns = np.random.normal(drift, volatility, projection_steps)
        path = base_capital * np.cumprod(1 + returns)

        normalized = (path - path.min()) / (path.max() - path.min())

        fig_flow.add_trace(
            go.Scatter(
                x=np.arange(projection_steps),
                y=normalized,
                mode="lines",
                line=dict(width=1),
                opacity=0.08,
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Median structural path
    structural_runs = []
    for _ in range(300):
        returns = np.random.normal(drift, volatility, projection_steps)
        path = base_capital * np.cumprod(1 + returns)
        structural_runs.append(path)

    structural_runs = np.array(structural_runs)
    median_path = np.median(structural_runs, axis=0)
    normalized_median = (median_path - median_path.min()) / (median_path.max() - median_path.min())

    fig_flow.add_trace(
        go.Scatter(
            x=np.arange(projection_steps),
            y=normalized_median,
            mode="lines",
            line=dict(width=4),
            opacity=0.95,
            hovertemplate="Median Structural Flow<extra></extra>",
            showlegend=False
        )
    )

    fig_flow.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan"
    )

    fig_flow.update_xaxes(visible=False)
    fig_flow.update_yaxes(visible=False)

    st.plotly_chart(
        fig_flow,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
            "doubleClick": "reset"
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Forward Structural Evolution</div>", unsafe_allow_html=True)

    projection_steps = 220
    simulations = 800
    base_capital = 1_000_000

    volatility_forward = 0.006 + instability_value * 0.10
    drift_forward = 0.0002

    forward_paths = []

    for _ in range(simulations):
        returns = np.random.normal(drift_forward, volatility_forward, projection_steps)
        path = base_capital * np.cumprod(1 + returns)
        forward_paths.append(path)

    forward_paths = np.array(forward_paths)

    median_path = np.median(forward_paths, axis=0)
    q05 = np.quantile(forward_paths, 0.05, axis=0)
    q25 = np.quantile(forward_paths, 0.25, axis=0)
    q75 = np.quantile(forward_paths, 0.75, axis=0)
    q95 = np.quantile(forward_paths, 0.95, axis=0)

    time_axis = np.arange(projection_steps)

    fig_forward = go.Figure()

    # Outer band (5–95)
    fig_forward.add_trace(go.Scatter(
        x=time_axis,
        y=q95,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_forward.add_trace(go.Scatter(
        x=time_axis,
        y=q05,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,0,0,0.08)',
        line=dict(width=0),
        showlegend=False
    ))

    # Core band (25–75)
    fig_forward.add_trace(go.Scatter(
        x=time_axis,
        y=q75,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_forward.add_trace(go.Scatter(
        x=time_axis,
        y=q25,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,0,0,0.15)',
        line=dict(width=0),
        showlegend=False
    ))

    # Median
    fig_forward.add_trace(go.Scatter(
        x=time_axis,
        y=median_path,
        mode='lines',
        line=dict(width=4, color='black'),
        hovertemplate="Median Projection<extra></extra>",
        showlegend=False
    ))

    fig_forward.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan"
    )

    fig_forward.update_xaxes(visible=False)
    fig_forward.update_yaxes(visible=False)

    st.plotly_chart(
        fig_forward,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
            "doubleClick": "reset"
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Institutional Risk Overview</div>", unsafe_allow_html=True)

    # Simulated core metrics
    regime_state = "Compression" if instability_value < 0.5 else "Elevated"
    var_95_estimate = 1_000_000 * (1 - 0.05 - instability_value * 0.1)
    survival_90 = round(1 - instability_value * 0.4, 3)
    composite_risk = round(instability_value * 0.85, 3)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Instability Index</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>{round(instability_value,3)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Current Regime</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>{regime_state}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>VaR 95% (Est.)</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>{round(var_95_estimate,2)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>90% Survival</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>{survival_90}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Systemic Contagion Network</div>", unsafe_allow_html=True)

    # Core asset nodes
    nodes = [
        "Equities",
        "Credit",
        "Rates",
        "Volatility",
        "Commodities",
        "Emerging Markets"
    ]

    node_count = len(nodes)

    # Circular geometry
    angle_step = 2 * np.pi / node_count
    radius = 1

    node_positions = {}
    for i, node in enumerate(nodes):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        node_positions[node] = (x, y)

    fig_network = go.Figure()

    # Dynamic edge intensity scaling
    base_connection_strength = instability_value

    # Draw edges
    for i in range(node_count):
        for j in range(i + 1, node_count):

            node_a = nodes[i]
            node_b = nodes[j]

            x0, y0 = node_positions[node_a]
            x1, y1 = node_positions[node_b]

            connection_strength = base_connection_strength * np.random.uniform(0.7, 1.2)
            opacity_strength = min(0.05 + connection_strength * 0.6, 0.8)

            fig_network.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(width=2),
                    opacity=opacity_strength,
                    hoverinfo='skip',
                    showlegend=False
                )
            )

    # Draw nodes
    for node in nodes:

        x, y = node_positions[node]

        node_size = 25 + instability_value * 40

        fig_network.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(size=node_size),
                text=[node],
                textposition="top center",
                textfont=dict(size=12),
                hoverinfo='skip',
                showlegend=False
            )
        )

    fig_network.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode=False
    )

    fig_network.update_xaxes(visible=False)
    fig_network.update_yaxes(visible=False)

    st.plotly_chart(
        fig_network,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Tail Risk Distribution</div>", unsafe_allow_html=True)

    # Simulate daily return distribution
    samples = 6000
    mean_return = 0.0002
    vol_return = 0.01 + instability_value * 0.06

    simulated_returns = np.random.normal(mean_return, vol_return, samples)

    # Compute VaR and CVaR
    var_95 = np.percentile(simulated_returns, 5)
    cvar_95 = simulated_returns[simulated_returns <= var_95].mean()

    # Smooth density estimation
    hist_y, hist_x = np.histogram(simulated_returns, bins=120, density=True)
    x_mid = (hist_x[:-1] + hist_x[1:]) / 2

    fig_tail = go.Figure()

    # Main density curve
    fig_tail.add_trace(go.Scatter(
        x=x_mid,
        y=hist_y,
        mode='lines',
        line=dict(width=4, color='black'),
        name="Return Density",
        hovertemplate="Return Density<extra></extra>"
    ))

    # Shade left tail region
    tail_mask = x_mid <= var_95

    fig_tail.add_trace(go.Scatter(
        x=x_mid[tail_mask],
        y=hist_y[tail_mask],
        fill='tozeroy',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0,0,0,0.18)',
        showlegend=False,
        hoverinfo='skip'
    ))

    # VaR marker line
    fig_tail.add_vline(
        x=var_95,
        line_width=3,
        line_dash="dash",
        line_color="black"
    )

    fig_tail.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan"
    )

    fig_tail.update_xaxes(title=None)
    fig_tail.update_yaxes(visible=False)

    st.plotly_chart(
        fig_tail,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True
        }
    )

    # Institutional metric row
    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>VaR 95%</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>{round(var_95,4)}</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>CVaR 95%</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>{round(cvar_95,4)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Global Stress Matrix</div>", unsafe_allow_html=True)

    # Define regions
    regions = [
        "United States",
        "Europe",
        "Asia",
        "Emerging Markets",
        "Commodities",
        "Credit Markets"
    ]

    # Simulated regional stress scaling from instability
    regional_stress = {}

    for region in regions:
        regional_stress[region] = min(
            round(instability_value * np.random.uniform(0.7, 1.3), 3),
            1.0
        )

    # Layout grid: 3 columns × 2 rows
    col1, col2, col3 = st.columns(3)

    region_list = list(regional_stress.items())

    def render_region_card(region_name, stress_value):

        # Stress intensity background scaling
        intensity_opacity = 0.05 + stress_value * 0.4

        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,{intensity_opacity});
            border-radius:24px;
            padding:40px;
            margin-bottom:40px;
            text-align:center;
            transition:all 0.6s ease;
        ">
            <div style="font-size:14px; color:#6e6e73;">
                {region_name}
            </div>
            <div style="font-size:30px; font-weight:600; margin-top:12px;">
                {stress_value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        render_region_card(region_list[0][0], region_list[0][1])
        render_region_card(region_list[3][0], region_list[3][1])

    with col2:
        render_region_card(region_list[1][0], region_list[1][1])
        render_region_card(region_list[4][0], region_list[4][1])

    with col3:
        render_region_card(region_list[2][0], region_list[2][1])
        render_region_card(region_list[5][0], region_list[5][1])

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Structural Memory Timeline</div>", unsafe_allow_html=True)

    # Generate synthetic historical instability
    timeline_length = 320
    time_axis = np.arange(timeline_length)

    memory_instability = (
        0.3 +
        0.2 * np.sin(time_axis * 0.03) +
        0.1 * np.sin(time_axis * 0.09 + 2) +
        0.05 * np.random.normal(0, 0.8, timeline_length)
    )

    memory_instability = np.clip(memory_instability, 0.05, 1.0)

    # Replay slider
    replay_position = st.slider(
        "Replay Timeline Position",
        min_value=0,
        max_value=timeline_length - 1,
        value=timeline_length - 1
    )

    current_replay_value = memory_instability[replay_position]

    # Regime classification
    def classify_regime(val):
        if val < 0.25:
            return "Stable"
        elif val < 0.5:
            return "Compression"
        elif val < 0.75:
            return "Elevated"
        else:
            return "Fragile"

    replay_regime = classify_regime(current_replay_value)

    fig_timeline = go.Figure()

    # Main instability curve
    fig_timeline.add_trace(go.Scatter(
        x=time_axis,
        y=memory_instability,
        mode="lines",
        line=dict(width=4, color="black"),
        showlegend=False,
        hovertemplate="Instability Level<extra></extra>"
    ))

    # Highlight replay position
    fig_timeline.add_trace(go.Scatter(
        x=[replay_position],
        y=[current_replay_value],
        mode="markers",
        marker=dict(size=14, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig_timeline.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan"
    )

    fig_timeline.update_xaxes(visible=False)
    fig_timeline.update_yaxes(visible=False)

    st.plotly_chart(
        fig_timeline,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True
        }
    )

    # Regime display row
    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Replay Instability</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {round(current_replay_value,3)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Replay Regime</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {replay_regime}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Autonomous Strategic Intelligence</div>", unsafe_allow_html=True)

    # -----------------------------
    # SYSTEMIC SIGNAL AGGREGATION
    # -----------------------------

    # Normalize components
    instability_component = instability_value
    regime_component = 0.3 if regime_state == "Compression" else 0.6
    tail_component = abs(var_95) * 10
    network_component = instability_value * 0.9

    composite_score = (
        0.4 * instability_component +
        0.25 * regime_component +
        0.2 * tail_component +
        0.15 * network_component
    )

    composite_score = min(composite_score, 1.0)
    composite_score = round(composite_score, 3)

    # -----------------------------
    # STRATEGIC POSTURE CLASSIFIER
    # -----------------------------

    if composite_score < 0.3:
        posture = "Aggressive Expansion"
        guidance = """
        Systemic stability remains supportive.
        Risk premia compression favors capital deployment.
        High-beta exposure acceptable.
        Liquidity buffers can remain moderate.
        """

    elif composite_score < 0.55:
        posture = "Balanced Allocation"
        guidance = """
        Transitional regime conditions.
        Diversification discipline required.
        Avoid leverage expansion.
        Maintain hedging optionality.
        """

    elif composite_score < 0.75:
        posture = "Defensive Rotation"
        guidance = """
        Elevated systemic fragility detected.
        Reduce cyclical exposure.
        Increase capital protection bias.
        Strengthen liquidity reserves.
        """

    else:
        posture = "Capital Preservation Mode"
        guidance = """
        High systemic instability environment.
        Downside convexity risk elevated.
        Prioritize liquidity.
        Avoid directional exposure concentration.
        """

    # -----------------------------
    # DISPLAY STRUCTURED SUMMARY
    # -----------------------------

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Composite Systemic Score</div>
            <div style='font-size:32px; font-weight:600; margin-top:12px;'>
                {composite_score}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-card' style='margin-top:30px;'>
            <div style='font-size:14px; color:#6e6e73;'>Strategic Posture</div>
            <div style='font-size:26px; font-weight:600; margin-top:12px;'>
                {posture}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:16px; line-height:1.8;'>
                {guidance}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    # Dynamic environmental scaling based on instability
    env_intensity = instability_value

    # Background tone mapping
    if env_intensity < 0.3:
        bg_top = "#ffffff"
        bg_mid = "#f6f8fb"
        bg_bottom = "#edf1f7"
        particle_speed = 80
    elif env_intensity < 0.6:
        bg_top = "#fafbfc"
        bg_mid = "#eef2f7"
        bg_bottom = "#e4e9f1"
        particle_speed = 60
    else:
        bg_top = "#f5f6f9"
        bg_mid = "#e8edf5"
        bg_bottom = "#dde3ec"
        particle_speed = 40

    st.markdown(f"""
    <style>

    body {{
        background: linear-gradient(
            180deg,
            {bg_top} 0%,
            {bg_mid} 40%,
            {bg_bottom} 100%
        );
        transition: background 0.8s ease;
    }}

    body::before {{
        animation-duration: {particle_speed}s !important;
    }}

    .metric-card {{
        box-shadow: 0 30px {40 + env_intensity*60}px rgba(0,0,0,0.05);
        transition: box-shadow 0.6s ease;
    }}

    </style>
    """, unsafe_allow_html=True)
    import time

# -------------------------------
# ADAPTIVE SIMULATION SCALING
# -------------------------------

if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Performance Intelligence Layer</div>", unsafe_allow_html=True)

    # Scale simulation intensity based on instability
    if instability_value < 0.3:
        adaptive_simulations = 400
        adaptive_projection_steps = 160
    elif instability_value < 0.6:
        adaptive_simulations = 800
        adaptive_projection_steps = 220
    else:
        adaptive_simulations = 1400
        adaptive_projection_steps = 280

    # -------------------------------
    # SMART CACHING LAYER
    # -------------------------------

    @st.cache_data(show_spinner=False)
    def generate_adaptive_paths(sim_count, steps, vol, drift):
        paths = []
        for _ in range(sim_count):
            returns = np.random.normal(drift, vol, steps)
            path = 1_000_000 * np.cumprod(1 + returns)
            paths.append(path)
        return np.array(paths)

    start_time = time.time()

    adaptive_paths = generate_adaptive_paths(
        adaptive_simulations,
        adaptive_projection_steps,
        0.006 + instability_value * 0.08,
        0.0002
    )

    compute_time = round(time.time() - start_time, 3)

    # -------------------------------
    # PERFORMANCE METRICS DISPLAY
    # -------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Simulations</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {adaptive_simulations}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Projection Steps</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {adaptive_projection_steps}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Compute Time (s)</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {compute_time}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>System Intelligence Narrative</div>", unsafe_allow_html=True)

    explanation_mode = st.radio(
        "Explanation Mode",
        ["Executive Summary", "Technical Analysis"],
        horizontal=True
    )

    # -----------------------------------
    # EXECUTIVE MODE
    # -----------------------------------

    if explanation_mode == "Executive Summary":

        if composite_score < 0.3:
            summary_text = """
            The financial system remains structurally stable.
            Market connectivity is moderate.
            Downside dispersion remains controlled.
            Capital deployment conditions are supportive.
            """

        elif composite_score < 0.55:
            summary_text = """
            The system is in a transitional phase.
            Cross-asset correlation is rising.
            Tail risk is moderately expanding.
            Caution and diversification discipline are recommended.
            """

        elif composite_score < 0.75:
            summary_text = """
            Elevated structural fragility detected.
            Shock transmission probability increasing.
            Downside risk asymmetry widening.
            Defensive allocation bias advised.
            """

        else:
            summary_text = """
            High systemic instability environment.
            Market contagion risk elevated.
            Tail risk amplification observed.
            Capital preservation strategy recommended.
            """

        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:18px; line-height:1.8;'>
                {summary_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------
    # TECHNICAL MODE
    # -----------------------------------

    else:

        technical_text = f"""
        Composite Systemic Score: {composite_score}

        Current Regime Classification: {regime_state}

        Instability Index: {round(instability_value,3)}

        The system exhibits volatility clustering dynamics
        with cross-asset correlation expansion consistent
        with transitional regime migration.

        Tail distribution analysis indicates VaR widening
        and conditional shortfall amplification.

        Network contagion density scales proportionally
        with systemic instability intensity.
        """

        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:16px; line-height:1.9;'>
                {technical_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Structural Density Surface</div>", unsafe_allow_html=True)

    # Define structural axes
    volatility_axis = np.linspace(0.01, 0.5, 60)
    correlation_axis = np.linspace(0.0, 1.0, 60)

    V, C = np.meshgrid(volatility_axis, correlation_axis)

    # Structural systemic stress function
    # Higher volatility + higher correlation + instability amplify stress
    stress_surface = (
        (V * 2.5) *
        (C * 1.8) *
        (0.5 + instability_value)
    )

    fig_surface = go.Figure(
        data=[
            go.Contour(
                z=stress_surface,
                x=volatility_axis,
                y=correlation_axis,
                colorscale="Greys",
                contours=dict(showlines=False),
                hovertemplate="Vol: %{x}<br>Corr: %{y}<br>Stress: %{z}<extra></extra>"
            )
        ]
    )

    fig_surface.update_layout(
        height=540,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig_surface.update_xaxes(title="Volatility")
    fig_surface.update_yaxes(title="Correlation")

    st.plotly_chart(
        fig_surface,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Shock Propagation Simulator</div>", unsafe_allow_html=True)

    # Define nodes (reuse earlier structure conceptually)
    nodes = [
        "Equities",
        "Credit",
        "Rates",
        "Volatility",
        "Commodities",
        "Emerging Markets"
    ]

    shock_origin = st.selectbox("Select Shock Origin", nodes)

    propagation_steps = 5
    decay_factor = 0.6 + instability_value * 0.3

    # Initial shock vector
    shock_vector = {node: 0 for node in nodes}
    shock_vector[shock_origin] = 1.0

    # Simulate propagation
    history = []

    for step in range(propagation_steps):
        new_vector = {}
        for node in nodes:
            transmission = sum(shock_vector[n] for n in nodes) / len(nodes)
            new_vector[node] = transmission * decay_factor
        history.append(new_vector)
        shock_vector = new_vector

    # Visualize final state
    final_state = history[-1]

    col1, col2, col3 = st.columns(3)

    for idx, node in enumerate(nodes):
        intensity = min(final_state[node], 1.0)
        opacity = 0.05 + intensity * 0.6

        card_html = f"""
        <div style="
            background: rgba(0,0,0,{opacity});
            border-radius:24px;
            padding:30px;
            text-align:center;
            margin-bottom:30px;
            transition:all 0.6s ease;
        ">
            <div style="font-size:14px; color:#6e6e73;">
                {node}
            </div>
            <div style="font-size:26px; font-weight:600; margin-top:12px;">
                {round(intensity,3)}
            </div>
        </div>
        """

        if idx % 3 == 0:
            with col1:
                st.markdown(card_html, unsafe_allow_html=True)
        elif idx % 3 == 1:
            with col2:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Regime Transition Matrix</div>", unsafe_allow_html=True)

    regimes = ["Stable", "Compression", "Elevated", "Fragile"]

    # Base transition matrix (Markov-like)
    base_matrix = np.array([
        [0.75, 0.20, 0.04, 0.01],
        [0.15, 0.65, 0.15, 0.05],
        [0.05, 0.20, 0.60, 0.15],
        [0.02, 0.08, 0.25, 0.65]
    ])

    # Adjust transition intensity with instability
    instability_factor = instability_value * 0.3

    adjusted_matrix = base_matrix.copy()

    for i in range(len(regimes)):
        for j in range(len(regimes)):
            if i != j:
                adjusted_matrix[i, j] += instability_factor * 0.05
        adjusted_matrix[i, i] -= instability_factor * 0.15

    # Normalize rows
    adjusted_matrix = adjusted_matrix / adjusted_matrix.sum(axis=1, keepdims=True)

    fig_matrix = go.Figure(
        data=go.Heatmap(
            z=adjusted_matrix,
            x=regimes,
            y=regimes,
            colorscale="Greys",
            hovertemplate="From %{y} → To %{x}: %{z:.2f}<extra></extra>"
        )
    )

    fig_matrix.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(
        fig_matrix,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    # Next-state probability summary
    current_regime_index = regimes.index(regime_state)
    next_probs = adjusted_matrix[current_regime_index]

    colA, colB, colC, colD = st.columns(4)

    for idx, regime in enumerate(regimes):

        card_html = f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>
                Prob → {regime}
            </div>
            <div style='font-size:26px; font-weight:600; margin-top:10px;'>
                {round(next_probs[idx],2)}
            </div>
        </div>
        """

        if idx == 0:
            with colA:
                st.markdown(card_html, unsafe_allow_html=True)
        elif idx == 1:
            with colB:
                st.markdown(card_html, unsafe_allow_html=True)
        elif idx == 2:
            with colC:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            with colD:
                st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Liquidity Stress Indicator</div>", unsafe_allow_html=True)

    # Simulated liquidity components
    spread_proxy = 0.002 + instability_value * 0.01
    volatility_component = 0.01 + instability_value * 0.08

    liquidity_stress_index = (
        spread_proxy * 0.5 +
        volatility_component * 0.4 +
        instability_value * 0.6
    )

    liquidity_stress_index = min(liquidity_stress_index, 1.0)
    liquidity_stress_index = round(liquidity_stress_index, 3)

    # Time series liquidity curve
    liquidity_steps = 200
    time_axis = np.arange(liquidity_steps)

    liquidity_curve = (
        liquidity_stress_index +
        0.1 * np.sin(time_axis * 0.05) +
        0.05 * np.random.normal(0, 0.4, liquidity_steps)
    )

    liquidity_curve = np.clip(liquidity_curve, 0, 1)

    fig_liquidity = go.Figure()

    fig_liquidity.add_trace(go.Scatter(
        x=time_axis,
        y=liquidity_curve,
        mode='lines',
        line=dict(width=4, color='black'),
        showlegend=False,
        hovertemplate="Liquidity Stress<extra></extra>"
    ))

    fig_liquidity.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan"
    )

    fig_liquidity.update_xaxes(visible=False)
    fig_liquidity.update_yaxes(visible=False)

    st.plotly_chart(
        fig_liquidity,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    # Institutional summary cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Liquidity Stress Index</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {liquidity_stress_index}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        liquidity_state = (
            "Normal Liquidity" if liquidity_stress_index < 0.4
            else "Tightening Conditions" if liquidity_stress_index < 0.7
            else "Liquidity Compression"
        )

        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Liquidity Regime</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {liquidity_state}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Capital Resilience Simulation</div>", unsafe_allow_html=True)

    horizon_days = 365
    simulations = 1200
    initial_capital = 1_000_000

    drift = 0.00025
    volatility = 0.008 + instability_value * 0.09

    final_values = []
    max_drawdowns = []

    for _ in range(simulations):

        returns = np.random.normal(drift, volatility, horizon_days)
        capital_path = initial_capital * np.cumprod(1 + returns)

        final_values.append(capital_path[-1])

        running_max = np.maximum.accumulate(capital_path)
        drawdown = (capital_path - running_max) / running_max
        max_drawdowns.append(drawdown.min())

    final_values = np.array(final_values)
    max_drawdowns = np.array(max_drawdowns)

    survival_probability = np.mean(final_values > 0.85 * initial_capital)
    survival_probability = round(float(survival_probability), 3)

    resilience_score = round(1 - abs(np.mean(max_drawdowns)), 3)

    # Density plot of final capital
    hist_y, hist_x = np.histogram(final_values, bins=120, density=True)
    x_mid = (hist_x[:-1] + hist_x[1:]) / 2

    fig_resilience = go.Figure()

    fig_resilience.add_trace(go.Scatter(
        x=x_mid,
        y=hist_y,
        mode='lines',
        line=dict(width=4, color='black'),
        showlegend=False,
        hovertemplate="Final Capital Density<extra></extra>"
    ))

    fig_resilience.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan"
    )

    fig_resilience.update_xaxes(title=None)
    fig_resilience.update_yaxes(visible=False)

    st.plotly_chart(
        fig_resilience,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    # Institutional summary row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Survival Probability (85%)</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {survival_probability}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:14px; color:#6e6e73;'>Resilience Score</div>
            <div style='font-size:28px; font-weight:600; margin-top:10px;'>
                {resilience_score}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if run_system:

    st.markdown("""
    <style>

    /* ==============================
       GLOBAL SPACING NORMALIZATION
    ============================== */

    .section {
        margin-top: 140px !important;
        margin-bottom: 140px !important;
    }

    .section-title {
        margin-bottom: 70px !important;
    }

    /* ==============================
       TYPOGRAPHY REFINEMENT
    ============================== */

    .hero-title {
        letter-spacing: -2.5px;
    }

    .metric-card div {
        transition: all 0.4s ease;
    }

    /* ==============================
       SMOOTH UI TRANSITIONS
    ============================== */

    * {
        transition: background 0.6s ease,
                    box-shadow 0.6s ease,
                    transform 0.4s ease;
    }

    /* ==============================
       SLIDER STYLING
    ============================== */

    .stSlider > div > div {
        background: linear-gradient(90deg, #111111, #555555);
        height: 6px;
        border-radius: 4px;
    }

    /* ==============================
       BUTTON MICRO INTERACTION
    ============================== */

    div.stButton > button {
        transform: translateY(0px);
        transition: all 0.25s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }

    /* ==============================
       METRIC CARD ELEVATION BALANCE
    ============================== */

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 40px 90px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)
