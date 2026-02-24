import streamlit as st
import numpy as np
import plotly.graph_objects as go
from eigo_engine import run_engine, digital_twin

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL DESIGN ENGINE (FULL IMMERSIVE SYSTEM)
# ============================================================

st.markdown("""
<style>

/* -----------------------------------------------------------
REMOVE STREAMLIT DEFAULT ELEMENTS
----------------------------------------------------------- */

header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* -----------------------------------------------------------
GLOBAL FOUNDATION
----------------------------------------------------------- */

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: radial-gradient(
        circle at 50% 0%,
        #ffffff 0%,
        #f8f9fb 40%,
        #eef1f5 100%
    );
    color: #111111;
}

/* -----------------------------------------------------------
LAYOUT GRID SYSTEM
----------------------------------------------------------- */

.block-container {
    padding-top: 60px;
    padding-left: 8%;
    padding-right: 8%;
    padding-bottom: 80px;
    max-width: 1600px;
    margin: auto;
}

/* -----------------------------------------------------------
TYPOGRAPHY SYSTEM
----------------------------------------------------------- */

.hero-title {
    font-size: 72px;
    font-weight: 600;
    letter-spacing: -2px;
    text-align: center;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 24px;
    text-align: center;
    color: #6e6e73;
    margin-bottom: 30px;
}

.hero-description {
    font-size: 18px;
    text-align: center;
    color: #8e8e93;
    max-width: 900px;
    margin: auto;
    line-height: 1.6;
}

/* -----------------------------------------------------------
FLOATING LAYER SYSTEM
----------------------------------------------------------- */

.floating-layer {
    padding: 60px;
    border-radius: 36px;
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(20px);
    box-shadow: 0 40px 120px rgba(0,0,0,0.06);
    margin-top: 80px;
    margin-bottom: 80px;
    transition: all 0.6s ease;
}

/* Subtle breathing animation */

@keyframes slowFloat {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

.floating-layer {
    animation: slowFloat 18s ease-in-out infinite;
}

/* -----------------------------------------------------------
SECTION TITLES
----------------------------------------------------------- */

.section-title {
    font-size: 34px;
    font-weight: 600;
    margin-bottom: 40px;
    text-align: center;
}

/* -----------------------------------------------------------
BUTTON STYLE
----------------------------------------------------------- */

div.stButton > button {
    background: #111111;
    color: white;
    border-radius: 999px;
    padding: 14px 36px;
    font-size: 16px;
    font-weight: 500;
    border: none;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: scale(1.04);
    background: #333333;
}

/* -----------------------------------------------------------
SLIDER STYLE
----------------------------------------------------------- */

div[data-baseweb="slider"] {
    margin-top: 20px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("<div class='hero-title'>EIGO</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Financial Nervous System</div>", unsafe_allow_html=True)

st.markdown("""
<div class='hero-description'>
An immersive structural intelligence platform modeling systemic fragility,
regime transitions, contagion networks, and probabilistic capital survival
under dynamic financial stress conditions.
</div>
""", unsafe_allow_html=True)

# ============================================================
# CONTROL PANEL (FLOATING LAYER)
# ============================================================

st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>System Control Interface</div>", unsafe_allow_html=True)

control_col1, control_col2 = st.columns([3,1])

with control_col1:
    shock = st.slider(
        "Macro Fragility Amplifier",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )

with control_col2:
    run_system = st.button("Initialize System")

st.markdown("</div>", unsafe_allow_html=True)
# ============================================================
# EXECUTION ENGINE
# ============================================================

if run_system:

    with st.spinner("Calibrating structural intelligence engine..."):
        results = run_engine()
        adjusted_instability = min(1.0, results["instability"] + shock)
        twin = digital_twin(adjusted_instability)

    # ========================================================
    # INSTABILITY CORE FLOATING LAYER
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Instability Reactor Core</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # DYNAMIC SIZE & INTENSITY
    # --------------------------------------------------------

    base_size = 240
    size_multiplier = 220 * adjusted_instability
    core_size = base_size + size_multiplier

    # Color intensity shift
    if adjusted_instability < 0.25:
        glow_color = "rgba(34,197,94,0.5)"
    elif adjusted_instability < 0.5:
        glow_color = "rgba(234,179,8,0.5)"
    elif adjusted_instability < 0.75:
        glow_color = "rgba(249,115,22,0.5)"
    else:
        glow_color = "rgba(239,68,68,0.5)"

    # --------------------------------------------------------
    # MULTI-LAYER REACTOR STRUCTURE
    # --------------------------------------------------------

    st.markdown(f"""
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        position:relative;
        height:500px;
    ">

        <!-- OUTER PULSE RING -->
        <div style="
            position:absolute;
            width:{core_size + 140}px;
            height:{core_size + 140}px;
            border-radius:50%;
            border:2px solid {glow_color};
            opacity:0.15;
            animation: pulseOuter 12s ease-in-out infinite;
        "></div>

        <!-- MIDDLE RING -->
        <div style="
            position:absolute;
            width:{core_size + 70}px;
            height:{core_size + 70}px;
            border-radius:50%;
            border:2px solid {glow_color};
            opacity:0.25;
            animation: pulseMid 9s ease-in-out infinite;
        "></div>

        <!-- INNER CORE -->
        <div style="
            width:{core_size}px;
            height:{core_size}px;
            border-radius:50%;
            background: radial-gradient(circle at center,
                rgba(255,255,255,0.95) 0%,
                {glow_color} 100%);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:42px;
            font-weight:600;
            color:#111;
            box-shadow: 0 0 {40 + adjusted_instability*60}px {glow_color};
            animation: coreBreath 6s ease-in-out infinite;
        ">
            {round(adjusted_instability, 3)}
        </div>

    </div>

    <style>

    @keyframes coreBreath {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}

    @keyframes pulseMid {{
        0% {{ transform: scale(1); opacity:0.25; }}
        50% {{ transform: scale(1.08); opacity:0.15; }}
        100% {{ transform: scale(1); opacity:0.25; }}
    }}

    @keyframes pulseOuter {{
        0% {{ transform: scale(1); opacity:0.15; }}
        50% {{ transform: scale(1.12); opacity:0.08; }}
        100% {{ transform: scale(1); opacity:0.15; }}
    }}

    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # CORE METRICS GRID (PERFECTLY CENTERED)
    # --------------------------------------------------------

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("Regime State", results["regime"])
    metric_col2.metric("90% Survival", twin["survival_90"])
    metric_col3.metric("80% Survival", twin["survival_80"])

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # NEURAL CAPITAL FIELD 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Neural Capital Flow Field</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SIMULATION PARAMETERS
    # --------------------------------------------------------

    projection_steps = 220
    path_count = 70
    base_capital = 1_000_000

    volatility_dynamic = 0.006 + adjusted_instability * 0.085
    drift_dynamic = 0.00025

    fig_field = go.Figure()

    # --------------------------------------------------------
    # LAYERED TRAJECTORY SYSTEM
    # --------------------------------------------------------

    for i in range(path_count):

        returns = np.random.normal(drift_dynamic, volatility_dynamic, projection_steps)
        path = base_capital * np.cumprod(1 + returns)

        normalized_path = (path - path.min()) / (path.max() - path.min())

        depth_opacity = 0.03 + (i / path_count) * 0.35

        fig_field.add_trace(
            go.Scatter(
                x=np.arange(projection_steps),
                y=normalized_path,
                mode="lines",
                line=dict(width=1.4),
                opacity=depth_opacity,
                hoverinfo="skip",
                showlegend=False
            )
        )

    # --------------------------------------------------------
    # DOMINANT STRUCTURAL PATH
    # --------------------------------------------------------

    structural_runs = []

    for _ in range(250):
        returns = np.random.normal(drift_dynamic, volatility_dynamic, projection_steps)
        path = base_capital * np.cumprod(1 + returns)
        structural_runs.append(path)

    structural_runs = np.array(structural_runs)

    median_path = np.median(structural_runs, axis=0)
    normalized_median = (median_path - median_path.min()) / (median_path.max() - median_path.min())

    fig_field.add_trace(
        go.Scatter(
            x=np.arange(projection_steps),
            y=normalized_median,
            mode="lines",
            line=dict(width=5),
            opacity=0.95,
            hovertemplate="Median Capital Flow<extra></extra>",
            showlegend=False
        )
    )

    # --------------------------------------------------------
    # INTERACTION CONTROL (NO UGLY ZOOM BOX)
    # --------------------------------------------------------

    fig_field.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode="pan",
        showlegend=False
    )

    fig_field.update_xaxes(
        showgrid=False,
        zeroline=False,
        visible=False
    )

    fig_field.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False
    )

    st.plotly_chart(
        fig_field,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
            "doubleClick": "reset"
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # STRUCTURAL SIGNAL MATRIX 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Structural Signal Matrix</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SYSTEMIC OSCILLATION FIELD
    # --------------------------------------------------------

    time_axis = np.linspace(0, 10 * np.pi, 800)

    wave_layers = 18
    base_amplitude = 0.06 + adjusted_instability * 0.9

    fig_signal = go.Figure()

    for layer in range(wave_layers):

        frequency = 0.5 + (layer * 0.18)
        phase_shift = layer * 0.45
        amplitude = base_amplitude * (0.4 + layer / wave_layers)

        wave = amplitude * np.sin(time_axis * frequency + phase_shift)

        vertical_offset = layer * 0.12

        opacity_level = 0.04 + (layer / wave_layers) * 0.35

        fig_signal.add_trace(
            go.Scatter(
                x=time_axis,
                y=wave + vertical_offset,
                mode="lines",
                line=dict(width=2),
                opacity=opacity_level,
                hoverinfo="skip",
                showlegend=False
            )
        )

    # --------------------------------------------------------
    # DOMINANT ENERGY ENVELOPE
    # --------------------------------------------------------

    envelope = (adjusted_instability * 1.2) * np.sin(time_axis * 0.7)

    fig_signal.add_trace(
        go.Scatter(
            x=time_axis,
            y=envelope,
            mode="lines",
            line=dict(width=5),
            opacity=0.85,
            hoverinfo="skip",
            showlegend=False
        )
    )

    # --------------------------------------------------------
    # CLEAN INTERACTION CONFIG
    # --------------------------------------------------------

    fig_signal.update_layout(
        height=480,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode="pan",
        showlegend=False
    )

    fig_signal.update_xaxes(
        visible=False,
        showgrid=False,
        zeroline=False
    )

    fig_signal.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False
    )

    st.plotly_chart(
        fig_signal,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
            "doubleClick": "reset"
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # TEMPORAL EVOLUTION ENGINE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Forward Structural Evolution</div>", unsafe_allow_html=True)

    projection_steps = 260
    simulation_count = 600
    base_capital = 1_000_000

    volatility_forward = 0.006 + adjusted_instability * 0.10
    drift_forward = 0.0002

    forward_bundle = []

    for _ in range(simulation_count):
        returns = np.random.normal(drift_forward, volatility_forward, projection_steps)
        path = base_capital * np.cumprod(1 + returns)
        forward_bundle.append(path)

    forward_bundle = np.array(forward_bundle)

    median_path = np.median(forward_bundle, axis=0)
    q25 = np.quantile(forward_bundle, 0.25, axis=0)
    q75 = np.quantile(forward_bundle, 0.75, axis=0)
    q05 = np.quantile(forward_bundle, 0.05, axis=0)
    q95 = np.quantile(forward_bundle, 0.95, axis=0)

    time_axis = np.arange(projection_steps)

    fig_evolution = go.Figure()

    # 5–95 envelope
    fig_evolution.add_trace(go.Scatter(
        x=time_axis,
        y=q95,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_evolution.add_trace(go.Scatter(
        x=time_axis,
        y=q05,
        mode='lines',
        fill='tonexty',
        opacity=0.12,
        showlegend=False
    ))

    # 25–75 envelope
    fig_evolution.add_trace(go.Scatter(
        x=time_axis,
        y=q75,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_evolution.add_trace(go.Scatter(
        x=time_axis,
        y=q25,
        mode='lines',
        fill='tonexty',
        opacity=0.22,
        showlegend=False
    ))

    # Median path
    fig_evolution.add_trace(go.Scatter(
        x=time_axis,
        y=median_path,
        mode='lines',
        line=dict(width=5),
        opacity=0.95,
        hovertemplate="Median Projection<extra></extra>",
        showlegend=False
    ))

    fig_evolution.update_layout(
        height=540,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode="pan",
        showlegend=False
    )

    fig_evolution.update_xaxes(visible=False)
    fig_evolution.update_yaxes(visible=False)

    st.plotly_chart(
        fig_evolution,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
            "doubleClick": "reset"
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # SYSTEMIC NETWORK GRAPH ENGINE
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Systemic Neural Network</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # DEFINE SYSTEM NODES
    # --------------------------------------------------------

    nodes = [
        "US Equities",
        "India Equities",
        "Volatility",
        "US Rates",
        "Gold",
        "High Yield Credit",
        "Investment Grade Credit"
    ]

    node_count = len(nodes)

    # Circular layout positions
    angles = np.linspace(0, 2*np.pi, node_count, endpoint=False)
    radius = 1.2

    x_nodes = radius * np.cos(angles)
    y_nodes = radius * np.sin(angles)

    # --------------------------------------------------------
    # EDGE WEIGHT GENERATION (INSTABILITY SCALED)
    # --------------------------------------------------------

    fig_network = go.Figure()

    for i in range(node_count):
        for j in range(i+1, node_count):

            connection_strength = adjusted_instability * np.random.uniform(0.2, 1.0)

            fig_network.add_trace(go.Scatter(
                x=[x_nodes[i], x_nodes[j]],
                y=[y_nodes[i], y_nodes[j]],
                mode="lines",
                line=dict(
                    width=1.5 + connection_strength * 4
                ),
                opacity=0.05 + connection_strength * 0.4,
                hoverinfo="skip",
                showlegend=False
            ))

    # --------------------------------------------------------
    # NODE VISUALS
    # --------------------------------------------------------

    node_sizes = 30 + adjusted_instability * 50

    fig_network.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            opacity=0.9
        ),
        text=nodes,
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ))

    fig_network.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode="pan",
        showlegend=False
    )

    fig_network.update_xaxes(visible=False)
    fig_network.update_yaxes(visible=False)

    st.plotly_chart(
        fig_network,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
            "doubleClick": "reset"
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # REGIME TRANSITION INTELLIGENCE ENGINE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Regime Transition Intelligence</div>", unsafe_allow_html=True)

    regimes = ["Stable", "Compression", "Elevated", "Fragile"]

    # Base transition probabilities
    base_matrix = np.array([
        [0.70, 0.20, 0.08, 0.02],
        [0.25, 0.50, 0.20, 0.05],
        [0.10, 0.30, 0.40, 0.20],
        [0.05, 0.15, 0.35, 0.45]
    ])

    # Instability-driven perturbation
    noise_scale = adjusted_instability * 0.25
    perturbation = np.random.uniform(-noise_scale, noise_scale, base_matrix.shape)

    transition_matrix = base_matrix + perturbation

    # Normalize rows
    transition_matrix = np.clip(transition_matrix, 0.001, None)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    # --------------------------------------------------------
    # HEATMAP VISUALIZATION
    # --------------------------------------------------------

    fig_regime = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=regimes,
        y=regimes,
        colorscale=[
            [0.0, "#ffffff"],
            [0.5, "#d1d5db"],
            [1.0, "#111111"]
        ],
        showscale=False
    ))

    fig_regime.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=40, r=40, t=10, b=10),
        dragmode=False
    )

    fig_regime.update_xaxes(showgrid=False)
    fig_regime.update_yaxes(showgrid=False)

    st.plotly_chart(
        fig_regime,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )

    # --------------------------------------------------------
    # CURRENT REGIME PROBABILITY PROJECTION
    # --------------------------------------------------------

    current_regime = results["regime"]

    regime_index = regimes.index(current_regime)

    future_probabilities = transition_matrix[regime_index]

    prob_cols = st.columns(4)

    for i, regime_name in enumerate(regimes):
        prob_cols[i].metric(
            regime_name,
            round(float(future_probabilities[i]), 3)
        )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # TAIL RISK & CONVEXITY ENGINE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Tail Risk & Convexity Engine</div>", unsafe_allow_html=True)

    projection_steps = 200
    simulations_tail = 1200
    base_capital = 1_000_000

    volatility_tail = 0.007 + adjusted_instability * 0.12
    drift_tail = 0.00015

    tail_paths = []

    for _ in range(simulations_tail):
        returns = np.random.normal(drift_tail, volatility_tail, projection_steps)
        path = base_capital * np.cumprod(1 + returns)
        tail_paths.append(path)

    tail_paths = np.array(tail_paths)

    final_values = tail_paths[:, -1]

    # --------------------------------------------------------
    # VaR & CVaR CALCULATION
    # --------------------------------------------------------

    var_95 = np.percentile(final_values, 5)
    var_99 = np.percentile(final_values, 1)

    cvar_95 = final_values[final_values <= var_95].mean()
    cvar_99 = final_values[final_values <= var_99].mean()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric("VaR 95%", round(float(var_95), 2))
    metric_col2.metric("CVaR 95%", round(float(cvar_95), 2))
    metric_col3.metric("VaR 99%", round(float(var_99), 2))
    metric_col4.metric("CVaR 99%", round(float(cvar_99), 2))

    # --------------------------------------------------------
    # DISTRIBUTION DENSITY PLOT
    # --------------------------------------------------------

    hist_values, hist_bins = np.histogram(final_values, bins=80, density=True)

    fig_tail = go.Figure()

    fig_tail.add_trace(
        go.Bar(
            x=hist_bins[:-1],
            y=hist_values,
            opacity=0.6,
            showlegend=False
        )
    )

    fig_tail.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0)
    )

    fig_tail.update_xaxes(visible=False)
    fig_tail.update_yaxes(visible=False)

    st.plotly_chart(
        fig_tail,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    # --------------------------------------------------------
    # CONVEXITY RESPONSE CURVE
    # --------------------------------------------------------

    shock_range = np.linspace(0, 1.0, 60)

    convexity_curve = []

    for shock_level in shock_range:
        adj_vol = 0.007 + (adjusted_instability + shock_level * 0.6) * 0.12
        simulated = []

        for _ in range(200):
            returns = np.random.normal(drift_tail, adj_vol, projection_steps)
            path = base_capital * np.cumprod(1 + returns)
            simulated.append(path[-1])

        simulated = np.array(simulated)
        convexity_curve.append(np.percentile(simulated, 5))

    fig_convexity = go.Figure()

    fig_convexity.add_trace(
        go.Scatter(
            x=shock_range,
            y=convexity_curve,
            mode="lines",
            line=dict(width=4),
            showlegend=False
        )
    )

    fig_convexity.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0)
    )

    fig_convexity.update_xaxes(visible=False)
    fig_convexity.update_yaxes(visible=False)

    st.plotly_chart(
        fig_convexity,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # GLOBAL STRESS GRID & CONTAGION SURFACE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Global Stress & Contagion Grid</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # DEFINE REGIONS
    # --------------------------------------------------------

    regions = [
        "United States",
        "India",
        "Europe",
        "Asia-Pacific",
        "Commodities"
    ]

    region_count = len(regions)

    # Base structural fragility per region
    base_fragility = np.array([0.35, 0.30, 0.32, 0.33, 0.28])

    # Instability-driven perturbation
    region_noise = np.random.uniform(-0.1, 0.1, region_count)
    regional_fragility = np.clip(
        base_fragility + adjusted_instability * 0.5 + region_noise * adjusted_instability,
        0.05,
        1.0
    )

    # --------------------------------------------------------
    # DOMINANT STRESS REGION
    # --------------------------------------------------------

    dominant_index = np.argmax(regional_fragility)
    dominant_region = regions[dominant_index]

    region_cols = st.columns(region_count)

    for i in range(region_count):
        region_cols[i].metric(
            regions[i],
            round(float(regional_fragility[i]), 3)
        )

    st.markdown(
        f"<div style='text-align:center; font-size:20px; margin-top:20px;'>"
        f"Dominant Stress Region: <strong>{dominant_region}</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CONTAGION DIFFUSION SURFACE
    # --------------------------------------------------------

    grid_size = 50

    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)

    contagion_intensity = 0

    for i in range(region_count):
        center_x = np.cos(i * 2*np.pi/region_count)
        center_y = np.sin(i * 2*np.pi/region_count)

        contagion_intensity += (
            regional_fragility[i] *
            np.exp(-((X - center_x)**2 + (Y - center_y)**2))
        )

    contagion_intensity *= adjusted_instability

    fig_contagion = go.Figure(
        data=[
            go.Surface(
                z=contagion_intensity,
                x=X,
                y=Y,
                showscale=False,
                opacity=0.9
            )
        ]
    )

    fig_contagion.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    st.plotly_chart(
        fig_contagion,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # STRUCTURAL MEMORY & REGIME REPLAY ENGINE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Structural Memory & Regime Replay</div>", unsafe_allow_html=True)

    memory_steps = 320

    # Generate synthetic historical instability curve
    time_memory = np.arange(memory_steps)

    memory_instability = (
        0.3 +
        0.2 * np.sin(time_memory * 0.04) +
        0.15 * np.sin(time_memory * 0.12 + 2) +
        0.1 * np.random.normal(0, 0.5, memory_steps)
    )

    memory_instability = np.clip(memory_instability, 0.05, 1.0)

    # --------------------------------------------------------
    # REPLAY CONTROL
    # --------------------------------------------------------

    replay_position = st.slider(
        "Replay Timeline Position",
        min_value=0,
        max_value=memory_steps - 1,
        value=memory_steps - 1
    )

    current_memory_instability = memory_instability[replay_position]

    # --------------------------------------------------------
    # REGIME CLASSIFICATION OVER TIME
    # --------------------------------------------------------

    memory_regimes = []

    for val in memory_instability:
        if val < 0.25:
            memory_regimes.append(0)
        elif val < 0.45:
            memory_regimes.append(1)
        elif val < 0.7:
            memory_regimes.append(2)
        else:
            memory_regimes.append(3)

    # --------------------------------------------------------
    # VISUALIZE MEMORY EVOLUTION
    # --------------------------------------------------------

    fig_memory = go.Figure()

    fig_memory.add_trace(
        go.Scatter(
            x=time_memory,
            y=memory_instability,
            mode="lines",
            line=dict(width=4),
            opacity=0.9,
            showlegend=False
        )
    )

    fig_memory.add_trace(
        go.Scatter(
            x=[replay_position],
            y=[current_memory_instability],
            mode="markers",
            marker=dict(size=16),
            showlegend=False
        )
    )

    fig_memory.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0)
    )

    fig_memory.update_xaxes(visible=False)
    fig_memory.update_yaxes(visible=False)

    st.plotly_chart(
        fig_memory,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    # --------------------------------------------------------
    # CURRENT REPLAY REGIME DISPLAY
    # --------------------------------------------------------

    regime_labels = ["Stable", "Compression", "Elevated", "Fragile"]
    current_replay_regime = regime_labels[memory_regimes[replay_position]]

    st.markdown(
        f"<div style='text-align:center; font-size:22px; margin-top:20px;'>"
        f"Replay Regime: <strong>{current_replay_regime}</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # ADAPTIVE COGNITIVE EXPLANATION ENGINE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>System Intelligence Narrative</div>", unsafe_allow_html=True)

    explanation_mode = st.radio(
        "Explanation Mode",
        ["Simple Explanation", "Technical Explanation"],
        horizontal=True
    )

    # --------------------------------------------------------
    # SYSTEM STATE CONTEXT
    # --------------------------------------------------------

    system_state = results["regime"]
    instability_value = round(adjusted_instability, 3)

    # --------------------------------------------------------
    # SIMPLE MODE
    # --------------------------------------------------------

    if explanation_mode == "Simple Explanation":

        if instability_value < 0.25:
            explanation_text = """
            The financial system is currently calm.
            Asset movements are not strongly connected.
            Risk of sudden shock spreading is low.
            Capital survival probability is strong.
            """

        elif instability_value < 0.5:
            explanation_text = """
            The system is showing signs of tension.
            Markets are becoming more connected.
            Risk levels are rising but not critical.
            Monitoring is recommended.
            """

        elif instability_value < 0.75:
            explanation_text = """
            The system is under stress.
            Market movements are tightly linked.
            Shock transmission risk is high.
            Capital survival probability is weakening.
            """

        else:
            explanation_text = """
            The system is highly unstable.
            Shock contagion is very likely.
            Market regimes may shift rapidly.
            Downside capital risk is elevated.
            """

        st.markdown(
            f"<div style='font-size:18px; line-height:1.7; text-align:center; max-width:900px; margin:auto;'>"
            f"{explanation_text}"
            f"</div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # TECHNICAL MODE
    # --------------------------------------------------------

    else:

        technical_text = f"""
        Current Structural Regime: {system_state}

        Global Instability Index: {instability_value}

        The system exhibits volatility clustering and
        cross-asset correlation expansion consistent with
        transitional regime dynamics.

        Tail risk distribution widening observed through
        elevated conditional value-at-risk (CVaR) metrics.

        Network density scaling with instability suggests
        increased contagion transmission probability.

        Regime transition matrix indicates non-trivial
        probability of migration toward elevated or fragile states.
        """

        st.markdown(
            f"<div style='font-size:17px; line-height:1.7; text-align:center; max-width:900px; margin:auto;'>"
            f"{technical_text}"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # ADAPTIVE THEMATIC MORPH ENGINE 2.0
    # ========================================================

    # Environmental scaling
    env_intensity = adjusted_instability

    # Temperature mapping (subtle shifts)
    if env_intensity < 0.25:
        bg_top = "#ffffff"
        bg_mid = "#f6f7f9"
        bg_bottom = "#eef1f4"
        text_color = "#111111"
    elif env_intensity < 0.5:
        bg_top = "#fcfcfd"
        bg_mid = "#f1f3f6"
        bg_bottom = "#e6eaf0"
        text_color = "#1c1c1e"
    elif env_intensity < 0.75:
        bg_top = "#f9fafc"
        bg_mid = "#eceff4"
        bg_bottom = "#dde3ea"
        text_color = "#202124"
    else:
        bg_top = "#f6f7f9"
        bg_mid = "#e8ecf1"
        bg_bottom = "#d8dee7"
        text_color = "#1a1a1a"

    # Dynamic CSS injection
    st.markdown(f"""
    <style>

    html, body {{
        background: linear-gradient(
            180deg,
            {bg_top} 0%,
            {bg_mid} 45%,
            {bg_bottom} 100%
        );
        color: {text_color};
        transition: background 0.8s ease, color 0.8s ease;
    }}

    .floating-layer {{
        background: rgba(255,255,255,{0.55 + (1-env_intensity)*0.25});
        backdrop-filter: blur({18 + env_intensity*6}px);
        transition: all 0.8s ease;
    }}

    .hero-title {{
        opacity: {1 - env_intensity*0.08};
        transition: opacity 0.6s ease;
    }}

    @keyframes adaptiveFloat {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-{6 + env_intensity*6}px); }}
        100% {{ transform: translateY(0px); }}
    }}

    .floating-layer {{
        animation: adaptiveFloat {16 - env_intensity*6}s ease-in-out infinite;
    }}

    </style>
    """, unsafe_allow_html=True)
    # ========================================================
    # PERFORMANCE OPTIMIZATION & SMART SAMPLING ENGINE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Adaptive Performance Engine</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # DYNAMIC SIMULATION SCALING
    # --------------------------------------------------------

    if adjusted_instability < 0.25:
        adaptive_simulations = 400
        adaptive_paths = 40
        resolution_factor = 0.6
    elif adjusted_instability < 0.5:
        adaptive_simulations = 700
        adaptive_paths = 60
        resolution_factor = 0.8
    elif adjusted_instability < 0.75:
        adaptive_simulations = 1000
        adaptive_paths = 80
        resolution_factor = 1.0
    else:
        adaptive_simulations = 1400
        adaptive_paths = 110
        resolution_factor = 1.2

    # --------------------------------------------------------
    # COMPUTATION CACHING LAYER
    # --------------------------------------------------------

    @st.cache_data(show_spinner=False)
    def generate_cached_simulation(sim_count, vol, drift, steps):
        bundle = []
        for _ in range(sim_count):
            returns = np.random.normal(drift, vol, steps)
            path = 1_000_000 * np.cumprod(1 + returns)
            bundle.append(path)
        return np.array(bundle)

    cached_bundle = generate_cached_simulation(
        adaptive_simulations,
        0.006 + adjusted_instability * 0.10,
        0.0002,
        int(200 * resolution_factor)
    )

    # --------------------------------------------------------
    # LOAD INTELLIGENCE METRICS
    # --------------------------------------------------------

    performance_col1, performance_col2, performance_col3 = st.columns(3)

    performance_col1.metric("Adaptive Simulations", adaptive_simulations)
    performance_col2.metric("Rendering Paths", adaptive_paths)
    performance_col3.metric("Resolution Scale", round(resolution_factor, 2))

    # --------------------------------------------------------
    # VISUAL PERFORMANCE INDICATOR
    # --------------------------------------------------------

    perf_bar = np.linspace(0, 1, 50)
    perf_curve = resolution_factor * np.sin(perf_bar * np.pi)

    fig_perf = go.Figure()

    fig_perf.add_trace(
        go.Scatter(
            x=perf_bar,
            y=perf_curve,
            mode="lines",
            line=dict(width=4),
            opacity=0.85,
            showlegend=False
        )
    )

    fig_perf.update_layout(
        height=300,
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0)
    )

    fig_perf.update_xaxes(visible=False)
    fig_perf.update_yaxes(visible=False)

    st.plotly_chart(
        fig_perf,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # AUTONOMOUS INSIGHT ENGINE 2.0
    # ========================================================

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Autonomous Strategic Intelligence</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SYSTEM-WIDE RISK SCORING
    # --------------------------------------------------------

    regime_score_map = {
        "Stable": 0.2,
        "Compression": 0.4,
        "Elevated": 0.7,
        "Fragile": 0.9
    }

    regime_score = regime_score_map.get(results["regime"], 0.5)

    tail_score = 1 - (var_95 / 1_000_000)
    network_score = adjusted_instability

    composite_risk_score = (
        0.4 * adjusted_instability +
        0.25 * regime_score +
        0.2 * tail_score +
        0.15 * network_score
    )

    composite_risk_score = np.clip(composite_risk_score, 0, 1)

    # --------------------------------------------------------
    # STRATEGIC POSTURE CLASSIFICATION
    # --------------------------------------------------------

    if composite_risk_score < 0.3:
        posture = "Aggressive Expansion"
        recommendation = """
        Risk environment supportive.
        Capital deployment may be increased.
        Exposure scaling acceptable.
        Diversification pressure low.
        """
    elif composite_risk_score < 0.55:
        posture = "Balanced Allocation"
        recommendation = """
        Moderate structural tension.
        Maintain diversification discipline.
        Avoid excessive leverage.
        Monitor regime transition probability.
        """
    elif composite_risk_score < 0.75:
        posture = "Defensive Rotation"
        recommendation = """
        Elevated fragility detected.
        Increase capital protection bias.
        Reduce high-beta exposure.
        Strengthen liquidity positioning.
        """
    else:
        posture = "Capital Preservation Mode"
        recommendation = """
        Systemic instability high.
        Prioritize liquidity.
        Minimize convex downside exposure.
        Prepare for regime shift.
        """

    # --------------------------------------------------------
    # DISPLAY STRATEGIC SUMMARY
    # --------------------------------------------------------

    col_s1, col_s2 = st.columns([1, 2])

    with col_s1:
        st.metric("Composite Risk Score", round(float(composite_risk_score), 3))
        st.metric("Strategic Posture", posture)

    with col_s2:
        st.markdown(
            f"<div style='font-size:18px; line-height:1.8;'>"
            f"{recommendation}"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
    # ========================================================
    # SYSTEM COHERENCE & POLISHING ENGINE 3.0
    # ========================================================

    # --------------------------------------------------------
    # GLOBAL SPACING NORMALIZATION
    # --------------------------------------------------------

    st.markdown("""
    <style>

    /* Remove extra top gaps */
    .block-container > div {
        margin-top: 0 !important;
    }

    /* Uniform vertical rhythm */
    .floating-layer {
        margin-top: 100px !important;
        margin-bottom: 100px !important;
        padding-top: 70px !important;
        padding-bottom: 70px !important;
    }

    /* Typography refinement */
    .section-title {
        letter-spacing: -0.5px;
        font-weight: 600;
        margin-bottom: 50px !important;
    }

    /* Smooth all animations */
    * {
        transition: all 0.6s ease;
    }

    /* Refine metric appearance */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.55);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(12px);
    }

    /* Remove sharp edges from plotly */
    .js-plotly-plot .plotly {
        border-radius: 30px;
    }

    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # MOTION HARMONY CONTROL
    # --------------------------------------------------------

    harmony_speed = 18 - adjusted_instability * 8

    st.markdown(f"""
    <style>

    @keyframes globalFloat {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-6px); }}
        100% {{ transform: translateY(0px); }}
    }}

    .floating-layer {{
        animation: globalFloat {harmony_speed}s ease-in-out infinite;
    }}

    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # FINAL SYSTEM STATUS BAR
    # --------------------------------------------------------

    st.markdown("<div class='floating-layer'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>System Status Overview</div>",
        unsafe_allow_html=True
    )

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    status_col1.metric("Instability Index", round(float(adjusted_instability), 3))
    status_col2.metric("Current Regime", results["regime"])
    status_col3.metric("Composite Risk", round(float(composite_risk_score), 3))
    status_col4.metric("Strategic Posture", posture)

    st.markdown("</div>", unsafe_allow_html=True)
