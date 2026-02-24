# ============================================================
# EIGO v5 — FOUNDATIONAL ARCHITECTURE LAYER
# ============================================================

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
# REMOVE STREAMLIT DEFAULT UI ARTIFACTS
# ============================================================

st.markdown("""
<style>

/* Remove Streamlit branding and default elements */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Global reset */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #ffffff;
}

/* Main container spacing system */
.block-container {
    padding-top: 5rem;
    padding-left: 10%;
    padding-right: 10%;
    padding-bottom: 6rem;
}

/* Design Tokens */
:root {
    --color-bg: #ffffff;
    --color-soft: #f5f5f7;
    --color-text: #111111;
    --color-muted: #6e6e73;
    --radius-large: 28px;
    --radius-medium: 20px;
    --radius-small: 14px;
}

/* Typography System */
.hero-title {
    font-size: 64px;
    font-weight: 600;
    letter-spacing: -2px;
    text-align: center;
    color: var(--color-text);
}

.hero-subtitle {
    font-size: 22px;
    text-align: center;
    color: var(--color-muted);
    margin-top: -12px;
}

.hero-description {
    font-size: 16px;
    color: var(--color-muted);
    text-align: center;
    max-width: 780px;
    margin: auto;
    margin-top: 24px;
    line-height: 1.6;
}

/* Glass container system */
.glass-panel {
    background: var(--color-soft);
    padding: 40px;
    border-radius: var(--radius-large);
    margin-top: 40px;
    transition: 0.4s ease;
}

.glass-panel:hover {
    transform: translateY(-2px);
}

/* Divider */
.soft-divider {
    height: 1px;
    background: #eaeaea;
    margin-top: 70px;
    margin-bottom: 70px;
}

/* Button Styling */
div.stButton > button {
    background-color: #111111;
    color: white;
    border-radius: 999px;
    padding: 12px 28px;
    border: none;
    font-weight: 500;
    transition: 0.3s ease;
}

div.stButton > button:hover {
    background-color: #333333;
    transform: scale(1.02);
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
A structural intelligence engine modeling systemic fragility,
contagion propagation, and probabilistic capital survival
through adaptive financial simulation.
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

# ============================================================
# CONTROL LAYER CONTAINER
# ============================================================

st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])

with col1:
    shock = st.slider(
        "Fragility Amplifier",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )

with col2:
    run_system = st.button("Run System")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
# ============================================================
# EXECUTION ENGINE
# ============================================================

if run_system:

    with st.spinner("Mapping structural financial topology..."):

        results = run_engine()
        adjusted_instability = min(1.0, results["instability"] + shock)
        twin = digital_twin(adjusted_instability)

    # ========================================================
    # INSTABILITY CORE VISUAL SYSTEM
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        System Instability Core
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic radius calculation
    base_size = 180
    dynamic_size = base_size + (adjusted_instability * 120)

    # Color shift logic
    if adjusted_instability < 0.25:
        core_color = "rgba(34,197,94,0.6)"
    elif adjusted_instability < 0.5:
        core_color = "rgba(234,179,8,0.6)"
    elif adjusted_instability < 0.75:
        core_color = "rgba(249,115,22,0.6)"
    else:
        core_color = "rgba(239,68,68,0.6)"

    # Animated pulse layers
    st.markdown(f"""
    <div style="
        position:relative;
        width:100%;
        display:flex;
        justify-content:center;
        align-items:center;
        margin-top:40px;
        margin-bottom:40px;
    ">
        <div style="
            width:{dynamic_size}px;
            height:{dynamic_size}px;
            border-radius:50%;
            background:{core_color};
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:36px;
            font-weight:600;
            color:#111;
            animation: pulse 3s infinite ease-in-out;
        ">
            {round(adjusted_instability,3)}
        </div>
    </div>

    <style>
    @keyframes pulse {{
        0% {{
            box-shadow: 0 0 0 0 rgba(0,0,0,0.15);
        }}
        50% {{
            box-shadow: 0 0 40px 20px rgba(0,0,0,0.05);
        }}
        100% {{
            box-shadow: 0 0 0 0 rgba(0,0,0,0.15);
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # ========================================================
    # CORE METRIC STRUCTURE
    # ========================================================

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:14px;color:#888;'>Regime State</div>
            <div style='font-size:22px;font-weight:600;margin-top:6px;'>
                {results["regime"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:14px;color:#888;'>90% Survival Probability</div>
            <div style='font-size:22px;font-weight:600;margin-top:6px;'>
                {twin["survival_90"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:14px;color:#888;'>80% Survival Probability</div>
            <div style='font-size:22px;font-weight:600;margin-top:6px;'>
                {twin["survival_80"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # NEURAL CAPITAL FLOW ENGINE
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        Capital Neural Flow
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SIMULATED MULTI-LAYER CAPITAL TRAJECTORIES
    # --------------------------------------------------------

    num_paths = 40
    time_steps = 160
    base_capital = 1_000_000

    volatility_scale = 0.008 + adjusted_instability * 0.06
    drift_factor = 0.0003

    fig_flow = go.Figure()

    for i in range(num_paths):

        noise = np.random.normal(drift_factor, volatility_scale, time_steps)
        path = base_capital * np.cumprod(1 + noise)

        normalized_path = (path - path.min()) / (path.max() - path.min())

        opacity_layer = 0.08 + (i / num_paths) * 0.25

        fig_flow.add_trace(
            go.Scatter(
                x=np.arange(time_steps),
                y=normalized_path,
                mode="lines",
                line=dict(width=1.2),
                opacity=opacity_layer,
                showlegend=False
            )
        )

    # --------------------------------------------------------
    # MEAN CAPITAL TRAJECTORY
    # --------------------------------------------------------

    mean_paths = []

    for _ in range(120):
        noise = np.random.normal(drift_factor, volatility_scale, time_steps)
        path = base_capital * np.cumprod(1 + noise)
        mean_paths.append(path)

    mean_paths = np.array(mean_paths)
    mean_curve = mean_paths.mean(axis=0)
    mean_normalized = (mean_curve - mean_curve.min()) / (mean_curve.max() - mean_curve.min())

    fig_flow.add_trace(
        go.Scatter(
            x=np.arange(time_steps),
            y=mean_normalized,
            mode="lines",
            line=dict(width=4),
            opacity=0.9,
            showlegend=False
        )
    )

    # --------------------------------------------------------
    # VISUAL CONFIGURATION
    # --------------------------------------------------------

    fig_flow.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_flow, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # STRUCTURAL SIGNAL MATRIX
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        Structural Signal Matrix
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MULTI-LAYER OSCILLATION FIELD
    # --------------------------------------------------------

    time_axis = np.linspace(0, 6 * np.pi, 600)

    fig_signal = go.Figure()

    # Generate multiple interacting wave layers
    wave_layers = 12

    for layer in range(wave_layers):

        frequency = 0.6 + (layer * 0.15)
        amplitude = 0.1 + (adjusted_instability * 0.8)
        phase_shift = layer * 0.4

        wave = amplitude * np.sin(time_axis * frequency + phase_shift)

        vertical_offset = layer * 0.12

        fig_signal.add_trace(
            go.Scatter(
                x=time_axis,
                y=wave + vertical_offset,
                mode="lines",
                line=dict(width=2),
                opacity=0.25 + (layer / wave_layers) * 0.4,
                showlegend=False
            )
        )

    # --------------------------------------------------------
    # SIGNAL INTENSITY ENVELOPE
    # --------------------------------------------------------

    envelope = (adjusted_instability * 0.9) * np.sin(time_axis * 0.8)

    fig_signal.add_trace(
        go.Scatter(
            x=time_axis,
            y=envelope,
            mode="lines",
            line=dict(width=5),
            opacity=0.9,
            showlegend=False
        )
    )

    # --------------------------------------------------------
    # VISUAL CONFIG
    # --------------------------------------------------------

    fig_signal.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_signal, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # TEMPORAL EVOLUTION FIELD
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        Temporal Evolution Projection
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MULTI-FUTURE PROJECTION SYSTEM
    # --------------------------------------------------------

    projection_steps = 180
    projection_paths = 60
    base_capital = 1_000_000

    volatility_projection = 0.006 + adjusted_instability * 0.07
    drift_projection = 0.00025

    fig_future = go.Figure()

    for i in range(projection_paths):

        returns = np.random.normal(
            drift_projection,
            volatility_projection,
            projection_steps
        )

        path = base_capital * np.cumprod(1 + returns)

        normalized = (path - path.min()) / (path.max() - path.min())

        opacity_level = 0.05 + (i / projection_paths) * 0.3

        fig_future.add_trace(
            go.Scatter(
                x=np.arange(projection_steps),
                y=normalized,
                mode="lines",
                line=dict(width=1),
                opacity=opacity_level,
                showlegend=False
            )
        )

    # --------------------------------------------------------
    # MEDIAN TRAJECTORY (STRUCTURAL CORE)
    # --------------------------------------------------------

    simulation_bundle = []

    for _ in range(250):
        returns = np.random.normal(
            drift_projection,
            volatility_projection,
            projection_steps
        )
        path = base_capital * np.cumprod(1 + returns)
        simulation_bundle.append(path)

    simulation_bundle = np.array(simulation_bundle)

    median_path = np.median(simulation_bundle, axis=0)
    median_normalized = (median_path - median_path.min()) / (median_path.max() - median_path.min())

    fig_future.add_trace(
        go.Scatter(
            x=np.arange(projection_steps),
            y=median_normalized,
            mode="lines",
            line=dict(width=5),
            opacity=1.0,
            showlegend=False
        )
    )

    # --------------------------------------------------------
    # VISUAL CONFIGURATION
    # --------------------------------------------------------

    fig_future.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_future, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # ADAPTIVE VISUAL ENVIRONMENT
    # ========================================================

    # Environmental intensity scaling
    instability_factor = adjusted_instability

    # Dynamic background softness
    if instability_factor < 0.25:
        bg_color = "#ffffff"
        panel_color = "#f5f5f7"
        accent_opacity = 0.08
    elif instability_factor < 0.5:
        bg_color = "#fcfcfc"
        panel_color = "#f2f2f2"
        accent_opacity = 0.12
    elif instability_factor < 0.75:
        bg_color = "#f9f9f9"
        panel_color = "#eeeeee"
        accent_opacity = 0.18
    else:
        bg_color = "#f4f4f4"
        panel_color = "#e9e9e9"
        accent_opacity = 0.25

    st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        transition: background-color 0.6s ease;
    }}

    .glass-panel {{
        background: {panel_color};
        transition: background 0.6s ease;
    }}

    .hero-title {{
        opacity: {1 - instability_factor * 0.1};
        transition: opacity 0.6s ease;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ========================================================
    # STRUCTURAL DEPTH FIELD (SUBTLE BACKGROUND MOTION ILLUSION)
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:26px;font-weight:600;'>
        Structural Depth Field
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    depth_time = np.linspace(0, 3 * np.pi, 500)

    fig_depth = go.Figure()

    depth_layers = 8

    for layer in range(depth_layers):

        frequency = 0.5 + (layer * 0.25)
        amplitude = 0.05 + (instability_factor * 0.5)

        wave = amplitude * np.sin(depth_time * frequency + layer)

        vertical_offset = layer * 0.08

        fig_depth.add_trace(
            go.Scatter(
                x=depth_time,
                y=wave + vertical_offset,
                mode="lines",
                line=dict(width=2),
                opacity=accent_opacity + (layer / depth_layers) * 0.3,
                showlegend=False
            )
        )

    fig_depth.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor=bg_color
    )

    st.plotly_chart(fig_depth, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # COGNITIVE INTERPRETATION ENGINE
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        System Intelligence Report
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # STRUCTURAL STATE CLASSIFICATION
    # --------------------------------------------------------

    instability = adjusted_instability

    if instability < 0.20:
        state_label = "Structural Stability"
        macro_description = """
        The financial system is operating within structurally stable parameters.
        Cross-asset correlations remain moderate.
        Volatility dispersion is controlled.
        Capital allocation efficiency is high.
        """
        contagion_analysis = """
        Contagion pathways are weak.
        Systemic shock propagation probability remains low.
        Network density does not indicate clustering stress.
        """
        capital_impact = """
        Forward capital survival probabilities remain resilient.
        Tail risk exposure is minimal.
        """

    elif instability < 0.40:
        state_label = "Compression Phase"
        macro_description = """
        Asset class correlations are compressing.
        Volatility clustering signals emerging tension.
        Structural coupling across credit and equity increasing.
        """
        contagion_analysis = """
        Contagion nodes forming across financial layers.
        Shock transmission probability increasing.
        Network coherence rising.
        """
        capital_impact = """
        Capital survival remains strong but sensitivity rising.
        Portfolio convexity exposure should be monitored.
        """

    elif instability < 0.65:
        state_label = "Elevated Fragility"
        macro_description = """
        Systemic fragility increasing.
        Correlation breakdown likely.
        Credit spreads widening structurally.
        """
        contagion_analysis = """
        Contagion network density elevated.
        Shock amplification mechanisms active.
        Cross-asset stress transmission accelerating.
        """
        capital_impact = """
        Forward survival dispersion widening.
        Median capital path volatility rising.
        Downside convexity exposure increasing.
        """

    else:
        state_label = "Systemic Instability"
        macro_description = """
        High systemic instability detected.
        Structural regime transition probability elevated.
        Correlation regime shift underway.
        """
        contagion_analysis = """
        Network contagion highly probable.
        Shock propagation velocity increasing.
        Structural stress amplification dominant.
        """
        capital_impact = """
        Capital preservation risk elevated.
        Tail scenarios intensifying.
        Forward dispersion wide and unstable.
        """

    # --------------------------------------------------------
    # DISPLAY COGNITIVE BLOCK
    # --------------------------------------------------------

    st.markdown(f"""
    <div style="
        font-size:22px;
        font-weight:600;
        text-align:center;
        margin-bottom:25px;
    ">
        {state_label}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:20px;font-size:16px;color:#555;">
        <strong>Macro Structural Assessment:</strong><br>
        {macro_description}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:20px;font-size:16px;color:#555;">
        <strong>Contagion Network Analysis:</strong><br>
        {contagion_analysis}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:20px;font-size:16px;color:#555;">
        <strong>Capital Impact Projection:</strong><br>
        {capital_impact}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # REGIME TRANSITION PROBABILITY MATRIX
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        Regime Transition Matrix
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # DEFINE REGIME STATES
    # --------------------------------------------------------

    regimes = [
        "Stable",
        "Compression",
        "Elevated",
        "Fragile"
    ]

    # Construct probabilistic transition matrix
    base_transition = np.array([
        [0.70, 0.20, 0.08, 0.02],
        [0.25, 0.50, 0.20, 0.05],
        [0.10, 0.30, 0.40, 0.20],
        [0.05, 0.15, 0.35, 0.45]
    ])

    # Instability-driven scaling
    scaling = adjusted_instability * 0.4

    transition_matrix = base_transition.copy()

    transition_matrix = transition_matrix + scaling * np.random.uniform(-0.05, 0.05, base_transition.shape)

    # Normalize rows
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    # --------------------------------------------------------
    # HEATMAP VISUALIZATION
    # --------------------------------------------------------

    fig_matrix = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=regimes,
        y=regimes,
        colorscale="Greys",
        showscale=False
    ))

    fig_matrix.update_layout(
        height=420,
        margin=dict(l=40, r=40, t=20, b=20),
        template="plotly_white"
    )

    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # STRESS INTERACTION HEAT SURFACE
    # ========================================================

    st.markdown("""
    <div style='text-align:center;font-size:26px;font-weight:600;'>
        Systemic Stress Interaction Surface
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    x_axis = np.linspace(0, 1, 50)
    y_axis = np.linspace(0, 1, 50)

    X, Y = np.meshgrid(x_axis, y_axis)

    Z = (
        np.sin(X * np.pi * (1 + adjusted_instability * 2)) *
        np.cos(Y * np.pi * (1 + adjusted_instability * 2)) *
        adjusted_instability
    )

    fig_surface = go.Figure(data=[go.Surface(
        z=Z,
        x=X,
        y=Y,
        colorscale="Greys",
        showscale=False,
        opacity=0.85
    )])

    fig_surface.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        template="plotly_white"
    )

    st.plotly_chart(fig_surface, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # CAPITAL SURVIVAL DENSITY BANDS
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        Capital Survival Density Bands
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    simulation_runs = 500
    projection_steps = 180
    base_capital = 1_000_000

    vol_tail = 0.007 + adjusted_instability * 0.08
    drift_tail = 0.0002

    simulated_bundle = []

    for _ in range(simulation_runs):
        returns = np.random.normal(drift_tail, vol_tail, projection_steps)
        path = base_capital * np.cumprod(1 + returns)
        simulated_bundle.append(path)

    simulated_bundle = np.array(simulated_bundle)

    median_path = np.median(simulated_bundle, axis=0)
    q10 = np.quantile(simulated_bundle, 0.10, axis=0)
    q25 = np.quantile(simulated_bundle, 0.25, axis=0)
    q75 = np.quantile(simulated_bundle, 0.75, axis=0)
    q90 = np.quantile(simulated_bundle, 0.90, axis=0)

    time_axis = np.arange(projection_steps)

    fig_density = go.Figure()

    # 90% envelope
    fig_density.add_trace(go.Scatter(
        x=time_axis,
        y=q90,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_density.add_trace(go.Scatter(
        x=time_axis,
        y=q10,
        mode='lines',
        fill='tonexty',
        line=dict(width=0),
        opacity=0.15,
        name='10-90%'
    ))

    # 50% envelope
    fig_density.add_trace(go.Scatter(
        x=time_axis,
        y=q75,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_density.add_trace(go.Scatter(
        x=time_axis,
        y=q25,
        mode='lines',
        fill='tonexty',
        line=dict(width=0),
        opacity=0.25,
        name='25-75%'
    ))

    # Median path
    fig_density.add_trace(go.Scatter(
        x=time_axis,
        y=median_path,
        mode='lines',
        line=dict(width=4),
        name='Median'
    ))

    fig_density.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    st.plotly_chart(fig_density, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # TAIL RISK METRICS
    # ========================================================

    final_values = simulated_bundle[:, -1]

    var_95 = np.percentile(final_values, 5)
    cvar_95 = final_values[final_values <= var_95].mean()

    st.markdown("""
    <div style='text-align:center;font-size:24px;font-weight:600;'>
        Tail Risk Envelope Metrics
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)

    col_t1.metric("Value at Risk (5%)", round(var_95, 2))
    col_t2.metric("Conditional VaR (Expected Shortfall)", round(cvar_95, 2))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    # ========================================================
    # SCENARIO STRESS SIMULATION ENGINE
    # ========================================================

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;font-size:28px;font-weight:600;'>
        Autonomous Stress Scenario Engine
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SCENARIO CONTROLS
    # --------------------------------------------------------

    scenario_col1, scenario_col2 = st.columns(2)

    with scenario_col1:
        macro_shock = st.slider(
            "Macro Shock Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="macro_shock"
        )

    with scenario_col2:
        contagion_amplifier = st.slider(
            "Contagion Amplifier",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="contagion_amp"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # BASELINE VS STRESSED COMPARISON
    # --------------------------------------------------------

    base_instability = adjusted_instability
    stressed_instability = min(
        1.0,
        adjusted_instability
        + macro_shock * 0.4
        + contagion_amplifier * 0.4
    )

    projection_steps = 180
    simulation_runs = 400
    base_capital = 1_000_000

    def simulate_paths(instability_level):

        vol = 0.007 + instability_level * 0.09
        drift = 0.0002

        bundle = []

        for _ in range(simulation_runs):
            returns = np.random.normal(drift, vol, projection_steps)
            path = base_capital * np.cumprod(1 + returns)
            bundle.append(path)

        return np.array(bundle)

    baseline_bundle = simulate_paths(base_instability)
    stressed_bundle = simulate_paths(stressed_instability)

    median_base = np.median(baseline_bundle, axis=0)
    median_stress = np.median(stressed_bundle, axis=0)

    # --------------------------------------------------------
    # COMPARATIVE ENVELOPE VISUALIZATION
    # --------------------------------------------------------

    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=np.arange(projection_steps),
        y=median_base,
        mode='lines',
        line=dict(width=4),
        name="Baseline"
    ))

    fig_compare.add_trace(go.Scatter(
        x=np.arange(projection_steps),
        y=median_stress,
        mode='lines',
        line=dict(width=4, dash="dash"),
        name="Stressed"
    ))

    fig_compare.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        template="plotly_white",
        showlegend=True
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # STRUCTURAL IMPACT METRICS
    # --------------------------------------------------------

    final_base = baseline_bundle[:, -1]
    final_stress = stressed_bundle[:, -1]

    base_var = np.percentile(final_base, 5)
    stress_var = np.percentile(final_stress, 5)

    base_survival = np.mean(final_base > 0.9 * base_capital)
    stress_survival = np.mean(final_stress > 0.9 * base_capital)

    impact_col1, impact_col2, impact_col3 = st.columns(3)

    impact_col1.metric("Baseline 5% VaR", round(base_var, 2))
    impact_col2.metric("Stressed 5% VaR", round(stress_var, 2))
    impact_col3.metric("Survival Shift",
                       round(base_survival - stress_survival, 3))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
