import streamlit as st
import plotly.graph_objects as go
import numpy as np

# =========================================================
# PAGE CONFIGURATION (Stable — No Sidebar Chaos)
# =========================================================

st.set_page_config(
    page_title="EIGO — Financial Nervous System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# GLOBAL CSS + ANIMATED PIXEL FIELD
# No floating mismatches.
# Pixel canvas sits behind all content.
# =========================================================

st.markdown("""
<style>

/* Remove Streamlit default UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Base typography */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Background gradient */
body {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f4f7fb 50%,
        #eef2f8 100%
    );
    overflow-x: hidden;
}

/* Pixel canvas container */
#pixel-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}

/* Section spacing */
.section {
    margin-top: 120px;
    margin-bottom: 120px;
}

/* Section title */
.section-title {
    font-size: 42px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 60px;
    letter-spacing: -1px;
}

/* Hero */
.hero-title {
    font-size: 70px;
    font-weight: 600;
    text-align: center;
    letter-spacing: -2px;
}

.hero-subtitle {
    font-size: 22px;
    text-align: center;
    color: #6e6e73;
}

/* Card styling */
.card {
    background: rgba(255,255,255,0.85);
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 20px 80px rgba(0,0,0,0.05);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 30px 100px rgba(0,0,0,0.08);
}

/* Slider alignment */
.stSlider {
    padding-left: 15%;
    padding-right: 15%;
}

</style>

<canvas id="pixel-canvas"></canvas>

<script>

const canvas = document.getElementById("pixel-canvas");
const ctx = canvas.getContext("2d");

let width = canvas.width = window.innerWidth;
let height = canvas.height = window.innerHeight;

window.addEventListener("resize", function() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
});

let particles = [];
const particleCount = 25;

for (let i = 0; i < particleCount; i++) {
    particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 2 + 1,
        dx: (Math.random() - 0.5) * 0.3,
        dy: (Math.random() - 0.5) * 0.3
    });
}

function animate() {
    ctx.clearRect(0, 0, width, height);

    particles.forEach(p => {
        p.x += p.dx;
        p.y += p.dy;

        if (p.x < 0 || p.x > width) p.dx *= -1;
        if (p.y < 0 || p.y > height) p.dy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(47,94,255,0.08)";
        ctx.fill();
    });

    requestAnimationFrame(animate);
}

animate();

</script>

""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("<div class='hero-title'>EIGO</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Institutional Financial Nervous System</div>", unsafe_allow_html=True)

# =========================================================
# CONTROL SECTION (LIVE — No Dead Button)
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Control Interface</div>", unsafe_allow_html=True)

instability_value = st.slider(
    "Macro Instability Level",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# GRID SECTION TEMPLATE (Side-by-Side Structure)
# This ensures alignment stability.
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Instability Reactor & Risk Metrics</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# -------------------------
# LEFT PANEL — Reactor
# -------------------------

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    reactor_size = 180 + instability_value * 200
    glow_strength = 20 + instability_value * 100

    st.markdown(f"""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        align-items:center;
        height:300px;
    ">
        <div style="
            width:{reactor_size}px;
            height:{reactor_size}px;
            border-radius:50%;
            background: radial-gradient(circle at center,
                rgba(255,255,255,0.95) 0%,
                rgba(17,17,17,0.85) 100%);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:40px;
            font-weight:600;
            color:#111111;
            box-shadow: 0 0 {glow_strength}px rgba(47,94,255,0.15);
            transition: all 0.3s ease;
        ">
            {round(instability_value,3)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# RIGHT PANEL — Metrics
# -------------------------

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    volatility = 0.006 + instability_value * 0.1
    correlation = 0.2 + instability_value * 0.6
    liquidity = 0.1 + instability_value * 0.7

    st.markdown(f"""
    <h3 style="text-align:center;">System Metrics</h3>
    <p style="text-align:center;">Volatility: {round(volatility,3)}</p>
    <p style="text-align:center;">Correlation: {round(correlation,3)}</p>
    <p style="text-align:center;">Liquidity Stress: {round(liquidity,3)}</p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: STRUCTURAL RISK GRID (2 x 2 ALIGNED PANELS)
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Structural Risk Surface & Forward Evolution</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 1
# -------------------------------

row1_col1, row1_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Structural Risk Surface
# --------------------------------

with row1_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    vol_axis = np.linspace(0.01, 0.6, 60)
    corr_axis = np.linspace(0.0, 1.0, 60)
    V, C = np.meshgrid(vol_axis, corr_axis)

    stress_surface = (V * 2.5) * (C * 1.8) * (0.5 + instability_value)

    fig_surface = go.Figure(
        data=[
            go.Contour(
                z=stress_surface,
                x=vol_axis,
                y=corr_axis,
                colorscale="Blues",
                contours=dict(showlines=False),
                hovertemplate="Vol: %{x:.2f}<br>Corr: %{y:.2f}<br>Stress: %{z:.2f}<extra></extra>"
            )
        ]
    )

    fig_surface.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_surface.update_xaxes(title="Volatility")
    fig_surface.update_yaxes(title="Correlation")

    st.plotly_chart(fig_surface, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Correlation Heat Grid
# --------------------------------

with row1_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    correlation_matrix = np.array([
        [1.0, correlation, 0.3],
        [correlation, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_corr.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row2_col1, row2_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Forward Capital Paths
# --------------------------------

with row2_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 180
    simulations = 400
    base_capital = 1_000_000

    volatility = 0.006 + instability_value * 0.1
    drift = 0.0002

    paths = []

    for _ in range(simulations):
        returns = np.random.normal(drift, volatility, steps)
        path = base_capital * np.cumprod(1 + returns)
        paths.append(path)

    paths = np.array(paths)

    median_path = np.median(paths, axis=0)
    q05 = np.quantile(paths, 0.05, axis=0)
    q95 = np.quantile(paths, 0.95, axis=0)

    time_axis = np.arange(steps)

    fig_paths = go.Figure()

    fig_paths.add_trace(go.Scatter(
        x=time_axis,
        y=q95,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_paths.add_trace(go.Scatter(
        x=time_axis,
        y=q05,
        fill='tonexty',
        fillcolor='rgba(47,94,255,0.15)',
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig_paths.add_trace(go.Scatter(
        x=time_axis,
        y=median_path,
        mode='lines',
        line=dict(width=3, color='#2F5EFF'),
        showlegend=False
    ))

    fig_paths.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_paths.update_xaxes(visible=False)
    fig_paths.update_yaxes(visible=False)

    st.plotly_chart(fig_paths, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Tail Risk Distribution
# --------------------------------

with row2_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    final_values = paths[:, -1]
    hist_y, hist_x = np.histogram(final_values, bins=80, density=True)
    x_mid = (hist_x[:-1] + hist_x[1:]) / 2

    fig_tail = go.Figure()

    fig_tail.add_trace(go.Scatter(
        x=x_mid,
        y=hist_y,
        mode='lines',
        line=dict(width=3, color='#2F5EFF'),
        showlegend=False
    ))

    fig_tail.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_tail.update_xaxes(visible=False)
    fig_tail.update_yaxes(visible=False)

    st.plotly_chart(fig_tail, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: SYSTEMIC LIQUIDITY & CONTAGION GRID
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Liquidity Stress & Systemic Contagion</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 1
# -------------------------------

row3_col1, row3_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Liquidity Stress Curve
# --------------------------------

with row3_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    liquidity_stress = 0.1 + instability_value * 0.8

    steps_liq = 200
    time_axis_liq = np.arange(steps_liq)

    liquidity_curve = (
        liquidity_stress
        + 0.05 * np.sin(time_axis_liq * 0.08)
        + np.random.normal(0, 0.01, steps_liq)
    )

    liquidity_curve = np.clip(liquidity_curve, 0, 1)

    fig_liquidity = go.Figure()

    fig_liquidity.add_trace(go.Scatter(
        x=time_axis_liq,
        y=liquidity_curve,
        mode='lines',
        line=dict(width=3, color='#2F5EFF'),
        showlegend=False
    ))

    fig_liquidity.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_liquidity.update_xaxes(visible=False)
    fig_liquidity.update_yaxes(visible=False)

    st.plotly_chart(fig_liquidity, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Contagion Network
# --------------------------------

with row3_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    nodes = {
        "Equities": (0.2, 0.8),
        "Credit": (0.8, 0.8),
        "Rates": (0.2, 0.2),
        "Commodities": (0.8, 0.2)
    }

    edges = [
        ("Equities", "Credit"),
        ("Equities", "Rates"),
        ("Credit", "Commodities"),
        ("Rates", "Commodities")
    ]

    fig_network = go.Figure()

    # Draw edges
    for edge in edges:
        x0, y0 = nodes[edge[0]]
        x1, y1 = nodes[edge[1]]

        fig_network.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=1 + instability_value * 6, color='rgba(47,94,255,0.4)'),
            showlegend=False
        ))

    # Draw nodes
    for node in nodes:
        x, y = nodes[node]
        fig_network.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=20 + instability_value * 40,
                color='#2F5EFF'
            ),
            text=[node],
            textposition="top center",
            showlegend=False
        ))

    fig_network.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_network.update_xaxes(visible=False)
    fig_network.update_yaxes(visible=False)

    st.plotly_chart(fig_network, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row4_col1, row4_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Regime Classification
# --------------------------------

with row4_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if instability_value < 0.25:
        regime = "Stable"
    elif instability_value < 0.5:
        regime = "Compression"
    elif instability_value < 0.75:
        regime = "Elevated"
    else:
        regime = "Fragile"

    st.markdown(f"""
    <h3 style="text-align:center;">Current Regime</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{regime}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Regime Transition Matrix
# --------------------------------

with row4_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    transition_matrix = np.array([
        [0.7, 0.2, 0.08, 0.02],
        [0.1, 0.6, 0.2, 0.1],
        [0.05, 0.2, 0.6, 0.15],
        [0.02, 0.08, 0.25, 0.65]
    ])

    # Slightly amplify transitions with instability
    transition_matrix = transition_matrix + instability_value * 0.05
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    fig_transition = go.Figure(
        data=go.Heatmap(
            z=transition_matrix,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_transition.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_transition, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: CAPITAL RESILIENCE & STRATEGIC INTELLIGENCE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Capital Resilience & Strategic Intelligence</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 1
# -------------------------------

row5_col1, row5_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Capital Survival Distribution
# --------------------------------

with row5_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    horizon = 365
    simulations_res = 600
    base_capital = 1_000_000

    drift_res = 0.00025
    vol_res = 0.008 + instability_value * 0.12

    final_vals = []
    drawdowns = []

    for _ in range(simulations_res):
        returns = np.random.normal(drift_res, vol_res, horizon)
        path = base_capital * np.cumprod(1 + returns)
        final_vals.append(path[-1])

        running_max = np.maximum.accumulate(path)
        dd = (path - running_max) / running_max
        drawdowns.append(dd.min())

    final_vals = np.array(final_vals)
    drawdowns = np.array(drawdowns)

    survival_prob = np.mean(final_vals > 0.85 * base_capital)

    hist_y, hist_x = np.histogram(final_vals, bins=90, density=True)
    x_mid = (hist_x[:-1] + hist_x[1:]) / 2

    fig_survival = go.Figure()

    fig_survival.add_trace(go.Scatter(
        x=x_mid,
        y=hist_y,
        mode='lines',
        line=dict(width=3, color='#2F5EFF'),
        showlegend=False
    ))

    fig_survival.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_survival.update_xaxes(visible=False)
    fig_survival.update_yaxes(visible=False)

    st.plotly_chart(fig_survival, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Drawdown Analysis
# --------------------------------

with row5_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    dd_mean = np.mean(drawdowns)
    dd_min = np.min(drawdowns)

    st.markdown(f"""
    <h3 style="text-align:center;">Drawdown Analysis</h3>
    <p style="text-align:center;">Average Max Drawdown: {round(dd_mean,3)}</p>
    <p style="text-align:center;">Worst Drawdown: {round(dd_min,3)}</p>
    <p style="text-align:center;">Survival Probability (85%): {round(survival_prob,3)}</p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row6_col1, row6_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Composite Risk Score
# --------------------------------

with row6_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    composite_score = (
        0.4 * instability_value +
        0.2 * volatility +
        0.2 * liquidity_stress +
        0.2 * (1 - survival_prob)
    )

    composite_score = min(composite_score, 1.0)

    st.markdown(f"""
    <h3 style="text-align:center;">Composite Risk Score</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(composite_score,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Strategic Posture
# --------------------------------

with row6_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if composite_score < 0.3:
        posture = "Aggressive Expansion"
    elif composite_score < 0.5:
        posture = "Balanced Allocation"
    elif composite_score < 0.75:
        posture = "Defensive Rotation"
    else:
        posture = "Capital Preservation"

    st.markdown(f"""
    <h3 style="text-align:center;">Strategic Posture</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{posture}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: SCENARIO TIMELINE & RISK RANKING
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Scenario Timeline & Dynamic Risk Ranking</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 1
# -------------------------------

row7_col1, row7_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Scenario Shock Simulator
# --------------------------------

with row7_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    scenario_shock = st.slider(
        "Scenario Shock Override",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01
    )

    adjusted_instability = min(instability_value + scenario_shock, 1.0)

    st.markdown(f"""
    <h3 style="text-align:center;">Adjusted Instability</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{round(adjusted_instability,3)}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Timeline Evolution Curve
# --------------------------------

with row7_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    timeline_steps = 250
    timeline_x = np.arange(timeline_steps)

    timeline_curve = (
        adjusted_instability
        + 0.1 * np.sin(timeline_x * 0.04)
        + 0.05 * np.cos(timeline_x * 0.02)
    )

    timeline_curve = np.clip(timeline_curve, 0, 1)

    fig_timeline = go.Figure()

    fig_timeline.add_trace(go.Scatter(
        x=timeline_x,
        y=timeline_curve,
        mode='lines',
        line=dict(width=3, color='#2F5EFF'),
        showlegend=False
    ))

    fig_timeline.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_timeline.update_xaxes(visible=False)
    fig_timeline.update_yaxes(visible=False)

    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row8_col1, row8_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Dynamic Risk Ranking Table
# --------------------------------

with row8_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    risk_components = {
        "Volatility": volatility,
        "Liquidity": liquidity_stress,
        "Correlation": correlation,
        "Survival Risk": 1 - survival_prob
    }

    sorted_risks = sorted(
        risk_components.items(),
        key=lambda x: x[1],
        reverse=True
    )

    table_html = "<h3 style='text-align:center;'>Risk Ranking</h3><table style='width:100%; text-align:center;'>"
    table_html += "<tr><th>Factor</th><th>Score</th></tr>"

    for factor, score in sorted_risks:
        table_html += f"<tr><td>{factor}</td><td>{round(score,3)}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Updated Composite Score
# --------------------------------

with row8_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    new_composite = (
        0.4 * adjusted_instability +
        0.2 * volatility +
        0.2 * liquidity_stress +
        0.2 * (1 - survival_prob)
    )

    new_composite = min(new_composite, 1.0)

    st.markdown(f"""
    <h3 style="text-align:center;">Scenario Composite Risk</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(new_composite,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: SYSTEM MEMORY & EARLY WARNING ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Memory & Early Warning Engine</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 1
# -------------------------------

row9_col1, row9_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Instability Momentum
# --------------------------------

with row9_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    momentum_series = np.gradient(timeline_curve)
    avg_momentum = np.mean(momentum_series[-20:])

    fig_momentum = go.Figure()

    fig_momentum.add_trace(go.Scatter(
        y=momentum_series,
        mode='lines',
        line=dict(width=2, color='#2F5EFF'),
        showlegend=False
    ))

    fig_momentum.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_momentum.update_xaxes(visible=False)
    fig_momentum.update_yaxes(visible=False)

    st.plotly_chart(fig_momentum, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Early Warning Indicator
# --------------------------------

with row9_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if avg_momentum > 0.02:
        warning = "Rising Systemic Pressure"
    elif avg_momentum < -0.02:
        warning = "Stabilizing Conditions"
    else:
        warning = "Neutral Momentum"

    st.markdown(f"""
    <h3 style="text-align:center;">Early Warning Signal</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{warning}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row10_col1, row10_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Rolling Instability Memory
# --------------------------------

with row10_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    memory_window = 50
    rolling_memory = np.convolve(
        timeline_curve,
        np.ones(memory_window)/memory_window,
        mode='valid'
    )

    fig_memory = go.Figure()

    fig_memory.add_trace(go.Scatter(
        y=rolling_memory,
        mode='lines',
        line=dict(width=3, color='#2F5EFF'),
        showlegend=False
    ))

    fig_memory.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_memory.update_xaxes(visible=False)
    fig_memory.update_yaxes(visible=False)

    st.plotly_chart(fig_memory, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Adaptive Intensity Scaling
# --------------------------------

with row10_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    adaptive_intensity = min(
        adjusted_instability +
        abs(avg_momentum) * 2,
        1.0
    )

    st.markdown(f"""
    <h3 style="text-align:center;">Adaptive Intensity</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(adaptive_intensity,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: GLOBAL REGIONAL FRAGILITY ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Global Regional Fragility Matrix</div>", unsafe_allow_html=True)

# -------------------------------
# DEFINE REGIONAL STRUCTURE
# -------------------------------

regions = {
    "US": adjusted_instability * 0.9 + 0.05,
    "Europe": adjusted_instability * 0.8 + 0.1,
    "Asia": adjusted_instability * 1.0,
    "Emerging": adjusted_instability * 1.1,
    "Commodities": adjusted_instability * 0.7 + 0.1
}

# Normalize
for key in regions:
    regions[key] = min(regions[key], 1.0)

# -------------------------------
# ROW 1
# -------------------------------

row11_col1, row11_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Regional Stress Heat Grid
# --------------------------------

with row11_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    region_names = list(regions.keys())
    region_values = list(regions.values())

    heat_matrix = np.array(region_values).reshape(1, -1)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heat_matrix,
            x=region_names,
            y=["Fragility"],
            colorscale="Blues",
            showscale=False
        )
    )

    fig_heat.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Regional Bar Ranking
# --------------------------------

with row11_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sorted_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)

    names_sorted = [r[0] for r in sorted_regions]
    values_sorted = [r[1] for r in sorted_regions]

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=values_sorted,
        y=names_sorted,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_bar.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_bar.update_xaxes(visible=False)
    fig_bar.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row12_col1, row12_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Regional Shock Simulation
# --------------------------------

with row12_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    selected_region = st.selectbox(
        "Select Region for Shock Simulation",
        region_names
    )

    region_shock_curve = (
        regions[selected_region]
        + 0.1 * np.sin(np.arange(200) * 0.05)
    )

    region_shock_curve = np.clip(region_shock_curve, 0, 1)

    fig_region = go.Figure()

    fig_region.add_trace(go.Scatter(
        y=region_shock_curve,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_region.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_region.update_xaxes(visible=False)
    fig_region.update_yaxes(visible=False)

    st.plotly_chart(fig_region, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Regional Contribution to Composite Risk
# --------------------------------

with row12_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    contribution_scores = {
        r: regions[r] * new_composite
        for r in regions
    }

    sorted_contrib = sorted(contribution_scores.items(), key=lambda x: x[1], reverse=True)

    table_html = "<h3 style='text-align:center;'>Regional Risk Contribution</h3><table style='width:100%; text-align:center;'>"
    table_html += "<tr><th>Region</th><th>Contribution</th></tr>"

    for region, score in sorted_contrib:
        table_html += f"<tr><td>{region}</td><td>{round(score,3)}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: CROSS-ASSET CAPITAL FLOW ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Cross-Asset Capital Flow & Rotation Engine</div>", unsafe_allow_html=True)

# -------------------------------
# DEFINE ASSET CLASSES
# -------------------------------

asset_classes = {
    "Equities": 0.6 - adjusted_instability * 0.5,
    "Credit": 0.5 - liquidity_stress * 0.4,
    "Rates": 0.4 + adjusted_instability * 0.4,
    "Commodities": 0.3 + regions["Commodities"] * 0.5,
    "Cash": 0.2 + new_composite * 0.6
}

# Normalize weights
total_weight = sum(asset_classes.values())
for key in asset_classes:
    asset_classes[key] = max(asset_classes[key] / total_weight, 0)

# -------------------------------
# ROW 1
# -------------------------------

row13_col1, row13_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Asset Allocation Pie
# --------------------------------

with row13_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=list(asset_classes.keys()),
                values=list(asset_classes.values()),
                hole=0.5
            )
        ]
    )

    fig_pie.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white",
        showlegend=True
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Flow Intensity Bar
# --------------------------------

with row13_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    flow_strength = {
        asset: abs(asset_classes[asset] - 0.2)
        for asset in asset_classes
    }

    sorted_flow = sorted(flow_strength.items(), key=lambda x: x[1], reverse=True)

    assets_sorted = [a[0] for a in sorted_flow]
    strengths_sorted = [a[1] for a in sorted_flow]

    fig_flow = go.Figure()

    fig_flow.add_trace(go.Bar(
        x=strengths_sorted,
        y=assets_sorted,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_flow.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_flow.update_xaxes(visible=False)
    fig_flow.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_flow, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row14_col1, row14_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Rotation Timeline
# --------------------------------

with row14_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    timeline_steps_rotation = 200
    rotation_curve = (
        adjusted_instability
        + 0.15 * np.sin(np.arange(timeline_steps_rotation) * 0.04)
        - 0.1 * liquidity_stress
    )

    rotation_curve = np.clip(rotation_curve, 0, 1)

    fig_rotation = go.Figure()

    fig_rotation.add_trace(go.Scatter(
        y=rotation_curve,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_rotation.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_rotation.update_xaxes(visible=False)
    fig_rotation.update_yaxes(visible=False)

    st.plotly_chart(fig_rotation, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Asset Risk Contribution
# --------------------------------

with row14_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    asset_risk_contrib = {
        asset: asset_classes[asset] * new_composite
        for asset in asset_classes
    }

    sorted_asset_risk = sorted(asset_risk_contrib.items(), key=lambda x: x[1], reverse=True)

    table_html = "<h3 style='text-align:center;'>Asset Risk Contribution</h3><table style='width:100%; text-align:center;'>"
    table_html += "<tr><th>Asset</th><th>Risk Contribution</th></tr>"

    for asset, score in sorted_asset_risk:
        table_html += f"<tr><td>{asset}</td><td>{round(score,3)}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: MULTI-FACTOR SENSITIVITY ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Multi-Factor Shock & Sensitivity Engine</div>", unsafe_allow_html=True)

# -------------------------------
# DEFINE FACTORS
# -------------------------------

factors = {
    "Volatility Shock": volatility,
    "Liquidity Shock": liquidity_stress,
    "Correlation Shock": correlation,
    "Macro Instability": adjusted_instability
}

# -------------------------------
# ROW 1
# -------------------------------

row15_col1, row15_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Shock Impact Matrix
# --------------------------------

with row15_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    shock_matrix = []

    for f1 in factors.values():
        row_vals = []
        for f2 in factors.values():
            impact = (f1 * 0.5 + f2 * 0.5) * new_composite
            row_vals.append(min(impact, 1.0))
        shock_matrix.append(row_vals)

    shock_matrix = np.array(shock_matrix)

    fig_shock = go.Figure(
        data=go.Heatmap(
            z=shock_matrix,
            x=list(factors.keys()),
            y=list(factors.keys()),
            colorscale="Blues",
            showscale=False
        )
    )

    fig_shock.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_shock, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Sensitivity Bar Chart
# --------------------------------

with row15_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sensitivity_scores = {
        factor: factors[factor] * new_composite
        for factor in factors
    }

    sorted_sens = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)

    sens_names = [x[0] for x in sorted_sens]
    sens_vals = [x[1] for x in sorted_sens]

    fig_sens = go.Figure()

    fig_sens.add_trace(go.Bar(
        x=sens_vals,
        y=sens_names,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_sens.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_sens.update_xaxes(visible=False)
    fig_sens.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row16_col1, row16_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Elasticity Surface
# --------------------------------

with row16_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    x_axis = np.linspace(0, 1, 60)
    y_axis = np.linspace(0, 1, 60)
    X, Y = np.meshgrid(x_axis, y_axis)

    elasticity_surface = (X * 0.6 + Y * 0.4) * adjusted_instability

    fig_elasticity = go.Figure(
        data=[
            go.Contour(
                z=elasticity_surface,
                x=x_axis,
                y=y_axis,
                colorscale="Blues",
                contours=dict(showlines=False)
            )
        ]
    )

    fig_elasticity.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_elasticity, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Composite Sensitivity Index
# --------------------------------

with row16_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    composite_sensitivity = min(
        np.mean(list(sensitivity_scores.values())) +
        np.std(list(sensitivity_scores.values())),
        1.0
    )

    st.markdown(f"""
    <h3 style="text-align:center;">Composite Sensitivity Index</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(composite_sensitivity,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: SYSTEM HEALTH & STATE ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Health & State Transition Engine</div>", unsafe_allow_html=True)

# -------------------------------
# COMPUTE SYSTEM HEALTH SCORE
# -------------------------------

system_health = 1 - (
    0.35 * adjusted_instability +
    0.25 * liquidity_stress +
    0.20 * composite_sensitivity +
    0.20 * (1 - survival_prob)
)

system_health = max(min(system_health, 1.0), 0.0)

# -------------------------------
# ROW 1
# -------------------------------

row17_col1, row17_col2 = st.columns(2)

# --------------------------------
# PANEL 1: System Health Gauge
# --------------------------------

with row17_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_health = go.Figure(go.Indicator(
        mode="gauge+number",
        value=system_health,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"},
        }
    ))

    fig_health.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_health, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: State Transition Model
# --------------------------------

with row17_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if system_health > 0.75:
        state = "Expansion"
    elif system_health > 0.5:
        state = "Stabilizing"
    elif system_health > 0.3:
        state = "Fragile"
    else:
        state = "Critical"

    st.markdown(f"""
    <h3 style="text-align:center;">System State</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{state}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row18_col1, row18_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Risk Pulse Wave
# --------------------------------

with row18_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    pulse_steps = 250
    pulse_wave = (
        adjusted_instability +
        0.15 * np.sin(np.arange(pulse_steps) * 0.06) *
        composite_sensitivity
    )

    pulse_wave = np.clip(pulse_wave, 0, 1)

    fig_pulse = go.Figure()

    fig_pulse.add_trace(go.Scatter(
        y=pulse_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_pulse.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_pulse.update_xaxes(visible=False)
    fig_pulse.update_yaxes(visible=False)

    st.plotly_chart(fig_pulse, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Health Breakdown Table
# --------------------------------

with row18_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    breakdown = {
        "Instability Impact": adjusted_instability,
        "Liquidity Impact": liquidity_stress,
        "Sensitivity Impact": composite_sensitivity,
        "Survival Impact": (1 - survival_prob)
    }

    sorted_breakdown = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)

    table_html = "<h3 style='text-align:center;'>Health Impact Breakdown</h3><table style='width:100%; text-align:center;'>"
    table_html += "<tr><th>Component</th><th>Impact</th></tr>"

    for comp, score in sorted_breakdown:
        table_html += f"<tr><td>{comp}</td><td>{round(score,3)}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: HISTORICAL REPLAY & TIME COMPRESSION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Historical Replay & Time Compression</div>", unsafe_allow_html=True)

# -------------------------------
# DEFINE REPLAY WINDOW
# -------------------------------

replay_speed = st.slider(
    "Replay Speed (Compression Factor)",
    min_value=1,
    max_value=10,
    value=3
)

# Simulated historical instability path
history_steps = 400
historical_instability = (
    0.4 +
    0.2 * np.sin(np.arange(history_steps) * 0.02) +
    0.1 * np.cos(np.arange(history_steps) * 0.05)
)

historical_instability = np.clip(historical_instability, 0, 1)

# -------------------------------
# ROW 1
# -------------------------------

row19_col1, row19_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Historical Instability Timeline
# --------------------------------

with row19_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_history = go.Figure()

    fig_history.add_trace(go.Scatter(
        y=historical_instability,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_history.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_history.update_xaxes(visible=False)
    fig_history.update_yaxes(visible=False)

    st.plotly_chart(fig_history, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Compressed Replay View
# --------------------------------

with row19_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    compressed_series = historical_instability[::replay_speed]

    fig_compress = go.Figure()

    fig_compress.add_trace(go.Scatter(
        y=compressed_series,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_compress.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_compress.update_xaxes(visible=False)
    fig_compress.update_yaxes(visible=False)

    st.plotly_chart(fig_compress, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row20_col1, row20_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Replay-Based Risk Estimate
# --------------------------------

with row20_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    replay_risk = np.mean(compressed_series) * new_composite

    st.markdown(f"""
    <h3 style="text-align:center;">Replay Risk Estimate</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(replay_risk,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Replay Volatility Surface
# --------------------------------

with row20_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    vol_surface = np.std(compressed_series)

    st.markdown(f"""
    <h3 style="text-align:center;">Replay Volatility</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(vol_surface,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: RISK CLUSTERING & PHASE DETECTION
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Risk Clustering & Phase Detection</div>", unsafe_allow_html=True)

# -------------------------------
# PREPARE CLUSTER DATA
# -------------------------------

cluster_data = np.array([
    adjusted_instability,
    liquidity_stress,
    composite_sensitivity,
    new_composite,
    1 - survival_prob
])

# Expand to time dimension
cluster_matrix = np.vstack([
    cluster_data + 0.05 * np.sin(i * 0.1)
    for i in range(200)
])

# -------------------------------
# ROW 1
# -------------------------------

row21_col1, row21_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Risk Cluster Heatmap
# --------------------------------

with row21_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_cluster = go.Figure(
        data=go.Heatmap(
            z=cluster_matrix.T,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_cluster.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Risk Factor Scatter Map
# --------------------------------

with row21_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    x_vals = cluster_matrix[:, 0]
    y_vals = cluster_matrix[:, 1]

    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        marker=dict(size=6, color='#2F5EFF', opacity=0.6),
        showlegend=False
    ))

    fig_scatter.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_scatter.update_xaxes(visible=False)
    fig_scatter.update_yaxes(visible=False)

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row22_col1, row22_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Phase Detection Engine
# --------------------------------

with row22_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    mean_cluster = np.mean(cluster_data)

    if mean_cluster < 0.3:
        phase = "Accumulation Phase"
    elif mean_cluster < 0.5:
        phase = "Stability Phase"
    elif mean_cluster < 0.7:
        phase = "Distribution Phase"
    else:
        phase = "Crisis Phase"

    st.markdown(f"""
    <h3 style="text-align:center;">Detected Market Phase</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{phase}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Phase Probability Index
# --------------------------------

with row22_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    phase_probability = min(mean_cluster + composite_sensitivity * 0.3, 1.0)

    st.markdown(f"""
    <h3 style="text-align:center;">Phase Probability Index</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(phase_probability,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: CAPITAL ALLOCATION & OPTIMIZATION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Capital Allocation & Optimization Engine</div>", unsafe_allow_html=True)

# -------------------------------
# DEFINE PORTFOLIO PARAMETERS
# -------------------------------

expected_returns = {
    "Equities": 0.08,
    "Credit": 0.05,
    "Rates": 0.03,
    "Commodities": 0.06,
    "Cash": 0.02
}

# Use previously computed asset_classes weights
weights = np.array(list(asset_classes.values()))
assets = list(asset_classes.keys())

returns_vector = np.array([expected_returns[a] for a in assets])

# Simple synthetic covariance
cov_matrix = np.diag([
    volatility,
    liquidity_stress,
    0.02,
    regions["Commodities"],
    0.01
])

# -------------------------------
# ROW 1
# -------------------------------

row23_col1, row23_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Efficient Frontier Simulation
# --------------------------------

with row23_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    num_portfolios = 500
    frontier_returns = []
    frontier_risk = []

    for _ in range(num_portfolios):
        random_weights = np.random.random(len(assets))
        random_weights /= np.sum(random_weights)

        port_return = np.dot(random_weights, returns_vector)
        port_risk = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))

        frontier_returns.append(port_return)
        frontier_risk.append(port_risk)

    fig_frontier = go.Figure()

    fig_frontier.add_trace(go.Scatter(
        x=frontier_risk,
        y=frontier_returns,
        mode='markers',
        marker=dict(size=6, color='#2F5EFF', opacity=0.6),
        showlegend=False
    ))

    fig_frontier.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_frontier.update_xaxes(visible=False)
    fig_frontier.update_yaxes(visible=False)

    st.plotly_chart(fig_frontier, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Current Portfolio Position
# --------------------------------

with row23_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    current_return = np.dot(weights, returns_vector)
    current_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    st.markdown(f"""
    <h3 style="text-align:center;">Current Portfolio Position</h3>
    <p style="text-align:center;">Expected Return: {round(current_return,3)}</p>
    <p style="text-align:center;">Expected Risk: {round(current_risk,3)}</p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row24_col1, row24_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Risk Budget Allocation
# --------------------------------

with row24_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    marginal_risk = np.dot(cov_matrix, weights)
    risk_contribution = weights * marginal_risk
    risk_contribution /= np.sum(risk_contribution)

    fig_risk_budget = go.Figure()

    fig_risk_budget.add_trace(go.Bar(
        x=risk_contribution,
        y=assets,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_risk_budget.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_risk_budget.update_xaxes(visible=False)
    fig_risk_budget.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_risk_budget, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Optimized Allocation Suggestion
# --------------------------------

with row24_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    optimized_weights = returns_vector / np.sum(returns_vector)

    table_html = "<h3 style='text-align:center;'>Optimized Allocation Suggestion</h3><table style='width:100%; text-align:center;'>"
    table_html += "<tr><th>Asset</th><th>Weight</th></tr>"

    for asset, w in zip(assets, optimized_weights):
        table_html += f"<tr><td>{asset}</td><td>{round(w,3)}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: NARRATIVE INTELLIGENCE & EXECUTIVE INSIGHTS
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Narrative Intelligence & Executive Insights</div>", unsafe_allow_html=True)

# -------------------------------
# AUTO-INSIGHT LOGIC
# -------------------------------

risk_direction = "increasing" if adjusted_instability > 0.5 else "contained"
liquidity_status = "tightening" if liquidity_stress > 0.5 else "stable"
survival_status = "resilient" if survival_prob > 0.8 else "vulnerable"

# -------------------------------
# ROW 1
# -------------------------------

row25_col1, row25_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Executive Risk Summary
# --------------------------------

with row25_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    executive_summary = f"""
    The system currently reflects a {state} macro environment.
    Instability levels are {risk_direction}, while liquidity conditions are {liquidity_status}.
    Capital resilience remains {survival_status}.
    Composite systemic risk is measured at {round(new_composite,3)}.
    """

    st.markdown("<h3 style='text-align:center;'>Executive Summary</h3>", unsafe_allow_html=True)
    st.write(executive_summary)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Strategic Interpretation
# --------------------------------

with row25_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if new_composite > 0.7:
        strategy_note = "Priority should shift toward capital preservation and defensive positioning."
    elif new_composite > 0.5:
        strategy_note = "Balanced allocation recommended with selective risk exposure."
    else:
        strategy_note = "Opportunity exists for selective expansion with controlled risk."

    st.markdown("<h3 style='text-align:center;'>Strategic Interpretation</h3>", unsafe_allow_html=True)
    st.write(strategy_note)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row26_col1, row26_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Risk Signal Breakdown
# --------------------------------

with row26_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    signal_breakdown = {
        "Instability": adjusted_instability,
        "Liquidity": liquidity_stress,
        "Sensitivity": composite_sensitivity,
        "Survival Weakness": 1 - survival_prob
    }

    strongest_signal = max(signal_breakdown, key=signal_breakdown.get)

    st.markdown("<h3 style='text-align:center;'>Dominant Risk Signal</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; color:#2F5EFF;'>{strongest_signal}</h2>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Forward Risk Outlook
# --------------------------------

with row26_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    forward_projection = min(
        adjusted_instability +
        composite_sensitivity * 0.4,
        1.0
    )

    outlook_text = (
        "Elevated forward volatility expected."
        if forward_projection > 0.6
        else "Forward risk outlook remains moderate."
    )

    st.markdown("<h3 style='text-align:center;'>Forward Risk Outlook</h3>", unsafe_allow_html=True)
    st.write(outlook_text)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: SYSTEM COHESION & MICRO-MOTION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Cohesion & Dynamic Pulse Engine</div>", unsafe_allow_html=True)

# -------------------------------
# CALCULATE COHERENCE SCORE
# -------------------------------

coherence_score = 1 - np.std([
    adjusted_instability,
    liquidity_stress,
    composite_sensitivity,
    new_composite
])

coherence_score = max(min(coherence_score, 1.0), 0.0)

# -------------------------------
# ROW 1
# -------------------------------

row27_col1, row27_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Coherence Indicator
# --------------------------------

with row27_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_coherence = go.Figure(go.Indicator(
        mode="gauge+number",
        value=coherence_score,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_coherence.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_coherence, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Dynamic Metric Pulse Bars
# --------------------------------

with row27_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    pulse_metrics = {
        "Instability": adjusted_instability,
        "Liquidity": liquidity_stress,
        "Sensitivity": composite_sensitivity,
        "Composite Risk": new_composite
    }

    names = list(pulse_metrics.keys())
    values = list(pulse_metrics.values())

    fig_pulse_bars = go.Figure()

    fig_pulse_bars.add_trace(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_pulse_bars.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_pulse_bars.update_xaxes(range=[0,1], visible=False)
    fig_pulse_bars.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_pulse_bars, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row28_col1, row28_col2 = st.columns(2)

# --------------------------------
# PANEL 3: System Stability Oscillator
# --------------------------------

with row28_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    osc_steps = 300
    stability_wave = (
        system_health +
        0.1 * np.sin(np.arange(osc_steps) * 0.05) *
        coherence_score
    )

    stability_wave = np.clip(stability_wave, 0, 1)

    fig_stability = go.Figure()

    fig_stability.add_trace(go.Scatter(
        y=stability_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_stability.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_stability.update_xaxes(visible=False)
    fig_stability.update_yaxes(visible=False)

    st.plotly_chart(fig_stability, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Live System Classification
# --------------------------------

with row28_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if coherence_score > 0.8:
        coherence_label = "Highly Coordinated System"
    elif coherence_score > 0.6:
        coherence_label = "Moderately Aligned"
    elif coherence_score > 0.4:
        coherence_label = "Fragmented Risk Signals"
    else:
        coherence_label = "Dislocated System State"

    st.markdown(f"""
    <h3 style="text-align:center;">System Coherence Classification</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{coherence_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: VISUAL RHYTHM & LAYOUT REFINEMENT ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Visual Refinement & Rhythm Analysis</div>", unsafe_allow_html=True)

# -------------------------------
# COMPUTE LAYOUT BALANCE SCORE
# -------------------------------

balance_metrics = np.array([
    adjusted_instability,
    liquidity_stress,
    composite_sensitivity,
    new_composite,
    system_health,
    coherence_score
])

layout_balance = 1 - np.std(balance_metrics)
layout_balance = max(min(layout_balance, 1.0), 0.0)

# -------------------------------
# ROW 1
# -------------------------------

row29_col1, row29_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Layout Stability Index
# --------------------------------

with row29_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_layout = go.Figure(go.Indicator(
        mode="gauge+number",
        value=layout_balance,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_layout.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_layout, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Panel Density Oscillator
# --------------------------------

with row29_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    density_wave = (
        layout_balance +
        0.1 * np.sin(np.arange(250) * 0.04)
    )

    density_wave = np.clip(density_wave, 0, 1)

    fig_density = go.Figure()

    fig_density.add_trace(go.Scatter(
        y=density_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_density.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_density.update_xaxes(visible=False)
    fig_density.update_yaxes(visible=False)

    st.plotly_chart(fig_density, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row30_col1, row30_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Visual Symmetry Classification
# --------------------------------

with row30_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if layout_balance > 0.8:
        symmetry_label = "Highly Harmonized Layout"
    elif layout_balance > 0.6:
        symmetry_label = "Balanced Visual Structure"
    elif layout_balance > 0.4:
        symmetry_label = "Moderate Structural Drift"
    else:
        symmetry_label = "Overloaded Visual Density"

    st.markdown(f"""
    <h3 style="text-align:center;">Visual Symmetry Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{symmetry_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: System Completion Index
# --------------------------------

with row30_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    completion_index = min(
        (coherence_score +
         system_health +
         layout_balance) / 3,
        1.0
    )

    st.markdown(f"""
    <h3 style="text-align:center;">System Completion Index</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(completion_index,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: INTERACTIVE DRILLDOWN ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Interactive Deep Analysis Modules</div>", unsafe_allow_html=True)

# --------------------------------
# DRILLDOWN 1 — Instability Deep Dive
# --------------------------------

with st.expander("Deep Dive: Instability Dynamics", expanded=False):

    deep_series = (
        adjusted_instability +
        0.2 * np.sin(np.arange(300) * 0.03) +
        0.1 * np.cos(np.arange(300) * 0.07)
    )

    deep_series = np.clip(deep_series, 0, 1)

    fig_deep = go.Figure()

    fig_deep.add_trace(go.Scatter(
        y=deep_series,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_deep.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_deep, use_container_width=True)

# --------------------------------
# DRILLDOWN 2 — Liquidity Structural View
# --------------------------------

with st.expander("Deep Dive: Liquidity Structural Breakdown", expanded=False):

    liquidity_series = (
        liquidity_stress +
        0.15 * np.sin(np.arange(300) * 0.05)
    )

    liquidity_series = np.clip(liquidity_series, 0, 1)

    fig_liq_deep = go.Figure()

    fig_liq_deep.add_trace(go.Scatter(
        y=liquidity_series,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_liq_deep.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_liq_deep, use_container_width=True)

# --------------------------------
# DRILLDOWN 3 — Allocation Sensitivity
# --------------------------------

with st.expander("Deep Dive: Allocation Sensitivity Matrix", expanded=False):

    sensitivity_matrix = np.outer(
        np.array(list(asset_classes.values())),
        np.array(list(asset_classes.values()))
    )

    fig_alloc = go.Figure(
        data=go.Heatmap(
            z=sensitivity_matrix,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_alloc.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_alloc, use_container_width=True)

# --------------------------------
# DRILLDOWN 4 — Stress Propagation Map
# --------------------------------

with st.expander("Deep Dive: Stress Propagation Simulation", expanded=False):

    propagation_series = (
        composite_sensitivity +
        0.25 * np.sin(np.arange(300) * 0.04)
    )

    propagation_series = np.clip(propagation_series, 0, 1)

    fig_prop = go.Figure()

    fig_prop.add_trace(go.Scatter(
        y=propagation_series,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_prop.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_prop, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: ADAPTIVE SCENARIO & LIVE SHOCK ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Adaptive Scenario & Live Shock Simulator</div>", unsafe_allow_html=True)

# -------------------------------
# SCENARIO CONTROLS
# -------------------------------

scenario_level = st.slider(
    "Macro Shock Intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

shock_speed = st.slider(
    "Shock Propagation Speed",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1
)

# -------------------------------
# ADJUSTED METRICS
# -------------------------------

shock_adjusted_instability = min(adjusted_instability + scenario_level * 0.6, 1.0)
shock_adjusted_liquidity = min(liquidity_stress + scenario_level * 0.5, 1.0)
shock_adjusted_composite = min(new_composite + scenario_level * 0.7, 1.0)

# -------------------------------
# ROW 1
# -------------------------------

row31_col1, row31_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Shock Instability Wave
# --------------------------------

with row31_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 350
    shock_wave = (
        shock_adjusted_instability +
        0.2 * np.sin(np.arange(steps) * 0.05 * shock_speed)
    )

    shock_wave = np.clip(shock_wave, 0, 1)

    fig_shock_instability = go.Figure()

    fig_shock_instability.add_trace(go.Scatter(
        y=shock_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_shock_instability.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_shock_instability.update_xaxes(visible=False)
    fig_shock_instability.update_yaxes(visible=False)

    st.plotly_chart(fig_shock_instability, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Shock Liquidity Drift
# --------------------------------

with row31_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    liquidity_wave = (
        shock_adjusted_liquidity +
        0.15 * np.cos(np.arange(steps) * 0.04 * shock_speed)
    )

    liquidity_wave = np.clip(liquidity_wave, 0, 1)

    fig_shock_liq = go.Figure()

    fig_shock_liq.add_trace(go.Scatter(
        y=liquidity_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_shock_liq.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_shock_liq.update_xaxes(visible=False)
    fig_shock_liq.update_yaxes(visible=False)

    st.plotly_chart(fig_shock_liq, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row32_col1, row32_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Shock Composite Indicator
# --------------------------------

with row32_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_shock_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=shock_adjusted_composite,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_shock_gauge.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_shock_gauge, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Capital Shock Survival
# --------------------------------

with row32_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    shock_survival = max(1 - shock_adjusted_composite * 0.8, 0)

    st.markdown(f"""
    <h3 style="text-align:center;">Shock Survival Probability</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(shock_survival,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: CROSS-ASSET FLOW DYNAMICS ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Cross-Asset Flow Dynamics</div>", unsafe_allow_html=True)

# -------------------------------
# DEFINE BASE FLOWS
# -------------------------------

equity_flow = max(1 - shock_adjusted_instability, 0)
credit_flow = max(1 - shock_adjusted_liquidity, 0)
rates_flow = 1 - shock_adjusted_composite
commodity_flow = 0.5 + 0.3 * np.sin(shock_adjusted_instability * 5)
cash_flow = shock_adjusted_composite

flow_vector = np.array([
    equity_flow,
    credit_flow,
    rates_flow,
    commodity_flow,
    cash_flow
])

flow_vector = np.clip(flow_vector, 0, None)
flow_vector = flow_vector / np.sum(flow_vector)

assets_labels = ["Equities", "Credit", "Rates", "Commodities", "Cash"]

# -------------------------------
# ROW 1
# -------------------------------

row33_col1, row33_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Capital Rotation Bar
# --------------------------------

with row33_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_rotation = go.Figure()

    fig_rotation.add_trace(go.Bar(
        x=flow_vector,
        y=assets_labels,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_rotation.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_rotation.update_xaxes(visible=False)
    fig_rotation.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_rotation, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Flow Oscillation Field
# --------------------------------

with row33_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    oscillation = (
        flow_vector[0] +
        0.2 * np.sin(np.arange(steps) * 0.05)
    )

    oscillation = np.clip(oscillation, 0, 1)

    fig_osc = go.Figure()

    fig_osc.add_trace(go.Scatter(
        y=oscillation,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_osc.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_osc.update_xaxes(visible=False)
    fig_osc.update_yaxes(visible=False)

    st.plotly_chart(fig_osc, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row34_col1, row34_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Momentum Heat Surface
# --------------------------------

with row34_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    momentum_matrix = np.outer(flow_vector, flow_vector)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=momentum_matrix,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_heat.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Dominant Flow Signal
# --------------------------------

with row34_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    dominant_asset = assets_labels[np.argmax(flow_vector)]

    st.markdown(f"""
    <h3 style="text-align:center;">Dominant Capital Destination</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{dominant_asset}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: GLOBAL CONTAGION NETWORK ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Global Contagion Network</div>", unsafe_allow_html=True)

# -------------------------------
# BUILD CONTAGION MATRIX
# -------------------------------

base_intensity = shock_adjusted_composite

contagion_matrix = np.array([
    [1, base_intensity*0.6, base_intensity*0.5, base_intensity*0.4, base_intensity*0.3],
    [base_intensity*0.6, 1, base_intensity*0.5, base_intensity*0.4, base_intensity*0.3],
    [base_intensity*0.5, base_intensity*0.5, 1, base_intensity*0.4, base_intensity*0.3],
    [base_intensity*0.4, base_intensity*0.4, base_intensity*0.4, 1, base_intensity*0.3],
    [base_intensity*0.3, base_intensity*0.3, base_intensity*0.3, base_intensity*0.3, 1]
])

assets_labels = ["Equities", "Credit", "Rates", "Commodities", "Cash"]

# -------------------------------
# ROW 1
# -------------------------------

row35_col1, row35_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Contagion Heatmap
# --------------------------------

with row35_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_contagion = go.Figure(
        data=go.Heatmap(
            z=contagion_matrix,
            x=assets_labels,
            y=assets_labels,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_contagion.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_contagion, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Average Transmission Strength
# --------------------------------

with row35_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    avg_transmission = np.mean(contagion_matrix[np.triu_indices(5, k=1)])

    st.markdown(f"""
    <h3 style="text-align:center;">Average Transmission Strength</h3>
    <h1 style="text-align:center; color:#2F5EFF;">{round(avg_transmission,3)}</h1>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row36_col1, row36_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Propagation Wave Simulation
# --------------------------------

with row36_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    propagation_wave = (
        avg_transmission +
        0.25 * np.sin(np.arange(steps) * 0.04)
    )

    propagation_wave = np.clip(propagation_wave, 0, 1)

    fig_prop_wave = go.Figure()

    fig_prop_wave.add_trace(go.Scatter(
        y=propagation_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_prop_wave.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_prop_wave.update_xaxes(visible=False)
    fig_prop_wave.update_yaxes(visible=False)

    st.plotly_chart(fig_prop_wave, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Systemic Risk Classification
# --------------------------------

with row36_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if avg_transmission > 0.7:
        contagion_label = "High Systemic Contagion"
    elif avg_transmission > 0.5:
        contagion_label = "Elevated Interconnected Risk"
    elif avg_transmission > 0.3:
        contagion_label = "Moderate Transmission"
    else:
        contagion_label = "Low Systemic Linkage"

    st.markdown(f"""
    <h3 style="text-align:center;">Contagion Classification</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{contagion_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: GLOBAL REGION SYNCHRONIZATION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Global Region Risk Synchronization</div>", unsafe_allow_html=True)

# -------------------------------
# REGION BASE STRESS
# -------------------------------

us_stress = shock_adjusted_instability
europe_stress = shock_adjusted_instability * 0.9
asia_stress = shock_adjusted_instability * 0.8
emerging_stress = shock_adjusted_instability * 1.1

region_vector = np.array([
    us_stress,
    europe_stress,
    asia_stress,
    emerging_stress
])

region_vector = np.clip(region_vector, 0, 1)

region_labels = ["US", "Europe", "Asia", "Emerging Markets"]

# -------------------------------
# ROW 1
# -------------------------------

row37_col1, row37_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Regional Stress Bar
# --------------------------------

with row37_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_region_bar = go.Figure()

    fig_region_bar.add_trace(go.Bar(
        x=region_vector,
        y=region_labels,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_region_bar.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_region_bar.update_xaxes(range=[0,1], visible=False)
    fig_region_bar.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_region_bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Synchronization Heat Surface
# --------------------------------

with row37_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sync_matrix = np.outer(region_vector, region_vector)

    fig_sync = go.Figure(
        data=go.Heatmap(
            z=sync_matrix,
            x=region_labels,
            y=region_labels,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_sync.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_sync, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row38_col1, row38_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Regional Phase Wave
# --------------------------------

with row38_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    phase_wave = (
        np.mean(region_vector) +
        0.2 * np.sin(np.arange(steps) * 0.05)
    )

    phase_wave = np.clip(phase_wave, 0, 1)

    fig_phase = go.Figure()

    fig_phase.add_trace(go.Scatter(
        y=phase_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_phase.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_phase.update_xaxes(visible=False)
    fig_phase.update_yaxes(visible=False)

    st.plotly_chart(fig_phase, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Global Alignment Status
# --------------------------------

with row38_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sync_level = np.std(region_vector)

    if sync_level < 0.05:
        alignment_label = "Highly Synchronized Global Risk"
    elif sync_level < 0.15:
        alignment_label = "Moderate Regional Divergence"
    else:
        alignment_label = "Fragmented Regional Stress"

    st.markdown(f"""
    <h3 style="text-align:center;">Global Alignment Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{alignment_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: LIQUIDITY PRESSURE & FUNDING STRESS ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Liquidity Pressure & Funding Stress</div>", unsafe_allow_html=True)

# -------------------------------
# FUNDING PRESSURE CALCULATION
# -------------------------------

funding_pressure = min(
    shock_adjusted_liquidity * 0.7 +
    avg_transmission * 0.3,
    1.0
)

market_depth = max(1 - funding_pressure * 0.8, 0)

liquidity_spiral_risk = min(
    funding_pressure * shock_adjusted_composite,
    1.0
)

# -------------------------------
# ROW 1
# -------------------------------

row39_col1, row39_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Funding Pressure Gauge
# --------------------------------

with row39_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_funding = go.Figure(go.Indicator(
        mode="gauge+number",
        value=funding_pressure,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_funding.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_funding, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Market Depth Compression
# --------------------------------

with row39_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_depth = go.Figure(go.Indicator(
        mode="gauge+number",
        value=market_depth,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_depth.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_depth, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row40_col1, row40_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Liquidity Spiral Simulation
# --------------------------------

with row40_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    spiral_wave = (
        liquidity_spiral_risk +
        0.3 * np.sin(np.arange(steps) * 0.05)
    )

    spiral_wave = np.clip(spiral_wave, 0, 1)

    fig_spiral = go.Figure()

    fig_spiral.add_trace(go.Scatter(
        y=spiral_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_spiral.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_spiral.update_xaxes(visible=False)
    fig_spiral.update_yaxes(visible=False)

    st.plotly_chart(fig_spiral, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Funding Stability Classification
# --------------------------------

with row40_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if funding_pressure > 0.7:
        funding_label = "Severe Funding Stress"
    elif funding_pressure > 0.5:
        funding_label = "Elevated Liquidity Pressure"
    elif funding_pressure > 0.3:
        funding_label = "Moderate Funding Tightness"
    else:
        funding_label = "Stable Funding Conditions"

    st.markdown(f"""
    <h3 style="text-align:center;">Funding Stability Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{funding_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: VOLATILITY REGIME PHASE TRANSITION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Volatility Regime Phase Transition</div>", unsafe_allow_html=True)

# -------------------------------
# VOLATILITY CLUSTER MODEL
# -------------------------------

volatility_cluster = min(
    shock_adjusted_instability * 0.6 +
    funding_pressure * 0.4,
    1.0
)

transition_intensity = min(
    avg_transmission * 0.5 +
    volatility_cluster * 0.5,
    1.0
)

phase_acceleration = min(
    transition_intensity * shock_adjusted_composite,
    1.0
)

# -------------------------------
# ROW 1
# -------------------------------

row41_col1, row41_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Volatility Cluster Gauge
# --------------------------------

with row41_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_vol_cluster = go.Figure(go.Indicator(
        mode="gauge+number",
        value=volatility_cluster,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_vol_cluster.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_vol_cluster, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Transition Intensity Gauge
# --------------------------------

with row41_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_transition = go.Figure(go.Indicator(
        mode="gauge+number",
        value=transition_intensity,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_transition.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_transition, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row42_col1, row42_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Phase Acceleration Wave
# --------------------------------

with row42_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    phase_wave = (
        phase_acceleration +
        0.25 * np.sin(np.arange(steps) * 0.06)
    )

    phase_wave = np.clip(phase_wave, 0, 1)

    fig_phase_wave = go.Figure()

    fig_phase_wave.add_trace(go.Scatter(
        y=phase_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_phase_wave.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_phase_wave.update_xaxes(visible=False)
    fig_phase_wave.update_yaxes(visible=False)

    st.plotly_chart(fig_phase_wave, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Regime Classification
# --------------------------------

with row42_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if phase_acceleration > 0.7:
        regime_label = "Rapid Regime Shift"
    elif phase_acceleration > 0.5:
        regime_label = "Elevated Transition Risk"
    elif phase_acceleration > 0.3:
        regime_label = "Moderate Volatility Phase"
    else:
        regime_label = "Stable Volatility Regime"

    st.markdown(f"""
    <h3 style="text-align:center;">Volatility Regime Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{regime_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: CAPITAL PRESERVATION & DRAWDOWN DEFENSE ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Capital Preservation & Drawdown Defense</div>", unsafe_allow_html=True)

# -------------------------------
# DEFENSE CALCULATIONS
# -------------------------------

capital_shield_strength = max(
    1 - shock_adjusted_composite * 0.8,
    0
)

drawdown_pressure = min(
    funding_pressure * phase_acceleration,
    1.0
)

drawdown_compression = max(
    1 - drawdown_pressure,
    0
)

defensive_ratio = min(
    shock_survival * capital_shield_strength,
    1.0
)

# -------------------------------
# ROW 1
# -------------------------------

row43_col1, row43_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Capital Shield Gauge
# --------------------------------

with row43_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_shield = go.Figure(go.Indicator(
        mode="gauge+number",
        value=capital_shield_strength,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_shield.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_shield, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Drawdown Compression Gauge
# --------------------------------

with row43_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_drawdown = go.Figure(go.Indicator(
        mode="gauge+number",
        value=drawdown_compression,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_drawdown.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_drawdown, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row44_col1, row44_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Defensive Oscillation Wave
# --------------------------------

with row44_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    defense_wave = (
        defensive_ratio +
        0.2 * np.sin(np.arange(steps) * 0.05)
    )

    defense_wave = np.clip(defense_wave, 0, 1)

    fig_defense = go.Figure()

    fig_defense.add_trace(go.Scatter(
        y=defense_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_defense.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_defense.update_xaxes(visible=False)
    fig_defense.update_yaxes(visible=False)

    st.plotly_chart(fig_defense, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Capital Defense Classification
# --------------------------------

with row44_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if defensive_ratio > 0.7:
        defense_label = "Strong Capital Defense"
    elif defensive_ratio > 0.5:
        defense_label = "Moderate Protection"
    elif defensive_ratio > 0.3:
        defense_label = "Weakening Defense"
    else:
        defense_label = "High Drawdown Risk"

    st.markdown(f"""
    <h3 style="text-align:center;">Capital Defense Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{defense_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: LONG-TERM STABILITY PROJECTION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Long-Term Stability Projection</div>", unsafe_allow_html=True)

# -------------------------------
# PROJECTION CALCULATIONS
# -------------------------------

base_projection = max(
    1 - shock_adjusted_composite * 0.7,
    0
)

resilience_factor = (
    capital_shield_strength * 0.5 +
    coherence_score * 0.5
)

projection_strength = min(
    base_projection * resilience_factor,
    1.0
)

# -------------------------------
# SIMULATED 5-YEAR TREND
# -------------------------------

years = 300
stability_trend = (
    projection_strength +
    0.15 * np.sin(np.arange(years) * 0.03) -
    0.1 * phase_acceleration
)

stability_trend = np.clip(stability_trend, 0, 1)

# -------------------------------
# ROW 1
# -------------------------------

row45_col1, row45_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Projection Strength Gauge
# --------------------------------

with row45_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_proj = go.Figure(go.Indicator(
        mode="gauge+number",
        value=projection_strength,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_proj.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: 5-Year Stability Trend
# --------------------------------

with row45_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        y=stability_trend,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_trend.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_trend.update_xaxes(visible=False)
    fig_trend.update_yaxes(visible=False)

    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row46_col1, row46_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Resilience Indicator
# --------------------------------

with row46_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_resilience = go.Figure(go.Indicator(
        mode="gauge+number",
        value=resilience_factor,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_resilience.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_resilience, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Stability Outlook Classification
# --------------------------------

with row46_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if projection_strength > 0.7:
        outlook_label = "Strong Long-Term Stability"
    elif projection_strength > 0.5:
        outlook_label = "Moderate Structural Resilience"
    elif projection_strength > 0.3:
        outlook_label = "Fragile Long-Term Outlook"
    else:
        outlook_label = "High Structural Instability"

    st.markdown(f"""
    <h3 style="text-align:center;">Long-Term Stability Outlook</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{outlook_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: MACRO CYCLE PHASE DETECTION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Macro Cycle Phase Detection</div>", unsafe_allow_html=True)

# -------------------------------
# MACRO CYCLE CALCULATIONS
# -------------------------------

expansion_score = max(
    projection_strength * (1 - funding_pressure),
    0
)

contraction_score = min(
    shock_adjusted_composite * phase_acceleration,
    1.0
)

late_cycle_score = min(
    volatility_cluster * 0.6 +
    funding_pressure * 0.4,
    1.0
)

# Normalize
cycle_total = expansion_score + contraction_score + late_cycle_score
if cycle_total == 0:
    cycle_total = 1

expansion_score /= cycle_total
contraction_score /= cycle_total
late_cycle_score /= cycle_total

# -------------------------------
# ROW 1
# -------------------------------

row47_col1, row47_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Cycle Distribution Bar
# --------------------------------

with row47_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_cycle_bar = go.Figure()

    fig_cycle_bar.add_trace(go.Bar(
        x=[expansion_score, contraction_score, late_cycle_score],
        y=["Expansion", "Contraction", "Late Cycle"],
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_cycle_bar.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_cycle_bar.update_xaxes(range=[0,1], visible=False)
    fig_cycle_bar.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_cycle_bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Macro Phase Wave
# --------------------------------

with row47_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    macro_wave = (
        expansion_score -
        contraction_score +
        0.2 * np.sin(np.arange(steps) * 0.04)
    )

    macro_wave = np.clip(macro_wave, -1, 1)

    fig_macro_wave = go.Figure()

    fig_macro_wave.add_trace(go.Scatter(
        y=macro_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_macro_wave.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_macro_wave.update_xaxes(visible=False)
    fig_macro_wave.update_yaxes(visible=False)

    st.plotly_chart(fig_macro_wave, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row48_col1, row48_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Dominant Macro Phase
# --------------------------------

with row48_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    scores = {
        "Expansion": expansion_score,
        "Contraction": contraction_score,
        "Late Cycle": late_cycle_score
    }

    dominant_phase = max(scores, key=scores.get)

    st.markdown(f"""
    <h3 style="text-align:center;">Dominant Macro Phase</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{dominant_phase}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Macro Stability Classification
# --------------------------------

with row48_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if dominant_phase == "Expansion":
        macro_label = "Growth-Driven Stability"
    elif dominant_phase == "Late Cycle":
        macro_label = "Overheating Risk"
    else:
        macro_label = "Downturn Risk"

    st.markdown(f"""
    <h3 style="text-align:center;">Macro Stability Outlook</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{macro_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: MULTI-FACTOR RISK DECOMPOSITION ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Multi-Factor Risk Decomposition</div>", unsafe_allow_html=True)

# -------------------------------
# FACTOR CONTRIBUTIONS
# -------------------------------

vol_factor = volatility_cluster * 0.25
liq_factor = funding_pressure * 0.25
contagion_factor = avg_transmission * 0.25
macro_factor = phase_acceleration * 0.25

defense_offset = capital_shield_strength * 0.3

total_risk_raw = vol_factor + liq_factor + contagion_factor + macro_factor
net_systemic_risk = max(total_risk_raw - defense_offset, 0)

factor_vector = np.array([
    vol_factor,
    liq_factor,
    contagion_factor,
    macro_factor
])

factor_labels = ["Volatility", "Liquidity", "Contagion", "Macro"]

# Normalize for display
factor_total = np.sum(factor_vector)
if factor_total == 0:
    factor_total = 1

factor_vector = factor_vector / factor_total

# -------------------------------
# ROW 1
# -------------------------------

row49_col1, row49_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Factor Contribution Bar
# --------------------------------

with row49_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_factor_bar = go.Figure()

    fig_factor_bar.add_trace(go.Bar(
        x=factor_vector,
        y=factor_labels,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_factor_bar.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_factor_bar.update_xaxes(range=[0,1], visible=False)
    fig_factor_bar.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_factor_bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Net Systemic Risk Gauge
# --------------------------------

with row49_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_net_risk = go.Figure(go.Indicator(
        mode="gauge+number",
        value=net_systemic_risk,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_net_risk.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_net_risk, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row50_col1, row50_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Risk Interaction Surface
# --------------------------------

with row50_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    interaction_matrix = np.outer(factor_vector, factor_vector)

    fig_interaction = go.Figure(
        data=go.Heatmap(
            z=interaction_matrix,
            x=factor_labels,
            y=factor_labels,
            colorscale="Blues",
            showscale=False
        )
    )

    fig_interaction.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_interaction, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Risk Balance Classification
# --------------------------------

with row50_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if net_systemic_risk > 0.7:
        risk_balance_label = "High Multi-Factor Risk"
    elif net_systemic_risk > 0.5:
        risk_balance_label = "Elevated Risk Mix"
    elif net_systemic_risk > 0.3:
        risk_balance_label = "Balanced Risk Structure"
    else:
        risk_balance_label = "Contained Risk Environment"

    st.markdown(f"""
    <h3 style="text-align:center;">Risk Structure Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{risk_balance_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: FORWARD SCENARIO TREE ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Forward Scenario Tree Projection</div>", unsafe_allow_html=True)

# -------------------------------
# SCENARIO BASE CALCULATIONS
# -------------------------------

base_path = max(
    projection_strength - net_systemic_risk * 0.5,
    0
)

optimistic_path = min(
    base_path + 0.2 * (1 - phase_acceleration),
    1.0
)

adverse_path = min(
    net_systemic_risk * 0.8,
    1.0
)

severe_path = min(
    shock_adjusted_composite * 0.9,
    1.0
)

scenario_vector = np.array([
    base_path,
    optimistic_path,
    adverse_path,
    severe_path
])

scenario_labels = [
    "Base",
    "Optimistic",
    "Adverse",
    "Severe"
]

# Normalize probabilities
total_scenario = np.sum(scenario_vector)
if total_scenario == 0:
    total_scenario = 1

scenario_vector = scenario_vector / total_scenario

# -------------------------------
# ROW 1
# -------------------------------

row51_col1, row51_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Scenario Probability Bar
# --------------------------------

with row51_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_scenario_bar = go.Figure()

    fig_scenario_bar.add_trace(go.Bar(
        x=scenario_vector,
        y=scenario_labels,
        orientation='h',
        marker_color='#2F5EFF'
    ))

    fig_scenario_bar.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_scenario_bar.update_xaxes(range=[0,1], visible=False)
    fig_scenario_bar.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_scenario_bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Scenario Path Simulation
# --------------------------------

with row51_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300

    scenario_wave = (
        scenario_vector[0] +
        0.15 * np.sin(np.arange(steps) * 0.04)
    )

    scenario_wave = np.clip(scenario_wave, 0, 1)

    fig_scenario_wave = go.Figure()

    fig_scenario_wave.add_trace(go.Scatter(
        y=scenario_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_scenario_wave.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_scenario_wave.update_xaxes(visible=False)
    fig_scenario_wave.update_yaxes(visible=False)

    st.plotly_chart(fig_scenario_wave, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row52_col1, row52_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Dominant Scenario
# --------------------------------

with row52_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    dominant_scenario = scenario_labels[np.argmax(scenario_vector)]

    st.markdown(f"""
    <h3 style="text-align:center;">Dominant Forward Scenario</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{dominant_scenario}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Scenario Risk Outlook
# --------------------------------

with row52_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if dominant_scenario == "Optimistic":
        scenario_label = "Growth-Driven Outlook"
    elif dominant_scenario == "Base":
        scenario_label = "Moderate Stability Outlook"
    elif dominant_scenario == "Adverse":
        scenario_label = "Heightened Risk Environment"
    else:
        scenario_label = "Severe Downside Risk"

    st.markdown(f"""
    <h3 style="text-align:center;">Scenario Outlook Classification</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{scenario_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: SYSTEM RESILIENCE STRESS TEST ENGINE
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>System Resilience Stress Test</div>", unsafe_allow_html=True)

# -------------------------------
# EXTREME SHOCK MODEL
# -------------------------------

extreme_shock_level = min(
    shock_adjusted_composite * 1.2,
    1.0
)

absorption_capacity = min(
    capital_shield_strength * 0.6 +
    coherence_score * 0.4,
    1.0
)

failure_probability = min(
    extreme_shock_level * (1 - absorption_capacity),
    1.0
)

resilience_index = max(
    1 - failure_probability,
    0
)

# -------------------------------
# ROW 1
# -------------------------------

row53_col1, row53_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Extreme Shock Gauge
# --------------------------------

with row53_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_extreme = go.Figure(go.Indicator(
        mode="gauge+number",
        value=extreme_shock_level,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_extreme.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(
        fig_extreme,
        use_container_width=True,
        key="p29_extreme_shock_gauge_unique"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: Absorption Capacity Gauge
# --------------------------------

with row53_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_absorption = go.Figure(go.Indicator(
        mode="gauge+number",
        value=absorption_capacity,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_absorption.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(
        fig_absorption,
        use_container_width=True,
        key="p29_absorption_capacity_unique"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row54_col1, row54_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Failure Probability Wave
# --------------------------------

with row54_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    steps = 300
    failure_wave = (
        failure_probability +
        0.25 * np.sin(np.arange(steps) * 0.05)
    )

    failure_wave = np.clip(failure_wave, 0, 1)

    fig_failure = go.Figure()

    fig_failure.add_trace(go.Scatter(
        y=failure_wave,
        mode='lines',
        line=dict(width=3, color='#2F5EFF')
    ))

    fig_failure.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    fig_failure.update_xaxes(visible=False)
    fig_failure.update_yaxes(visible=False)

    st.plotly_chart(
        fig_failure,
        use_container_width=True,
        key="p29_failure_wave_unique"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Resilience Classification
# --------------------------------

with row54_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if resilience_index > 0.7:
        resilience_label = "Highly Resilient System"
    elif resilience_index > 0.5:
        resilience_label = "Moderately Resilient"
    elif resilience_index > 0.3:
        resilience_label = "Vulnerable to Extreme Shock"
    else:
        resilience_label = "High Failure Risk"

    st.markdown(f"""
    <h3 style="text-align:center;">System Resilience Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{resilience_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# SECTION: EXECUTIVE INTELLIGENCE SUMMARY LAYER
# =========================================================

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Executive Intelligence Summary</div>", unsafe_allow_html=True)

# -------------------------------
# EXECUTIVE SCORE CALCULATION
# -------------------------------

executive_risk_score = min(
    net_systemic_risk * 0.6 +
    (1 - resilience_index) * 0.4,
    1.0
)

system_stability_score = max(
    projection_strength * 0.5 +
    coherence_score * 0.5,
    0
)

# -------------------------------
# ROW 1
# -------------------------------

row55_col1, row55_col2 = st.columns(2)

# --------------------------------
# PANEL 1: Executive Risk Gauge
# --------------------------------

with row55_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_exec_risk = go.Figure(go.Indicator(
        mode="gauge+number",
        value=executive_risk_score,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_exec_risk.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(
        fig_exec_risk,
        use_container_width=True,
        key="p30_executive_risk_gauge_unique"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 2: System Stability Gauge
# --------------------------------

with row55_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_exec_stability = go.Figure(go.Indicator(
        mode="gauge+number",
        value=system_stability_score,
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2F5EFF"}
        }
    ))

    fig_exec_stability.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    st.plotly_chart(
        fig_exec_stability,
        use_container_width=True,
        key="p30_system_stability_gauge_unique"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ROW 2
# -------------------------------

row56_col1, row56_col2 = st.columns(2)

# --------------------------------
# PANEL 3: Executive Risk Classification
# --------------------------------

with row56_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if executive_risk_score > 0.7:
        exec_label = "High Systemic Risk Environment"
    elif executive_risk_score > 0.5:
        exec_label = "Elevated Institutional Risk"
    elif executive_risk_score > 0.3:
        exec_label = "Balanced Risk Landscape"
    else:
        exec_label = "Stable Institutional Conditions"

    st.markdown(f"""
    <h3 style="text-align:center;">Executive Risk Status</h3>
    <h2 style="text-align:center; color:#2F5EFF;">{exec_label}</h2>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# PANEL 4: Unified Institutional Summary
# --------------------------------

with row56_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    summary_text = f"""
    Dominant Macro Phase: {dominant_phase}
    
    Dominant Forward Scenario: {dominant_scenario}
    
    Long-Term Stability Projection: {round(projection_strength, 3)}
    
    System Resilience Index: {round(resilience_index, 3)}
    
    Net Systemic Risk: {round(net_systemic_risk, 3)}
    """

    st.markdown("<h3 style='text-align:center;'>Unified Institutional Summary</h3>", unsafe_allow_html=True)
    st.write(summary_text)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
