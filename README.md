# EIGO — Financial Nervous System

A multi-layer probabilistic financial instability engine modeling systemic fragility, contagion dynamics, and capital survival using stochastic simulation.

---

## Overview

EIGO (Evolutionary Intelligence for Global Optimization) is a research-grade financial risk engine.

It does not attempt to predict stock prices.

Instead, it answers deeper questions:

- How stable is the global financial system?
- Where is stress building?
- How does stress spread across markets?
- What is the probability that capital survives under uncertainty?

EIGO models the financial system as a dynamic state machine rather than a price prediction tool.

---

## Why EIGO Was Built

Traditional financial dashboards measure:

- Volatility
- Returns
- Drawdowns

But systemic crises occur because of:

- Correlation compression
- Credit stress
- Regime shifts
- Shock propagation

EIGO focuses on structural modeling instead of short-term forecasting.

---

## System Architecture

EIGO is built using a modular architecture.

### 1. Data Layer
- Collects cross-asset financial data
- Cleans and preprocesses data
- Computes daily returns

Assets include:
- Equity indices (US, India, Europe, Asia)
- Credit ETFs
- Commodities
- Volatility index
- Interest rates

---

### 2. Structural Risk Engine
Computes systemic fragility using:

- Rolling volatility
- Cross-asset correlation
- Credit stress signals

Fragility is normalized to a 0–100 scale.

Higher value means higher structural stress.

---

### 3. Regime Detection
Classifies the system into structural states:

- Stable
- Compression
- Elevated
- Fragile

A transition matrix estimates regime shift probabilities.

---

### 4. Contagion Modeling
Builds asset-level and regional networks.

Measures:
- Cascade strength
- Network density
- Dominant stress region

This models how financial stress spreads.

---

### 5. Global Instability Index
Combines multiple structural signals into a unified score between 0 and 1.

Interpretation:

- Below 0.3 → Stable
- 0.3 to 0.6 → Elevated
- Above 0.6 → High Instability

---

### 6. Simulation Engine

#### Monte Carlo Capital Survival
Runs thousands of simulations to estimate the probability that capital remains above:

- 90% threshold
- 80% threshold

#### Digital Twin Modeling
Simulates how personal capital behaves under current systemic conditions.

Higher instability increases simulated volatility.

---

## Validation

The system has been validated through:

- Cold-start execution testing
- Numerical stability checks
- Monte Carlo distribution analysis
- Sensitivity testing
- Extreme shock testing

All outputs remain bounded and logically consistent.

---

## Key Outputs

EIGO produces:

- Global Instability Index
- Adaptive Instability Index
- Current Regime Classification
- Early Warning Signal
- Dominant Stress Region
- Capital Survival Probabilities
- Contagion Strength Metrics

---

## Design Principles

EIGO follows three core principles:

1. Structural modeling over price prediction
2. Probabilistic simulation over deterministic forecasting
3. Modular architecture over monolithic scripting

---

## Tools Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab
- Open financial data (Yahoo Finance)

No paid tools were used.

---

## Limitations

- Uses historical correlation
- Assumes normal distribution in simulation
- Does not model extreme fat-tail events explicitly
- Not real-time streaming

Future versions may integrate:

- Regime-switching volatility models
- Heavy-tailed distributions
- Real-time data pipelines
- Machine learning extensions

---

## Why This Project Matters

EIGO demonstrates:

- Quantitative reasoning
- Risk modeling capability
- Systems thinking
- Contagion analysis
- Probabilistic simulation design
- Clean engineering structure

It is not a dashboard.

It is a prototype financial nervous system.

---

## Author

Built as an independent research project focused on systemic risk modeling and probabilistic financial stability analysis.
