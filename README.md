# EIGO — Financial Nervous System

Live App: https://eigo-financial-system.streamlit.app/

A multi-layer probabilistic financial instability engine modeling systemic fragility, contagion dynamics, and capital survival using stochastic simulation.

---

## Overview

EIGO (Evolutionary Intelligence for Global Optimization) is a research-grade financial risk engine.

It does not attempt to predict stock prices.

Instead, it answers deeper structural questions:

- How stable is the global financial system?
- Where is systemic stress building?
- How does financial stress propagate across markets?
- What is the probability that capital survives under uncertainty?

EIGO models the financial system as a dynamic structural state machine rather than a price prediction tool.

---

## Why EIGO Was Built

Traditional financial dashboards focus on:

- Volatility
- Returns
- Drawdowns

However, systemic crises occur because of:

- Correlation compression
- Credit stress
- Liquidity breakdown
- Regime shifts
- Shock propagation

EIGO focuses on structural modeling instead of short-term forecasting.

---

## System Architecture

EIGO is built using a modular, layered architecture.

---

### 1. Data Layer

- Collects cross-asset financial data
- Cleans and preprocesses time series
- Computes daily returns and rolling statistics

Assets include:

- US Equity Index
- India Equity Index
- European & Asian indices
- Credit ETFs (Investment Grade & High Yield)
- Commodities (Gold, Oil)
- Volatility Index (VIX)
- Interest Rates

---

### 2. Structural Risk Engine

Computes systemic fragility using:

- Rolling volatility
- Cross-asset correlation compression
- Credit spread stress signals

Fragility is normalized to a 0–100 scale.

Higher values represent higher structural systemic stress.

---

### 3. Regime Detection Engine

Classifies the system into structural states:

- Stable
- Compression
- Elevated
- Fragile

Includes transition logic to model regime shift probability.

---

### 4. Contagion Modeling

Builds asset-level and regional transmission structures.

Measures:

- Cascade strength
- Network density
- Dominant stress region

This models how financial stress spreads across markets.

---

### 5. Global Instability Index

Combines structural signals into a unified score between 0 and 1.

Interpretation:

- Below 0.3 → Stable environment
- 0.3 – 0.6 → Elevated systemic pressure
- Above 0.6 → High structural instability

---

### 6. Simulation Engine

#### Monte Carlo Capital Survival

Runs thousands of simulations to estimate probability that capital remains above:

- 90% threshold
- 80% threshold

#### Digital Twin Modeling

Simulates how personal capital behaves under current systemic conditions.

Higher instability dynamically increases simulated volatility.

---

### 7. Multi-Layer Intelligence Stack

EIGO integrates:

- Shock propagation modeling
- Macro cycle phase detection
- Factor decomposition analysis
- Scenario branching
- Long-term stability projections
- System resilience stress testing
- Executive-level unified risk summary

Total: 30 structured analytical layers.

---

## Validation

The system has been validated through:

- Cold-start execution testing
- Numerical stability checks
- Monte Carlo distribution analysis
- Sensitivity testing
- Extreme shock stress modeling

All outputs remain bounded and logically consistent.

---

## Key Outputs

EIGO produces:

- Global Instability Index
- Adaptive Instability Index
- Regime Classification
- Early Warning Signal
- Dominant Stress Region
- Capital Survival Probabilities
- Contagion Strength Metrics
- Multi-Factor Risk Decomposition
- Scenario Probability Tree
- System Resilience Index
- Executive Intelligence Summary

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
- Plotly
- Streamlit
- Google Colab
- Open financial data (Yahoo Finance)

No paid tools were used.

---

## Limitations

- Uses historical correlation
- Assumes normal distribution in baseline simulation
- Does not explicitly model fat-tail distributions
- Not high-frequency or real-time streaming

Future versions may integrate:

- Regime-switching volatility models
- Heavy-tailed distributions
- Real-time data pipelines
- Machine learning extensions
- Bayesian structural inference

---

## Why This Project Matters

EIGO demonstrates:

- Quantitative reasoning
- Systems thinking
- Structural risk modeling
- Contagion analysis
- Monte Carlo simulation design
- Clean modular engineering
- Cloud deployment capability

This is not a dashboard.

It is a prototype financial nervous system.

---

## Author

Built as an independent research project focused on systemic risk modeling and probabilistic financial stability analysis.
