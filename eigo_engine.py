import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from datetime import datetime


# ==============================
# CONFIG
# ==============================

CONFIG = {
    "start_date": "2015-01-01",
    "simulation_days": 90,
    "simulations": 2000,
    "initial_capital": 1_000_000
}


# ==============================
# DATA FETCH (ROBUST VERSION)
# ==============================

def fetch_data():
    tickers = ["^GSPC", "^NSEI", "^VIX", "^TNX", "GC=F", "HYG", "LQD"]
    end_date = datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(tickers, start=CONFIG["start_date"], end=end_date)

    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance")

    # Handle MultiIndex columns safely
    if isinstance(raw.columns, pd.MultiIndex):

        # Try Adjusted Close first
        if "Adj Close" in raw.columns.levels[0]:
            data = raw["Adj Close"]

        # Fallback to Close
        elif "Close" in raw.columns.levels[0]:
            data = raw["Close"]

        else:
            raise ValueError("Price columns not found in Yahoo response")

    else:
        data = raw

    data = data.dropna()

    return data


# ==============================
# FRAGILITY MODEL
# ==============================

def compute_fragility(data):

    returns = data.pct_change().dropna()

    volatility = returns.rolling(30).std().mean(axis=1)

    corr_matrix = (
        returns
        .rolling(30)
        .corr()
        .groupby(level=0)
        .mean()
        .mean(axis=1)
    )

    # Credit spread calculation (safe version)
    if "HYG" in data.columns and "LQD" in data.columns:
        credit_spread = (
            (data["HYG"] / data["LQD"])
            .pct_change()
            .rolling(30)
            .mean()
        )
    else:
        credit_spread = pd.Series(index=volatility.index, data=0)

    df = pd.concat([volatility, corr_matrix, credit_spread], axis=1)
    df.columns = ["vol", "corr", "credit"]
    df = df.dropna()

    fragility = (
        df["vol"].rank(pct=True) +
        df["corr"].rank(pct=True) +
        df["credit"].rank(pct=True)
    ) / 3

    fragility = fragility * 100

    return fragility


# ==============================
# REGIME DETECTION
# ==============================

def detect_regime(fragility):

    df = pd.DataFrame()
    df["fragility"] = fragility
    df["momentum"] = fragility.diff()
    df = df.dropna()

    if len(df) < 10:
        return "Stable"

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["regime"] = model.fit_predict(df)

    latest_regime = df["regime"].iloc[-1]

    regime_map = {
        0: "Stable",
        1: "Compression",
        2: "Elevated",
        3: "Fragile"
    }

    return regime_map.get(latest_regime, "Unknown")


# ==============================
# INSTABILITY INDEX
# ==============================

def compute_instability(fragility):

    if len(fragility) == 0:
        return 0.0

    instability = fragility.iloc[-1] / 100
    instability = max(0, min(instability, 1))  # bound between 0 and 1

    return round(float(instability), 3)


# ==============================
# DIGITAL TWIN SIMULATION
# ==============================

def digital_twin(instability):

    np.random.seed(42)

    base_vol = 0.01
    adjusted_vol = base_vol * (1 + instability * 3)
    mean_return = 0.0005

    end_values = []

    for _ in range(CONFIG["simulations"]):

        simulated_returns = np.random.normal(
            mean_return,
            adjusted_vol,
            CONFIG["simulation_days"]
        )

        path = CONFIG["initial_capital"] * np.cumprod(1 + simulated_returns)
        end_values.append(path[-1])

    end_values = np.array(end_values)

    survival_90 = np.mean(end_values > 0.9 * CONFIG["initial_capital"])
    survival_80 = np.mean(end_values > 0.8 * CONFIG["initial_capital"])

    return {
        "survival_90": round(float(survival_90), 3),
        "survival_80": round(float(survival_80), 3),
        "distribution": end_values
    }


# ==============================
# MASTER ENGINE
# ==============================

def run_engine():

    data = fetch_data()

    fragility = compute_fragility(data)

    regime = detect_regime(fragility)

    instability = compute_instability(fragility)

    twin = digital_twin(instability)

    return {
        "instability": instability,
        "regime": regime,
        "survival_90": twin["survival_90"],
        "survival_80": twin["survival_80"],
        "distribution": twin["distribution"]
    }
