"""
fetch_data.py
=============
Fetches historical stock data using yfinance and engineers three features:
  - Momentum   : 12-month trailing return (t-12 to t-1)
  - Volatility : 12-month rolling std of monthly returns
  - PE Ratio   : trailing P/E from yfinance fast_info

Target y:
  - Next month's return (forward 1-month return)

Usage:
  python fetch_data.py

Output:
  data/features.csv   — cleaned feature matrix ready for neural net
"""

import os
import numpy as np
import pandas as pd

# ── Try real yfinance fetch, fall back to synthetic data ───────────────────
try:
    import yfinance as yf
    USE_REAL = True
except ImportError:
    USE_REAL = False
    print("yfinance not installed. Run: pip install yfinance")
    print("Using synthetic data for now.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "JPM",  "GS",   "BAC",   "WMT",  "XOM"]
START    = "2015-01-01"
END      = "2024-01-01"
OUT_DIR  = "data"
OUT_FILE = os.path.join(OUT_DIR, "features.csv")


# ══════════════════════════════════════════════════════════════════════════════
# REAL DATA — yfinance
# ══════════════════════════════════════════════════════════════════════════════

def fetch_real_data():
    """
    Downloads monthly price data for each ticker.
    Engineers momentum, volatility, PE ratio.
    Returns a clean DataFrame.
    """
    print(f"Fetching data for {len(TICKERS)} tickers from {START} to {END}...")

    all_rows = []

    for ticker in TICKERS:
        print(f"  Processing {ticker}...")
        try:
            stock = yf.Ticker(ticker)

            # ── Monthly adjusted close prices ──────────────────────────────
            hist = stock.history(start=START, end=END, interval="1mo")
            if hist.empty or len(hist) < 15:
                print(f"    Skipping {ticker} — insufficient data")
                continue

            hist = hist[["Close"]].copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)

            # ── Monthly returns ────────────────────────────────────────────
            hist["ret"] = hist["Close"].pct_change()

            # ── Feature 1: Momentum (12-month trailing return, skip last) ──
            # Standard academic momentum: t-12 to t-1 (skip most recent month)
            hist["momentum"] = (
                hist["Close"].shift(1) / hist["Close"].shift(13) - 1
            )

            # ── Feature 2: Volatility (12-month rolling std of returns) ────
            hist["volatility"] = hist["ret"].rolling(12).std()

            # ── Feature 3: PE Ratio ────────────────────────────────────────
            # yfinance gives current PE — we use it as a constant per ticker
            # (historical PE requires premium data sources)
            try:
                pe = stock.fast_info.get("trailingPE", np.nan)
            except Exception:
                pe = np.nan
            hist["pe_ratio"] = pe

            # ── Target: Forward 1-month return ────────────────────────────
            hist["target"] = hist["ret"].shift(-1)

            # ── Add ticker label ───────────────────────────────────────────
            hist["ticker"] = ticker
            all_rows.append(hist)

        except Exception as e:
            print(f"    Error fetching {ticker}: {e}")
            continue

    if not all_rows:
        raise ValueError("No data fetched — check your internet connection.")

    df = pd.concat(all_rows)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA — for offline testing
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples=500, seed=42):
    """
    Generates realistic synthetic financial features.
    Injects a weak true signal so the NN has something to learn.

    True relationship (with noise):
        y = 0.03 * momentum - 0.02 * volatility + 0.001 * pe_ratio + noise
    """
    print("Generating synthetic financial data...")
    print(f"  n_samples = {n_samples}")
    print(f"  Features  = momentum, volatility, pe_ratio")
    print(f"  Target    = next month return\n")

    rng = np.random.default_rng(seed)

    # ── Simulate realistic feature distributions ───────────────────────────
    momentum   = rng.normal(loc=0.08,  scale=0.20, size=n_samples)
    volatility = rng.uniform(low=0.01, high=0.08,  size=n_samples)
    pe_ratio   = rng.uniform(low=8.0,  high=45.0,  size=n_samples)

    # ── Weak true signal + noise (finance is noisy!) ───────────────────────
    noise  = rng.normal(loc=0.0, scale=0.04, size=n_samples)
    target = (0.03 * momentum
             - 0.02 * volatility
             + 0.001 * pe_ratio
             + noise)

    df = pd.DataFrame({
        "momentum"  : momentum,
        "volatility": volatility,
        "pe_ratio"  : pe_ratio,
        "target"    : target,
        "ticker"    : "SYNTHETIC",
    })

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN & SAVE
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_save(df):
    """
    Selects features, drops NaNs, normalises features, saves to CSV.
    NOTE: We normalise X features only — never the target y.
    """
    features = ["momentum", "volatility", "pe_ratio", "target"]
    df = df[features + ["ticker"]].copy()

    # Drop NaN and infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # ── Normalise features (zero mean, unit variance) ──────────────────────
    for col in ["momentum", "volatility", "pe_ratio"]:
        mean = df[col].mean()
        std  = df[col].std()
        df[col] = (df[col] - mean) / (std + 1e-8)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} rows to {OUT_FILE}")
    print(f"\nFeature stats after normalisation:")
    print(df[["momentum","volatility","pe_ratio","target"]].describe().round(4))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if USE_REAL:
        df = fetch_real_data()
    else:
        df = generate_synthetic_data(n_samples=500)

    df = clean_and_save(df)

    print(f"\nFinal dataset shape : {df.shape}")
    print(f"Columns             : {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
