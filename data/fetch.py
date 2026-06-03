"""
data/fetch.py
-------------
Centralised data fetching for the project.
All scripts import from here rather than calling yfinance directly.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_returns(
    ticker: str = "^GSPC",
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> np.ndarray:
    """
    Download adjusted closing prices and return daily log-returns.
    r_t = log(S_t / S_{t-1})
    """
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    returns = np.log(raw / raw.shift(1)).dropna().values
    return returns


def fetch_returns_with_dates(
    ticker: str = "^GSPC",
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.Series:
    """
    Same as fetch_returns but returns a dated pd.Series.
    Used for time-series plots and rolling window analysis.
    """
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    returns = np.log(raw / raw.shift(1)).dropna()
    return returns


def fetch_vix(
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.Series:
    """
    Download CBOE VIX index (^VIX).
    Used as predictor variable in the Week 6 VG/NIG parameter regression.
    Returns a dated pd.Series of daily VIX closing levels.
    """
    raw = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    return raw.dropna()


def fetch_aligned(
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [returns, vix] aligned on trading dates.
    Convenience function for the rolling regression in Week 6.
    """
    returns = fetch_returns_with_dates(start=start, end=end)
    vix = fetch_vix(start=start, end=end)
    df = pd.DataFrame({"returns": returns, "vix": vix}).dropna()
    return df


# Four market shock periods used throughout the project
SHOCK_PERIODS = {
    "Dot-com crash":  ("2000-03-01", "2002-10-31"),
    "GFC":            ("2007-10-01", "2009-03-31"),
    "COVID-19":       ("2020-02-01", "2020-06-30"),
    "Fed rate hikes": ("2022-01-01", "2023-12-31"),
}
