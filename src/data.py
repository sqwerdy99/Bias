"""Download and cache daily OHLCV data from Yahoo Finance."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR, YEARS_OF_HISTORY

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
CACHE_MAX_AGE_SECONDS = 12 * 60 * 60  # refetch at most twice a day


def _cache_path(ticker: str, years: int) -> Path:
    return DATA_DIR / f"{ticker.replace('.', '_')}_{years}y.csv"


def download_prices(
    ticker: str, years: int = YEARS_OF_HISTORY, use_cache: bool = True
) -> pd.DataFrame:
    """Return a DataFrame of daily OHLCV indexed by date (tz-naive), oldest first.

    Prices are split/dividend adjusted (auto_adjust=True) so returns are not
    distorted by corporate actions. Results are cached as CSV in data/.
    """
    path = _cache_path(ticker, years)
    if use_cache and path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE_SECONDS:
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    hist = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    df = hist[OHLCV].copy()
    # Yahoo returns exchange-local timestamps; dates are all we need.
    df.index = pd.to_datetime(df.index.date)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Drop zero-volume rows (holidays / bad prints) that would create fake returns.
    df = df[df["Volume"] > 0].dropna()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df
