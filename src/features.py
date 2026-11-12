"""Feature engineering for next-day direction prediction.

Every feature at row t uses only information available at the close of day t,
so there is no look-ahead. The target at row t looks one day forward.

Features are deliberately scale-free (ratios, returns, z-scores) rather than raw
prices, so a model trained on 2019 prices still makes sense on 2025 prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_WINDOWS = [1, 2, 5, 10]
SMA_WINDOWS = [5, 10, 20, 50]
VOL_WINDOWS = [5, 20]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (an EMA with alpha=1/window)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # If there were no losses in the window RSI is 100 by definition.
    return out.where(avg_loss != 0, 100.0)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to an OHLCV frame. Returns a new DataFrame."""
    df = prices.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    daily_ret = close.pct_change()

    # --- Momentum: trailing returns over several horizons
    for n in RETURN_WINDOWS:
        df[f"ret_{n}d"] = close.pct_change(n)

    # --- Trend: price relative to simple / exponential moving averages
    for n in SMA_WINDOWS:
        df[f"close_vs_sma_{n}"] = close / close.rolling(n).mean() - 1
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["close_vs_ema_12"] = close / ema_12 - 1
    df["close_vs_ema_26"] = close / ema_26 - 1
    df["macd_pct"] = (ema_12 - ema_26) / close  # MACD normalised by price
    df["sma_5_vs_sma_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1

    # --- Oscillator
    df["rsi_14"] = rsi(close, 14)

    # --- Volatility
    for n in VOL_WINDOWS:
        df[f"volatility_{n}d"] = daily_ret.rolling(n).std()
    df["vol_ratio_5_20"] = df["volatility_5d"] / df["volatility_20d"]
    df["hl_range"] = (high - low) / close  # intraday range as % of close

    # --- Volume
    df["volume_change"] = volume.pct_change()
    df["volume_vs_sma_20"] = volume / volume.rolling(20).mean() - 1

    # --- Position within recent range
    df["dist_from_20d_high"] = close / high.rolling(20).max() - 1
    df["dist_from_20d_low"] = close / low.rolling(20).min() - 1
    df["dist_from_52w_high"] = close / high.rolling(252, min_periods=126).max() - 1

    # --- Calendar (weak, but cheap and occasionally informative)
    df["day_of_week"] = df.index.day_of_week

    # --- Target: 1 if tomorrow's close is higher than today's
    df["next_return"] = close.shift(-1) / close - 1
    df["target"] = (df["next_return"] > 0).astype(int)

    # The last row has no "tomorrow", and the first ~50-126 rows have warm-up NaNs.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns(df) + ["next_return"])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Names of model input columns (everything that isn't raw OHLCV or the target)."""
    excluded = {"Open", "High", "Low", "Close", "Volume", "next_return", "target"}
    return [c for c in df.columns if c not in excluded]
