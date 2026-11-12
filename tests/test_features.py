"""Sanity checks on feature engineering — mainly that nothing peeks into the future."""

import numpy as np
import pandas as pd

from src.features import build_features, feature_columns, rsi
from src.models import time_split


def synthetic_prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
        },
        index=idx,
    )


def test_target_is_next_day_direction():
    df = build_features(synthetic_prices())
    full = synthetic_prices()["Close"]
    for date in df.index[:20]:
        pos = full.index.get_loc(date)
        assert df.loc[date, "target"] == int(full.iloc[pos + 1] > full.iloc[pos])


def test_no_nans_and_last_row_dropped():
    prices = synthetic_prices()
    df = build_features(prices)
    assert not df[feature_columns(df)].isna().any().any()
    assert df.index[-1] < prices.index[-1]  # last day has no "tomorrow"


def test_features_do_not_use_future_data():
    """Changing tomorrow's prices must not change today's features."""
    prices = synthetic_prices()
    base = build_features(prices)
    cutoff = base.index[200]

    tampered = prices.copy()
    tampered.loc[tampered.index > cutoff, ["Open", "High", "Low", "Close"]] *= 3
    tampered.loc[tampered.index > cutoff, "Volume"] *= 5
    changed = build_features(tampered)

    cols = feature_columns(base)
    pd.testing.assert_frame_equal(base.loc[:cutoff, cols], changed.loc[:cutoff, cols])


def test_time_split_is_chronological():
    df = build_features(synthetic_prices())
    train, test = time_split(df, 0.2)
    assert train.index.max() < test.index.min()
    assert len(test) == len(df) - int(len(df) * 0.8)


def test_rsi_bounds():
    r = rsi(synthetic_prices()["Close"]).dropna()
    assert r.between(0, 100).all()
