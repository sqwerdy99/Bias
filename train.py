"""Command-line training run: fetch data, train all models, print and save results.

Usage:
    python train.py                      # default tickers from src/config.py
    python train.py --tickers AAPL SHOP.TO --refresh
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import REPORTS_DIR, TICKERS, YEARS_OF_HISTORY
from src.pipeline import run_pipeline
from src.plots import save_static_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--years", type=int, default=YEARS_OF_HISTORY)
    parser.add_argument("--refresh", action="store_true", help="ignore cached CSVs and re-download")
    args = parser.parse_args()

    all_metrics = []
    for ticker in args.tickers:
        print(f"\n=== {ticker} ===")
        result = run_pipeline(ticker, years=args.years, use_cache=not args.refresh)
        print(f"train: {result.train.index[0]:%Y-%m-%d} -> {result.train.index[-1]:%Y-%m-%d} "
              f"({len(result.train)} rows)")
        print(f"test:  {result.test.index[0]:%Y-%m-%d} -> {result.test.index[-1]:%Y-%m-%d} "
              f"({len(result.test)} rows, {result.test['target'].mean():.1%} up days)")
        print(result.metrics.round(3).to_string())

        save_static_reports(result, REPORTS_DIR)
        all_metrics.append(result.metrics.assign(ticker=ticker).rename_axis("model").reset_index())

    summary = pd.concat(all_metrics, ignore_index=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(REPORTS_DIR / "metrics.csv", index=False)

    print("\n=== Test accuracy by model (rows) and ticker (columns) ===")
    print(summary.pivot(index="model", columns="ticker", values="accuracy").round(3).to_string())
    print(f"\nSaved metrics and charts to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
