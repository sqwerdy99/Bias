# stock-direction-xgboost

Predicts whether a stock's **next-day close will be higher than today's** using
engineered technical features and gradient-boosted trees (XGBoost, LightGBM),
benchmarked against scikit-learn baselines. Includes a Streamlit dashboard for
exploring the data, the metrics and the predictions.

> **This is not a trading strategy.** It is a demonstration of a clean, leak-free
> ML workflow on financial time series. Next-day direction for large-cap stocks
> is close to a coin flip, and the results below show that honestly.

## What it does

1. **Data:** downloads about 8 years of split- and dividend-adjusted daily OHLCV from Yahoo
   Finance (`yfinance`) for AAPL, MSFT, TSLA and RY.TO (Royal Bank of Canada). Any
   Yahoo symbol (e.g. `SHOP.TO`) also works. Downloads are cached in `data/`.
2. **Features:** 24 scale-free features that use only information available at the
   day's close (see the table below).
3. **Target:** `1` if tomorrow's close > today's close, else `0`.
4. **Split:** chronological. The first 80% of days train the models and the last 20% (about 18
   months) are the test set. Nothing is shuffled.
5. **Models:** majority-class dummy, Logistic Regression, Random Forest, **XGBoost**,
   **LightGBM**.
6. **Evaluation:** accuracy, precision, recall, F1, ROC AUC and a confusion matrix,
   plus feature importance and an illustrative long/flat equity curve.

## Engineered features

| Group | Features |
|---|---|
| Momentum | `ret_1d`, `ret_2d`, `ret_5d`, `ret_10d`: trailing % returns |
| Trend | close vs SMA 5/10/20/50, close vs EMA 12/26, MACD (÷ price), SMA5 vs SMA20 |
| Oscillator | RSI(14) with Wilder smoothing |
| Volatility | 5- and 20-day std of daily returns, their ratio, intraday high-low range |
| Volume | day-over-day volume change, volume vs 20-day average |
| Price position | distance from 20-day high/low and from 52-week high |
| Calendar | day of week |

All features are ratios or returns rather than raw prices, so a model trained when
AAPL was $50 still makes sense when it's $250.

## Results (test period May 2025 – Nov 2026, ~377 trading days)

Test-set accuracy from `python train.py`:

| Model | AAPL | MSFT | TSLA | RY.TO |
|---|---|---|---|---|
| Majority class (always "up") | 0.528 | 0.517 | 0.512 | **0.570** |
| Logistic Regression | 0.501 | 0.499 | 0.501 | 0.520 |
| Random Forest | 0.541 | 0.493 | 0.515 | 0.541 |
| XGBoost | 0.557 | 0.509 | 0.493 | 0.541 |
| LightGBM | **0.565** | 0.477 | **0.525** | 0.523 |

**In plain language:**
- The boosted models beat the "always predict up" baseline on AAPL by about 3 points, and
  their ROC AUC there (~0.56) shows they rank days a little better than chance.
- On MSFT, TSLA and RY.TO, no model reliably beats the baseline. On RY.TO, which
  rose on 57% of test days, simply always guessing "up" wins.
- Feature importance is spread almost evenly across features. No single indicator
  carries much signal, which fits a market that is close to efficient at the
  daily horizon.
- A ~1-year test window is small. A 2–3 point edge is within the range that
  noise can produce, so treat these numbers as anecdotal, not significant.

Numbers will change slightly each time you re-download data.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py          # dashboard (main entry point)
python train.py               # CLI: trains all tickers, writes reports/metrics.csv + PNG charts
python train.py --tickers SHOP.TO NVDA --refresh
pytest                        # unit tests (look-ahead checks, split, RSI bounds)
```

### Dashboard

- Pick a ticker from the sidebar (or type any Yahoo symbol) and adjust years of history.
- **Price history:** adjusted close.
- **Model metrics:** metrics table for all five models and a confusion matrix.
- **Feature importance:** XGBoost, LightGBM or Random Forest gain importance, plus a
  correlation heatmap.
- **Test-set predictions:** predicted up/down days drawn on the price chart, a
  long/flat vs buy-and-hold curve, and a downloadable predictions table.
- **Re-download data & retrain** fetches fresh data and refits the models.

## Project structure

```
├── app.py               Streamlit dashboard
├── train.py             CLI training + report generation
├── src/
│   ├── config.py        tickers, history length, test fraction
│   ├── data.py          yfinance download + CSV cache
│   ├── features.py      feature engineering + target
│   ├── models.py        models, time split, metrics, equity curve
│   ├── pipeline.py      ticker -> trained TrainResult
│   └── plots.py         Plotly (dashboard) and Seaborn (PNG) charts
├── notebooks/exploration.ipynb
├── tests/test_features.py
├── data/                cached CSVs (git-ignored)
└── reports/             metrics.csv + charts from train.py
```

## Design choices & limitations

- **No look-ahead:** a unit test changes future prices and checks that past features
  stay identical. The train/test split is strictly chronological.
- **Conservative hyperparameters** (depth 3, learning rate 0.03, subsampling): deeper trees
  overfit noise and score worse out of sample. No hyperparameter search was run.
  Tuning on a single test split would just overfit the test set.
- **One static split.** A walk-forward (rolling-origin) evaluation would give a more
  reliable estimate and is the obvious next step.
- **The equity curve ignores transaction costs, slippage and taxes.** A strategy that
  trades daily on a ~52% hit rate would very likely lose money after costs.
- **Survivorship and selection bias:** these tickers were chosen because they are
  well-known winners today.
- Yahoo Finance data is free but unofficial. Occasional gaps or revisions happen.

## Possible extensions

Walk-forward validation, probability calibration and a confidence threshold
(only trade when P(up) > 0.55), market-wide features (SPY/VIX/sector returns),
SHAP explanations, and cost-aware backtesting.
