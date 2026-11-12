"""Streamlit dashboard for the Bias project.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import TICKERS, YEARS_OF_HISTORY
from src.models import strategy_returns
from src.pipeline import run_pipeline
from src import plots

st.set_page_config(page_title="Bias", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner=False)
def get_result(ticker: str, years: int, refresh_token: int):
    """Train (or reuse) models for a ticker. `refresh_token` busts the cache on retrain."""
    return run_pipeline(ticker, years=years, use_cache=refresh_token == 0)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
ticker = st.sidebar.selectbox("Ticker", TICKERS + ["Other…"])
if ticker == "Other…":
    ticker = st.sidebar.text_input("Yahoo Finance symbol", value="SHOP.TO").strip().upper()
years = st.sidebar.slider("Years of history", min_value=5, max_value=15, value=YEARS_OF_HISTORY)

if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if st.sidebar.button("Re-download data & retrain", width="stretch"):
    st.session_state.refresh += 1

st.sidebar.markdown("---")
st.sidebar.caption(
    "Educational project. Predicting next-day direction is close to a coin flip; "
    "nothing here is investment advice or a tradable strategy."
)

# ---------------------------------------------------------------- data + training
if not ticker:
    st.stop()
try:
    with st.spinner(f"Fetching {ticker} and training 5 models…"):
        result = get_result(ticker, years, st.session_state.refresh)
except Exception as exc:  # bad symbol, network issue, etc.
    st.error(f"Could not load **{ticker}**: {exc}")
    st.stop()

featured = pd.concat([result.train, result.test])
test = result.test

# ---------------------------------------------------------------- header
st.title("Bias")
st.markdown(
    "Gradient-boosted trees (XGBoost, LightGBM) vs. simple baselines, trained on "
    "technical features to predict whether tomorrow's close will be **higher** than today's."
)

c1, c2, c3, c4 = st.columns(4)
xgb = result.metrics.loc["XGBoost"]
base = result.metrics.loc["Majority class"]
c1.metric("Ticker", ticker)
c2.metric("Test period", f"{test.index[0]:%b %Y} – {test.index[-1]:%b %Y}", f"{len(test)} days",
          delta_color="off")
c3.metric("XGBoost accuracy", f"{xgb['accuracy']:.1%}",
          f"{xgb['accuracy'] - base['accuracy']:+.1%} vs majority class")
c4.metric("Up days in test set", f"{test['target'].mean():.1%}")

tab_price, tab_metrics, tab_features, tab_preds = st.tabs(
    ["Price history", "Model metrics", "Feature importance", "Test-set predictions"]
)

# ---------------------------------------------------------------- tabs
with tab_price:
    st.plotly_chart(plots.price_chart(featured, ticker), width="stretch")
    st.caption(f"Training: {result.train.index[0]:%Y-%m-%d} → {result.train.index[-1]:%Y-%m-%d} "
               f"({len(result.train)} days). Test: last {len(test)} days, never seen during training.")

with tab_metrics:
    st.subheader("Test-set performance")
    styled = (result.metrics.style
              .format("{:.3f}")
              .highlight_max(axis=0, props="font-weight: bold;"))
    st.dataframe(styled, width="stretch")
    st.caption(
        "**Majority class** always predicts the more common training label — any model that "
        "can't beat it has learned nothing useful. ROC AUC of 0.5 = random ranking."
    )
    left, right = st.columns(2)
    cm_model = left.selectbox("Confusion matrix for", list(result.models), index=3)
    left.plotly_chart(plots.confusion_chart(result.confusion[cm_model], f"{cm_model} — confusion matrix"),
                      width="stretch")
    with right:
        st.markdown("**How to read this**")
        st.markdown(
            "- **Precision**: when the model says *up*, how often it was right.\n"
            "- **Recall**: of the days that actually went up, how many the model caught.\n"
            "- **F1**: harmonic mean of precision and recall.\n"
            "- A model that predicts *up* almost every day gets high recall for free in a "
            "rising market — compare against the majority-class row."
        )

with tab_features:
    left, right = st.columns([3, 2])
    imp_model = right.radio("Model", ["XGBoost", "LightGBM", "Random Forest"], horizontal=True)
    top_n = right.slider("Features shown", 5, len(result.features), 15)
    importance = result.feature_importance(imp_model)
    left.plotly_chart(plots.feature_importance_chart(importance, top_n, f"{imp_model} — top {top_n} features"),
                      width="stretch")
    right.dataframe(importance.rename("share").to_frame().style.format("{:.1%}"),
                    width="stretch", height=400)
    with st.expander("Feature correlation heatmap (training window)"):
        st.plotly_chart(plots.correlation_chart(result.train, result.features), width="stretch")
        st.caption("Note how close to zero the `next_return` row is: individually, no feature "
                   "is strongly linearly related to tomorrow's return.")

with tab_preds:
    pred_model = st.selectbox("Model", list(result.models), index=3, key="pred_model")
    preds = result.predictions[pred_model]
    st.plotly_chart(plots.predictions_chart(featured, preds, ticker, pred_model), width="stretch")

    curves = strategy_returns(test, preds)
    st.plotly_chart(plots.equity_curve_chart(curves, "Illustrative: long when model predicts up, else cash"),
                    width="stretch")
    st.caption("No transaction costs, slippage, or taxes are included. This is a sanity check on the "
               "signal, not a backtest you should trade on.")

    table = pd.DataFrame({
        "Close": test["Close"],
        "Actual next-day": test["target"].map({1: "Up", 0: "Down/flat"}),
        "Predicted": preds.map({1: "Up", 0: "Down/flat"}),
        "P(up)": result.probabilities[pred_model],
        "Next-day return": test["next_return"],
    }).sort_index(ascending=False)
    table["Correct"] = table["Actual next-day"] == table["Predicted"]
    st.dataframe(table.style.format({"Close": "${:,.2f}", "P(up)": "{:.2f}", "Next-day return": "{:+.2%}"}),
                 width="stretch", height=360)
    st.download_button("Download predictions (CSV)", table.to_csv().encode(),
                       file_name=f"{ticker}_{pred_model.replace(' ', '_')}_predictions.csv")