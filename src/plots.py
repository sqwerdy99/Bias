"""Charts: interactive Plotly figures for the dashboard, static PNGs for reports/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; we only save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# Palette: one neutral for price, a blue/orange pair for up/down (colour-blind safe;
# red/green is avoided on purpose).
NEUTRAL = "#6b6a66"
UP = "#2a78d6"
DOWN = "#eb6834"
SHADE = "rgba(107,106,102,0.08)"


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


def price_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=prices.index, y=prices["Close"], name="Close",
            line=dict(color=UP, width=2), hovertemplate="$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Adjusted close", tickprefix="$")
    return _base_layout(fig, f"{ticker} — adjusted daily close")


def predictions_chart(featured: pd.DataFrame, preds: pd.Series, ticker: str, model_name: str) -> go.Figure:
    """Price line with the test window shaded and predicted up/down days marked."""
    test = featured.loc[preds.index]
    fig = go.Figure()
    fig.add_vrect(x0=test.index[0], x1=test.index[-1], fillcolor=SHADE, line_width=0,
                  annotation_text="test period", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=featured.index, y=featured["Close"], name="Close",
        line=dict(color=NEUTRAL, width=1.5), hovertemplate="$%{y:,.2f}<extra></extra>",
    ))
    correct = preds.values == test["target"].values
    for label, value, color, symbol in [("Predicted up", 1, UP, "triangle-up"),
                                        ("Predicted down", 0, DOWN, "triangle-down")]:
        mask = preds.values == value
        fig.add_trace(go.Scatter(
            x=test.index[mask], y=test["Close"][mask], mode="markers", name=label,
            marker=dict(color=color, symbol=symbol, size=8, line=dict(width=1, color="white")),
            customdata=np.where(correct[mask], "correct", "wrong"),
            hovertemplate=label + " (%{customdata})<extra></extra>",
        ))
    fig.update_yaxes(title="Adjusted close", tickprefix="$")
    return _base_layout(fig, f"{ticker} — {model_name} next-day predictions", height=480)


def feature_importance_chart(importance: pd.Series, top_n: int = 15, title: str = "") -> go.Figure:
    imp = importance.head(top_n).iloc[::-1]  # largest at the top
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h", marker_color=UP,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(title="Share of total gain", tickformat=".0%")
    fig = _base_layout(fig, title or f"Top {top_n} features", height=max(320, 26 * len(imp)))
    fig.update_layout(hovermode="closest")
    return fig


def confusion_chart(cm: np.ndarray, title: str = "Confusion matrix") -> go.Figure:
    labels = ["Down/flat (0)", "Up (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, colorscale=[[0, "#eaf2fc"], [1, UP]], showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=16),
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} days<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual", autorange="reversed")
    fig = _base_layout(fig, title, height=360)
    fig.update_layout(hovermode="closest")
    return fig


def equity_curve_chart(curves: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col, color in zip(curves.columns, [NEUTRAL, UP]):
        fig.add_trace(go.Scatter(x=curves.index, y=curves[col], name=col,
                                 line=dict(color=color, width=2),
                                 hovertemplate="%{y:.2f}x<extra>" + col + "</extra>"))
    fig.update_yaxes(title="Growth of $1")
    return _base_layout(fig, title, height=380)


def correlation_chart(df: pd.DataFrame, features: list[str]) -> go.Figure:
    corr = df[features + ["next_return"]].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale=[[0, DOWN], [0.5, "#f0efec"], [1, UP]],
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    ))
    fig = _base_layout(fig, "Feature correlation (incl. next-day return)", height=640)
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# Static report images (used by train.py)
# ---------------------------------------------------------------------------

def save_static_reports(result, out_dir: Path) -> None:
    """Write feature-importance and correlation PNGs for one ticker."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = result.ticker.replace(".", "_")
    sns.set_theme(style="whitegrid")

    imp = result.feature_importance("XGBoost").head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, color=UP, ax=ax)
    ax.set(title=f"{result.ticker} — XGBoost feature importance (gain)",
           xlabel="Share of total gain", ylabel="")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_feature_importance.png", dpi=150)
    plt.close(fig)

    corr = result.train[result.features + ["next_return"]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap=sns.blend_palette([DOWN, "#f0efec", UP], as_cmap=True),
                vmin=-1, vmax=1, center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title(f"{result.ticker} — feature correlation (training window)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_correlation.png", dpi=150)
    plt.close(fig)
