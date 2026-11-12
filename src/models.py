"""Model definitions, time-series split, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, TEST_FRACTION
from src.features import feature_columns


def make_models() -> dict[str, object]:
    """Return fresh, unfitted models.

    Hyperparameters are intentionally conservative (shallow trees, slow learning
    rate, row/column subsampling). Daily returns are extremely noisy, and deep
    trees will happily memorise noise in the training set.
    """
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            importance_type="gain",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=20,
            importance_type="gain",
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def time_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Chronological split: earliest rows train, most recent rows test. No shuffling.

    Shuffling would leak future market regimes into training and inflate scores.
    """
    split = int(len(df) * (1 - test_fraction))
    return df.iloc[:split], df.iloc[split:]


def evaluate(y_true, y_pred, y_prob) -> dict[str, float]:
    """Standard classification metrics. Positive class = 'next day up'."""
    # AUC is undefined when the model outputs a constant (e.g. the dummy baseline).
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


@dataclass
class TrainResult:
    """Everything the CLI and dashboard need after training on one ticker."""

    ticker: str
    features: list[str]
    train: pd.DataFrame
    test: pd.DataFrame
    models: dict[str, object]
    metrics: pd.DataFrame  # one row per model
    predictions: pd.DataFrame  # test-set predictions, one column per model
    probabilities: pd.DataFrame
    confusion: dict[str, np.ndarray] = field(default_factory=dict)

    def feature_importance(self, model_name: str = "XGBoost") -> pd.Series:
        model = self.models[model_name]
        imp = pd.Series(model.feature_importances_, index=self.features)
        return (imp / imp.sum()).sort_values(ascending=False)


def train_and_evaluate(ticker: str, featured: pd.DataFrame) -> TrainResult:
    """Fit every model on the training window and score it on the test window."""
    features = feature_columns(featured)
    train, test = time_split(featured)
    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    models = make_models()
    rows, preds, probs, confusion = {}, {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows[name] = evaluate(y_test, pred, prob)
        preds[name] = pred
        probs[name] = prob
        confusion[name] = confusion_matrix(y_test, pred, labels=[0, 1])

    return TrainResult(
        ticker=ticker,
        features=features,
        train=train,
        test=test,
        models=models,
        metrics=pd.DataFrame(rows).T,
        predictions=pd.DataFrame(preds, index=test.index),
        probabilities=pd.DataFrame(probs, index=test.index),
        confusion=confusion,
    )


def strategy_returns(test: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Cumulative growth of $1: buy-and-hold vs. 'long when the model says up, else cash'.

    Illustrative only: ignores transaction costs, slippage, taxes and the bid/ask
    spread, all of which would erode a daily-rebalanced strategy substantially.
    """
    long_flat = test["next_return"] * predictions.astype(float)
    return pd.DataFrame(
        {
            "Buy & hold": (1 + test["next_return"]).cumprod(),
            "Model long/flat": (1 + long_flat).cumprod(),
        },
        index=test.index,
    )
