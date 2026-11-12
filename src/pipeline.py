"""One call that goes from ticker symbol to trained, evaluated models."""

from src.config import YEARS_OF_HISTORY
from src.data import download_prices
from src.features import build_features
from src.models import TrainResult, train_and_evaluate


def run_pipeline(ticker: str, years: int = YEARS_OF_HISTORY, use_cache: bool = True) -> TrainResult:
    prices = download_prices(ticker, years=years, use_cache=use_cache)
    featured = build_features(prices)
    return train_and_evaluate(ticker, featured)
