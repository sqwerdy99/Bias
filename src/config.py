"""Project-wide settings kept in one place so the app and scripts agree."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Liquid US large caps plus one Canadian name (TSX).
TICKERS = ["AAPL", "MSFT", "TSLA", "RY.TO"]

YEARS_OF_HISTORY = 8
TEST_FRACTION = 0.20  # last 20% of rows (chronologically) are held out
RANDOM_STATE = 42
