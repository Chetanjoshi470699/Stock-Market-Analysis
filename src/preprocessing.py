"""
preprocessing.py
================
Cleans raw OHLCV data and engineers common features used in
technical and machine-learning analysis.

Usage:
    from src.preprocessing import StockPreprocessor
    df = StockPreprocessor(df).run()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
class StockPreprocessor:
    """
    Pipeline that cleans and enriches a raw OHLCV DataFrame.

    Steps
    -----
    1. Remove duplicates and sort by date.
    2. Handle missing values (forward-fill, then drop remaining).
    3. Compute log returns and percentage daily returns.
    4. Add rolling statistics (SMA, EMA, volatility).
    5. Add lagged features for ML models.
    6. Save processed file to data/processed/.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "STOCK"):
        self.ticker = ticker.upper()
        self.df = df.copy()

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self, save: bool = True) -> pd.DataFrame:
        """Execute the full preprocessing pipeline."""
        self._validate()
        self._clean()
        self._add_returns()
        self._add_moving_averages()
        self._add_volatility()
        self._add_lags()
        if save:
            self._save()
        logger.info("[%s] Preprocessing complete. Shape: %s", self.ticker, self.df.shape)
        return self.df

    # ── Steps ─────────────────────────────────────────────────────────────────
    def _validate(self) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(self.df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        self.df.columns = self.df.columns.str.lower()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

    def _clean(self) -> None:
        before = len(self.df)
        self.df = self.df[~self.df.index.duplicated(keep="first")]
        self.df.sort_index(inplace=True)
        # Forward-fill gaps (weekends / holidays)
        self.df.ffill(inplace=True)
        self.df.dropna(inplace=True)
        logger.info("[%s] Cleaned: %d → %d rows.", self.ticker, before, len(self.df))

    def _add_returns(self) -> None:
        self.df["daily_return"] = self.df["close"].pct_change()
        self.df["log_return"] = np.log(self.df["close"] / self.df["close"].shift(1))
        self.df["cumulative_return"] = (1 + self.df["daily_return"]).cumprod() - 1

    def _add_moving_averages(
        self,
        sma_windows: tuple = (10, 20, 50, 200),
        ema_windows: tuple = (12, 26),
    ) -> None:
        for w in sma_windows:
            self.df[f"sma_{w}"] = self.df["close"].rolling(window=w).mean()
        for w in ema_windows:
            self.df[f"ema_{w}"] = self.df["close"].ewm(span=w, adjust=False).mean()
        # MACD
        self.df["macd"] = self.df["ema_12"] - self.df["ema_26"]
        self.df["macd_signal"] = self.df["macd"].ewm(span=9, adjust=False).mean()
        self.df["macd_hist"] = self.df["macd"] - self.df["macd_signal"]

    def _add_volatility(self, windows: tuple = (10, 21)) -> None:
        for w in windows:
            self.df[f"volatility_{w}d"] = (
                self.df["log_return"].rolling(window=w).std() * np.sqrt(252)
            )

    def _add_lags(self, n_lags: int = 5) -> None:
        for lag in range(1, n_lags + 1):
            self.df[f"close_lag_{lag}"] = self.df["close"].shift(lag)
            self.df[f"return_lag_{lag}"] = self.df["daily_return"].shift(lag)

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self) -> None:
        path = PROCESSED_DIR / f"{self.ticker}_processed.csv"
        self.df.to_csv(path)
        logger.info("[%s] Saved → %s", self.ticker, path)


# ─────────────────────────────────────────────────────────────────────────────
def load_processed(ticker: str) -> pd.DataFrame:
    """Load a previously processed CSV for the given ticker."""
    path = PROCESSED_DIR / f"{ticker.upper()}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"No processed file found for {ticker}: {path}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_collection import StockDataCollector  # noqa: F401

    collector = StockDataCollector(["AAPL"])
    raw = collector.fetch_single("AAPL", start="2020-01-01")
    processed = StockPreprocessor(raw, ticker="AAPL").run()
    print(processed.tail())
