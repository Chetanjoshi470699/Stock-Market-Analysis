"""
data_collection.py
==================
Handles fetching stock market data from various sources:
  - Yahoo Finance (via yfinance)
  - Alpha Vantage API (optional)

Usage:
    from src.data_collection import StockDataCollector
    collector = StockDataCollector(tickers=["AAPL", "MSFT", "TSLA"])
    collector.fetch_all(start="2020-01-01", end="2024-12-31")
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
class StockDataCollector:
    """Fetches historical OHLCV data for a list of ticker symbols."""

    def __init__(self, tickers: List[str]):
        self.tickers = [t.upper() for t in tickers]

    # ── Public API ────────────────────────────────────────────────────────────
    def fetch_all(
        self,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        save: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Download data for every ticker and optionally save to CSV."""
        end = end or datetime.today().strftime("%Y-%m-%d")
        results: dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            df = self._fetch_ticker(ticker, start, end, interval)
            if df is not None and not df.empty:
                results[ticker] = df
                if save:
                    self._save(ticker, df)

        logger.info("Finished fetching %d ticker(s).", len(results))
        return results

    def fetch_single(
        self,
        ticker: str,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch data for a single ticker symbol."""
        end = end or datetime.today().strftime("%Y-%m-%d")
        df = self._fetch_ticker(ticker, start, end, interval)
        return df if df is not None else pd.DataFrame()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _fetch_ticker(
        self, ticker: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch via yf.Ticker().history() — more reliable on cloud servers
        than yf.download(). Falls back to yf.download() on failure.
        """
        import time

        for attempt in range(3):
            try:
                logger.info("Downloading %s (%s to %s) attempt %d…", ticker, start, end, attempt + 1)

                # ── Primary: Ticker.history (cloud-friendlier) ──────────────
                t = yf.Ticker(ticker)
                df = t.history(start=start, end=end, interval=interval, auto_adjust=True)

                if df.empty:
                    # ── Fallback: yf.download ────────────────────────────────
                    logger.warning("Ticker.history empty for %s, trying yf.download…", ticker)
                    df = yf.download(
                        ticker,
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=True,
                        progress=False,
                    )

                if df.empty:
                    logger.warning("No data returned for %s on attempt %d.", ticker, attempt + 1)
                    time.sleep(1.5)
                    continue

                df.index.name = "Date"
                df.index = pd.to_datetime(df.index).tz_localize(None)
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                # Drop irrelevant yfinance columns
                df = df.drop(columns=[c for c in ["dividends", "stock splits", "capital gains"] if c in df.columns], errors="ignore")
                df["ticker"] = ticker
                logger.info("Fetched %d rows for %s.", len(df), ticker)
                return df

            except Exception as exc:
                logger.error("Error fetching %s (attempt %d): %s", ticker, attempt + 1, exc)
                time.sleep(1.5)

        logger.error("All attempts failed for %s.", ticker)
        return None

    @staticmethod
    def _save(ticker: str, df: pd.DataFrame) -> None:
        path = RAW_DATA_DIR / f"{ticker}.csv"
        df.to_csv(path)
        logger.info("Saved %s → %s", ticker, path)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Vantage helper (optional – requires API key in .env)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_alpha_vantage(
    ticker: str,
    api_key: str,
    outputsize: str = "full",
) -> pd.DataFrame:
    """
    Fetch daily adjusted data from Alpha Vantage.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. "AAPL").
    api_key : str
        Your Alpha Vantage API key (free tier available at alphavantage.co).
    outputsize : str
        'compact' (last 100 days) or 'full' (up to 20 years).
    """
    import requests  # local import – only needed if Alpha Vantage is used

    BASE_URL = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": outputsize,
        "apikey": api_key,
        "datatype": "json",
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    ts_key = "Time Series (Daily)"
    if ts_key not in payload:
        raise ValueError(f"Unexpected response from Alpha Vantage: {payload}")

    df = pd.DataFrame(payload[ts_key]).T
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.columns = [c.split(". ")[1] for c in df.columns]
    df = df.astype(float).sort_index()
    logger.info("Alpha Vantage: fetched %d rows for %s.", len(df), ticker)
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    collector = StockDataCollector(tickers=TICKERS)
    data = collector.fetch_all(start="2020-01-01")
    print(f"\nDownloaded data for: {list(data.keys())}")
    for t, df in data.items():
        print(f"  {t}: {df.shape[0]} rows, {df.shape[1]} columns")
