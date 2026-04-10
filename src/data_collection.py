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
        try:
            logger.info("Downloading %s (%s to %s) …", ticker, start, end)
            
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                session=session
            )
            if df.empty:
                logger.warning("No data returned for %s.", ticker)
                return None
            df.index.name = "Date"
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            df["ticker"] = ticker
            return df
        except Exception as exc:
            logger.error("Error fetching %s: %s", ticker, exc)
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
