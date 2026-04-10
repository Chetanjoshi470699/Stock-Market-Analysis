"""
analysis.py
===========
Exploratory Data Analysis (EDA) and technical indicator computation.

Includes:
  - Descriptive statistics
  - Correlation analysis
  - RSI, Bollinger Bands, ATR, Stochastic Oscillator
  - Sharpe / Sortino / Calmar ratios
  - Drawdown analysis

Usage:
    from src.analysis import TechnicalAnalysis, PortfolioMetrics
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class TechnicalAnalysis:
    """Add popular technical indicators to an OHLCV DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        self.df.columns = self.df.columns.str.lower()

    # ── All-in-one ────────────────────────────────────────────────────────────
    def add_all_indicators(self) -> pd.DataFrame:
        self.add_rsi()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_stochastic()
        self.add_obv()
        logger.info("All technical indicators added.")
        return self.df

    # ── Indicators ────────────────────────────────────────────────────────────
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index."""
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df["rsi"] = 100 - (100 / (1 + rs))
        return self.df

    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands (upper, middle, lower)."""
        mid = self.df["close"].rolling(period).mean()
        std = self.df["close"].rolling(period).std()
        self.df["bb_upper"] = mid + std_dev * std
        self.df["bb_mid"] = mid
        self.df["bb_lower"] = mid - std_dev * std
        self.df["bb_width"] = (self.df["bb_upper"] - self.df["bb_lower"]) / mid
        self.df["bb_pct"] = (self.df["close"] - self.df["bb_lower"]) / (
            self.df["bb_upper"] - self.df["bb_lower"]
        )
        return self.df

    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """Average True Range."""
        high_low = self.df["high"] - self.df["low"]
        high_prev = (self.df["high"] - self.df["close"].shift()).abs()
        low_prev = (self.df["low"] - self.df["close"].shift()).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        self.df["atr"] = tr.ewm(span=period, adjust=False).mean()
        return self.df

    def add_stochastic(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator (%K and %D)."""
        low_min = self.df["low"].rolling(period).min()
        high_max = self.df["high"].rolling(period).max()
        raw_k = 100 * (self.df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        self.df["stoch_k"] = raw_k.rolling(smooth_k).mean()
        self.df["stoch_d"] = self.df["stoch_k"].rolling(smooth_d).mean()
        return self.df

    def add_obv(self) -> pd.DataFrame:
        """On-Balance Volume."""
        direction = np.sign(self.df["close"].diff()).fillna(0)
        self.df["obv"] = (direction * self.df["volume"]).cumsum()
        return self.df


# ─────────────────────────────────────────────────────────────────────────────
class PortfolioMetrics:
    """
    Compute risk-adjusted performance metrics for a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily percentage returns (e.g., from preprocessing.py).
    risk_free_rate : float
        Annualised risk-free rate (default 5 % / 252 ≈ 0.0002 per day).
    """

    TRADING_DAYS = 252

    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.05):
        self.returns = returns.dropna()
        self.rf_daily = risk_free_rate / self.TRADING_DAYS

    # ── Metrics ───────────────────────────────────────────────────────────────
    def sharpe_ratio(self) -> float:
        excess = self.returns - self.rf_daily
        return float(
            excess.mean() / excess.std() * np.sqrt(self.TRADING_DAYS)
            if excess.std() != 0
            else np.nan
        )

    def sortino_ratio(self) -> float:
        excess = self.returns - self.rf_daily
        downside_std = excess[excess < 0].std()
        return float(
            excess.mean() / downside_std * np.sqrt(self.TRADING_DAYS)
            if downside_std != 0
            else np.nan
        )

    def calmar_ratio(self) -> float:
        annual_return = (1 + self.returns).prod() ** (self.TRADING_DAYS / len(self.returns)) - 1
        max_dd = self.max_drawdown()
        return float(annual_return / abs(max_dd) if max_dd != 0 else np.nan)

    def max_drawdown(self) -> float:
        cum = (1 + self.returns).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        return float(drawdown.min())

    def annualized_return(self) -> float:
        return float((1 + self.returns).prod() ** (self.TRADING_DAYS / len(self.returns)) - 1)

    def annualized_volatility(self) -> float:
        return float(self.returns.std() * np.sqrt(self.TRADING_DAYS))

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "Annualised Return": self.annualized_return(),
                "Annualised Volatility": self.annualized_volatility(),
                "Sharpe Ratio": self.sharpe_ratio(),
                "Sortino Ratio": self.sortino_ratio(),
                "Calmar Ratio": self.calmar_ratio(),
                "Max Drawdown": self.max_drawdown(),
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
def correlation_matrix(data: dict[str, pd.DataFrame], col: str = "close") -> pd.DataFrame:
    """Build a correlation matrix from multiple ticker DataFrames."""
    prices = pd.DataFrame({t: df[col] for t, df in data.items()})
    returns = prices.pct_change()
    return returns.corr()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing import load_processed

    df = load_processed("AAPL")
    ta = TechnicalAnalysis(df)
    df = ta.add_all_indicators()

    metrics = PortfolioMetrics(df["daily_return"])
    print(metrics.summary())
