"""
visualization.py
================
High-quality charting utilities for stock-market data.

Covers:
  - Candlestick + volume charts (Plotly)
  - Technical indicator overlays (RSI, Bollinger Bands, MACD)
  - Correlation heat-maps (Seaborn)
  - Return distribution plots
  - Drawdown charts

Usage:
    from src.visualization import StockVisualizer
    viz = StockVisualizer(df, ticker="AAPL")
    viz.plot_candlestick(save=True)
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# Optional – Plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHARTS_DIR = Path(__file__).resolve().parents[1] / "images" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

COLORS = {
    "up": "#26a69a",        # teal-green
    "down": "#ef5350",      # red
    "sma_20": "#f1c40f",    # yellow
    "sma_50": "#e67e22",    # orange
    "sma_200": "#9b59b6",   # purple
    "bb": "#3498db",        # blue
    "volume": "#546e7a",    # slate
}


# ─────────────────────────────────────────────────────────────────────────────
class StockVisualizer:
    """Generate publication-ready charts for a single stock."""

    def __init__(self, df: pd.DataFrame, ticker: str = "STOCK"):
        self.df = df.copy()
        self.ticker = ticker.upper()
        self.df.columns = self.df.columns.str.lower()

    # ── Candlestick (Plotly) ──────────────────────────────────────────────────
    def plot_candlestick(
        self,
        show_bb: bool = True,
        show_volume: bool = True,
        last_n: Optional[int] = None,
        save: bool = True,
        return_fig: bool = False,
    ):
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed – skipping candlestick chart.")
            return

        df = self.df.tail(last_n) if last_n else self.df
        rows = 2 if show_volume else 1
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25] if show_volume else [1],
            vertical_spacing=0.03,
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                increasing_line_color=COLORS["up"],
                decreasing_line_color=COLORS["down"],
                name="OHLC",
            ),
            row=1, col=1,
        )

        # Bollinger Bands
        if show_bb and "bb_upper" in df.columns:
            for band, label in [("bb_upper", "BB Upper"), ("bb_mid", "BB Mid"), ("bb_lower", "BB Lower")]:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[band], name=label,
                               line=dict(color=COLORS["bb"], dash="dot")),
                    row=1, col=1,
                )

        # Volume
        if show_volume and "volume" in df.columns:
            colors = [COLORS["up"] if c >= o else COLORS["down"]
                      for c, o in zip(df["close"], df["open"])]
            fig.add_trace(
                go.Bar(x=df.index, y=df["volume"], name="Volume",
                       marker_color=colors, opacity=0.6),
                row=2, col=1,
            )

        fig.update_layout(
            title=f"{self.ticker} – Candlestick Chart",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700,
        )

        if save:
            path = CHARTS_DIR / f"{self.ticker}_candlestick.html"
            fig.write_html(str(path))
            logger.info("Saved candlestick chart → %s", path)
        if return_fig:
            return fig
        fig.show()


    # ── RSI ───────────────────────────────────────────────────────────────────
    def plot_rsi(self, last_n: Optional[int] = 252, save: bool = True) -> None:
        df = self.df.tail(last_n) if last_n else self.df
        if "rsi" not in df.columns:
            raise ValueError("RSI not found. Run TechnicalAnalysis.add_rsi() first.")

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Close price
        axes[0].plot(df.index, df["close"], color=COLORS["up"], linewidth=1.5)
        axes[0].set_title(f"{self.ticker} – Price & RSI", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Price (USD)")

        # RSI
        axes[1].plot(df.index, df["rsi"], color="#e74c3c", linewidth=1.2)
        axes[1].axhline(70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)")
        axes[1].axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
        axes[1].fill_between(df.index, df["rsi"], 70,
                             where=(df["rsi"] >= 70), interpolate=True,
                             color="red", alpha=0.2)
        axes[1].fill_between(df.index, df["rsi"], 30,
                             where=(df["rsi"] <= 30), interpolate=True,
                             color="green", alpha=0.2)
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("RSI (14)")
        axes[1].legend(loc="upper left", fontsize=8)

        plt.xticks(rotation=30)
        plt.tight_layout()

        if save:
            path = CHARTS_DIR / f"{self.ticker}_rsi.png"
            plt.savefig(path, dpi=150)
            logger.info("Saved RSI chart → %s", path)
        plt.show()

    # ── MACD ──────────────────────────────────────────────────────────────────
    def plot_macd(self, last_n: Optional[int] = 252, save: bool = True) -> None:
        df = self.df.tail(last_n) if last_n else self.df
        required = {"macd", "macd_signal", "macd_hist"}
        if not required.issubset(df.columns):
            raise ValueError("MACD columns not found. Run StockPreprocessor first.")

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(df.index, df["close"], color=COLORS["up"], linewidth=1.5)
        axes[0].set_title(f"{self.ticker} – Price & MACD", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Price (USD)")

        axes[1].plot(df.index, df["macd"], label="MACD", color="#3498db")
        axes[1].plot(df.index, df["macd_signal"], label="Signal", color="#e74c3c")
        colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in df["macd_hist"]]
        axes[1].bar(df.index, df["macd_hist"], color=colors, alpha=0.5, label="Histogram")
        axes[1].axhline(0, color="white", linewidth=0.5)
        axes[1].set_ylabel("MACD")
        axes[1].legend(loc="upper left", fontsize=8)

        plt.xticks(rotation=30)
        plt.tight_layout()

        if save:
            path = CHARTS_DIR / f"{self.ticker}_macd.png"
            plt.savefig(path, dpi=150)
            logger.info("Saved MACD chart → %s", path)
        plt.show()

    # ── Return Distribution ───────────────────────────────────────────────────
    def plot_return_distribution(self, save: bool = True) -> None:
        if "daily_return" not in self.df.columns:
            raise ValueError("daily_return column missing.")

        returns = self.df["daily_return"].dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(returns, bins=80, kde=True, ax=ax,
                     color="#3498db", edgecolor="none")
        ax.axvline(returns.mean(), color="yellow", linestyle="--", label=f"Mean: {returns.mean():.4f}")
        ax.axvline(returns.quantile(0.05), color="red", linestyle="--",
                   label=f"VaR 95%: {returns.quantile(0.05):.4f}")
        ax.set_title(f"{self.ticker} – Daily Return Distribution", fontsize=14)
        ax.set_xlabel("Daily Return")
        ax.legend()
        plt.tight_layout()

        if save:
            path = CHARTS_DIR / f"{self.ticker}_return_dist.png"
            plt.savefig(path, dpi=150)
            logger.info("Saved distribution chart → %s", path)
        plt.show()

    # ── Correlation Heat-map ──────────────────────────────────────────────────
    @staticmethod
    def plot_correlation(corr_matrix: pd.DataFrame, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0, mask=mask,
            linewidths=0.5, ax=ax,
        )
        ax.set_title("Stock Return Correlation Matrix", fontsize=14)
        plt.tight_layout()

        if save:
            path = CHARTS_DIR / "correlation_matrix.png"
            plt.savefig(path, dpi=150)
            logger.info("Saved correlation matrix → %s", path)
        plt.show()
