"""
src package
===========
Stock Market Analysis – source modules.
"""
from src.data_collection import StockDataCollector, fetch_alpha_vantage
from src.preprocessing import StockPreprocessor, load_processed
from src.analysis import TechnicalAnalysis, PortfolioMetrics, correlation_matrix
from src.visualization import StockVisualizer
from src.model import StockPredictor

__all__ = [
    "StockDataCollector",
    "fetch_alpha_vantage",
    "StockPreprocessor",
    "load_processed",
    "TechnicalAnalysis",
    "PortfolioMetrics",
    "correlation_matrix",
    "StockVisualizer",
    "StockPredictor",
]
