import os
import argparse
import logging
from src.data_collection import StockDataCollector
from src.preprocessing import StockPreprocessor
from src.analysis import TechnicalAnalysis, PortfolioMetrics
from src.model import StockPredictor
from src.visualization import StockVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def main(ticker="AAPL", start_date="2020-01-01"):
    logger.info(f"--- Starting End-to-End Analysis for {ticker} ---")

    # 1. Data Collection
    logger.info("1. Collecting Data...")
    collector = StockDataCollector([ticker])
    raw_data = collector.fetch_single(ticker, start=start_date)

    if raw_data.empty:
        logger.error(f"Failed to fetch data for {ticker}. Exiting.")
        return

    # 2. Preprocessing
    logger.info("2. Preprocessing Data...")
    processed_df = StockPreprocessor(raw_data, ticker=ticker).run(save=True)

    # 3. Technical Analysis
    logger.info("3. Performing Technical Analysis...")
    ta = TechnicalAnalysis(processed_df)
    indicators_df = ta.add_all_indicators()

    metrics = PortfolioMetrics(indicators_df["daily_return"])
    logger.info(f"Portfolio Metrics:\n{metrics.summary().to_string()}")

    # 4. Visualization
    logger.info("4. Generating Visualizations...")
    viz = StockVisualizer(indicators_df, ticker=ticker)
    
    # Disable showing the plot blocks so script doesn't pause, only save them.
    # We will override plt.show temporarily or just use the plot_* methods which we can patch.
    import matplotlib.pyplot as plt
    plt.ion() # non-blocking
    
    # We will only save RSI and MACD as examples
    try:
        viz.plot_rsi(save=True)
        viz.plot_return_distribution(save=True)
        plt.close('all')
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    # 5. Machine Learning Prediction Model
    logger.info("5. Training Predictor Model (Random Forest)...")
    predictor = StockPredictor(indicators_df, target_col="close", horizon=1)
    
    logger.info("Training Random Forest Regressor...")
    predictor.train_random_forest()
    
    evaluation = predictor.evaluate()
    logger.info(f"Model Evaluation:\n{evaluation.to_string()}")

    logger.info(f"--- Pipeline Finished Successfully for {ticker}! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete stock analysis pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol to analyze")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    main(ticker=args.ticker.upper(), start_date=args.start)
