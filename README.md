# 📈 Stock Market Analysis

> End-to-end stock market analysis pipeline: data ingestion → cleaning → EDA → technical analysis → ML prediction.

---

## 🗂️ Project Structure

```
stock-market-analysis/
│
├── data/
│   ├── raw/                  # Original datasets (CSV, yfinance downloads)
│   └── processed/            # Cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_data_collection.ipynb      # Fetching data via yfinance / Alpha Vantage
│   ├── 02_data_cleaning.ipynb        # Preprocessing & feature engineering
│   ├── 03_eda_analysis.ipynb         # Exploratory data analysis
│   ├── 04_technical_analysis.ipynb   # RSI, Bollinger Bands, MACD, etc.
│   └── 05_prediction_model.ipynb     # ML models & evaluation
│
├── src/
│   ├── data_collection.py    # StockDataCollector class
│   ├── preprocessing.py      # StockPreprocessor class
│   ├── analysis.py           # TechnicalAnalysis & PortfolioMetrics
│   ├── visualization.py      # StockVisualizer (Plotly + Matplotlib)
│   └── model.py              # StockPredictor (RF, XGBoost, LSTM)
│
├── dashboards/
│   └── powerbi_dashboard.pbix
│
├── reports/
│   ├── final_report.pdf
│   └── insights_summary.md
│
├── images/
│   ├── charts/
│   └── dashboard_screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/stock-market-analysis.git
cd stock-market-analysis
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. (Optional) Configure API keys

Create a `.env` file in the project root:

```
ALPHA_VANTAGE_API_KEY=your_key_here
```

### 3. Fetch stock data

```python
from src.data_collection import StockDataCollector

collector = StockDataCollector(["AAPL", "MSFT", "TSLA"])
data = collector.fetch_all(start="2020-01-01")
```

### 4. Preprocess & engineer features

```python
from src.preprocessing import StockPreprocessor

df = StockPreprocessor(data["AAPL"], ticker="AAPL").run()
```

### 5. Add technical indicators

```python
from src.analysis import TechnicalAnalysis

df = TechnicalAnalysis(df).add_all_indicators()
```

### 6. Visualise

```python
from src.visualization import StockVisualizer

viz = StockVisualizer(df, ticker="AAPL")
viz.plot_candlestick()
viz.plot_rsi()
```

### 7. Train a prediction model

```python
from src.model import StockPredictor

predictor = StockPredictor(df, target_col="close", horizon=1)
predictor.train_random_forest()
print(predictor.evaluate())
```

### 8. Run notebooks

```bash
jupyter notebook
```

Open `notebooks/01_data_collection.ipynb` and run all cells sequentially.

---

## 🧠 Techniques & Models

| Category | Methods |
|---|---|
| **Technical Indicators** | SMA, EMA, MACD, RSI, Bollinger Bands, ATR, Stochastic, OBV |
| **Feature Engineering** | Log returns, volatility, lagged features, cumulative returns |
| **ML Models** | Linear Regression, Random Forest, Gradient Boosting, XGBoost |
| **Deep Learning** | LSTM (TensorFlow / Keras) |
| **Metrics** | MAE, RMSE, MAPE, R², Sharpe, Sortino, Calmar, Max Drawdown |

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for the full list.

Main packages: `yfinance`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `plotly`, `matplotlib`, `seaborn`

---

## 📄 License

MIT License – feel free to use, modify, and share.
