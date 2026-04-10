"""
model.py
========
Machine-learning models for stock price prediction.

Models implemented:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor
  - LSTM (via Keras / TensorFlow) – optional, only if TF is installed

Usage:
    from src.model import StockPredictor
    predictor = StockPredictor(df, target_col="close", horizon=1)
    predictor.train_random_forest()
    predictor.evaluate()
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

# Optional heavy dependencies
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.info("XGBoost not installed – xgb model unavailable.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.info("TensorFlow not installed – LSTM model unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
class StockPredictor:
    """
    Train and evaluate regression models that predict future closing price
    (or direction) of a stock.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame (output of StockPreprocessor).
    target_col : str
        Column to predict (default: 'close').
    horizon : int
        Number of days ahead to forecast (default: 1).
    feature_cols : list[str] | None
        Explicit feature list; auto-selected if None.
    test_size : float
        Fraction of data reserved for testing.
    """

    DEFAULT_FEATURES = [
        "open", "high", "low", "close", "volume",
        "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26", "macd", "macd_signal",
        "rsi", "bb_upper", "bb_lower", "bb_width", "bb_pct",
        "atr", "volatility_10d", "volatility_21d",
        "daily_return", "log_return",
        "close_lag_1", "close_lag_2", "close_lag_3",
        "return_lag_1", "return_lag_2",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        horizon: int = 1,
        feature_cols: Optional[list] = None,
        test_size: float = 0.2,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.horizon = horizon
        self.test_size = test_size

        available = [c for c in (feature_cols or self.DEFAULT_FEATURES) if c in df.columns]
        self.feature_cols = available

        self.scaler = MinMaxScaler()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None
        self._prepare_data()

    # ── Data Prep ─────────────────────────────────────────────────────────────
    def _prepare_data(self) -> None:
        df = self.df.dropna(subset=self.feature_cols + [self.target_col])
        df["target"] = df[self.target_col].shift(-self.horizon)
        df = df.dropna(subset=["target"])

        X = df[self.feature_cols].values
        y = df["target"].values

        split = int(len(X) * (1 - self.test_size))
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        logger.info(
            "Dataset split: %d train / %d test samples.", len(self.X_train), len(self.X_test)
        )

    # ── Models ────────────────────────────────────────────────────────────────
    def train_linear(self) -> LinearRegression:
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        logger.info("Linear Regression trained.")
        return self.model

    def train_random_forest(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> RandomForestRegressor:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model.fit(self.X_train, self.y_train)
        logger.info("Random Forest trained (%d trees).", n_estimators)
        return self.model

    def train_gradient_boosting(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
    ) -> GradientBoostingRegressor:
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        self.model.fit(self.X_train, self.y_train)
        logger.info("Gradient Boosting trained.")
        return self.model

    def train_xgboost(self, **kwargs) -> None:
        if not XGB_AVAILABLE:
            raise ImportError("Install xgboost: pip install xgboost")
        params = dict(n_estimators=300, learning_rate=0.05, max_depth=5,
                      subsample=0.8, colsample_bytree=0.8, random_state=42)
        params.update(kwargs)
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False,
        )
        logger.info("XGBoost trained.")
        return self.model

    def train_lstm(
        self,
        lookback: int = 60,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        if not TF_AVAILABLE:
            raise ImportError("Install TensorFlow: pip install tensorflow")

        # Reshape into (samples, timesteps, features)
        def make_sequences(X, y, lookback):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i - lookback:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        Xtr, ytr = make_sequences(self.X_train, self.y_train, lookback)
        Xte, yte = make_sequences(self.X_test, self.y_test, lookback)

        n_features = Xtr.shape[2]
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.model.fit(
            Xtr, ytr,
            validation_data=(Xte, yte),
            epochs=epochs, batch_size=batch_size,
            callbacks=[cb], verbose=1,
        )
        logger.info("LSTM training complete.")
        # Store for evaluate()
        self._lstm_Xte, self._lstm_yte = Xte, yte

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Train a model before evaluating.")

        if TF_AVAILABLE and isinstance(self.model, tf.keras.Model):
            y_pred = self.model.predict(self._lstm_Xte).flatten()
            y_true = self._lstm_yte
        else:
            y_pred = self.model.predict(self.X_test)
            y_true = self.y_test

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        r2 = r2_score(y_true, y_pred)

        metrics = pd.Series({"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R²": r2})
        logger.info("Evaluation metrics:\n%s", metrics.to_string())
        return metrics

    def feature_importance(self) -> pd.Series:
        """Return feature importances (tree-based models only)."""
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not expose feature_importances_.")
        imp = pd.Series(self.model.feature_importances_, index=self.feature_cols)
        return imp.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing import load_processed
    from src.analysis import TechnicalAnalysis

    df = load_processed("AAPL")
    df = TechnicalAnalysis(df).add_all_indicators()

    predictor = StockPredictor(df, target_col="close", horizon=1)
    predictor.train_random_forest()
    print(predictor.evaluate())
    print(predictor.feature_importance().head(10))
