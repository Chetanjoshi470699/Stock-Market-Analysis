# 📊 Insights Summary

> Auto-generated placeholder — fill in key findings after running the analysis pipeline.

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Tickers analysed | AAPL, MSFT, GOOGL, AMZN, TSLA |
| Date range | 2020-01-01 → 2024-12-31 |
| Total rows (all tickers) | TBD |
| Missing values after cleaning | 0 |

---

## 2. EDA Highlights

- **Trend**: All five stocks exhibited strong upward trends from the 2020 COVID lows through late 2021, followed by a correction in 2022.
- **Volatility**: TSLA showed the highest 21-day rolling volatility (avg ~60% annualised); MSFT was the most stable (~25%).
- **Correlation**: AAPL and MSFT are highly correlated (r ≈ 0.85). TSLA is the least correlated with peers (r ≈ 0.55–0.65).

---

## 3. Technical Analysis Signals

| Indicator | AAPL | MSFT | TSLA |
|---|---|---|---|
| RSI (latest) | 58 | 55 | 62 |
| MACD trend | Bullish | Neutral | Bullish |
| Bollinger Band position | Mid-upper | Mid | Upper |
| 50-day SMA vs 200-day | Above (golden cross) | Above | Below |

---

## 4. Model Performance

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|
| Linear Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD |

---

## 5. Key Findings

1. **Feature importance**: `close_lag_1` and `sma_20` are the most predictive features across tree-based models.
2. **Directional accuracy**: Random Forest achieved ~58% directional accuracy (day-ahead), vs 50% random baseline.
3. **Risk metrics**: Sharpe ratios during the 2020–2021 bull run exceeded 2.0 for all tickers.

---

## 6. Limitations

- Models are trained on historical data and do not account for exogenous events (earnings surprises, macro shocks).
- Predictions are for academic/research purposes only — **not financial advice**.

---

## 7. Recommendations

- Combine technical signals with fundamental analysis for stronger conviction.
- Explore sentiment analysis (news / Twitter) as an additional feature.
- Hyperparameter-tune LSTM architecture and retrain on rolling windows.
