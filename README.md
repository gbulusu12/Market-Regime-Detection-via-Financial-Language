# Market-Regime-Detection-via-Financial-Language
Multimodal ML system for market regime detection using financial news sentiment and technical indicators, achieving improved risk-adjusted performance via early signal detection.
# Market Regime Detection via Financial Language

## Overview

This project investigates whether financial news sentiment can improve the early detection of market regime shifts compared to traditional price-based models.

Conventional approaches rely on historical price and volatility data, making them inherently reactive. This work introduces a multimodal framework that combines technical indicators with sentiment extracted from financial news to capture changes in market narratives before they are reflected in prices.

The problem is formulated as a four-class classification task, predicting market regimes defined by direction (bull vs. bear) and volatility (low vs. high).

---

## Problem Statement

Financial markets are influenced not only by numerical indicators but also by narratives and expectations. Traditional models detect regime changes only after they occur in price data.

This project addresses the following question:

Can financial news sentiment, when combined with technical indicators, provide earlier and more accurate detection of market regime shifts?

---

## Dataset

The dataset integrates two primary sources:

- **Market Data (2015–2024)**  
  - S&P 500 index  
  - VIX (volatility index)  
  - 10-Year Treasury yield  
  - Source: Yahoo Finance

- **Financial News Sentiment**  
  - Headlines dataset filtered to align with the same time period  
  - Sentiment scores generated using FinBERT (positive, negative, neutral probabilities)

The final dataset consists of 2,514 trading days with 63 engineered features.

---

## Feature Engineering

### Technical Features (35)
- Returns over multiple horizons (1d, 5d, 10d, 20d)
- Rolling volatility metrics
- RSI, MACD, Bollinger Bands
- Moving averages and price deviations
- VIX-based indicators

### Sentiment Features (28)
- Daily sentiment aggregates
- Lagged sentiment values (1–10 days)
- Rolling averages (7, 20, 30 days)
- Sentiment momentum and volatility
- Interaction features with market indicators

---

## Target Definition

Market regimes are defined using two dimensions:

- **Market Direction**  
  - Bull: 20-day forward return > 0  
  - Bear: 20-day forward return < 0  

- **Volatility Level**  
  - Low: VIX below median  
  - High: VIX above median  

This results in four classes:
- Bull-LowVol
- Bull-HighVol
- Bear-LowVol
- Bear-HighVol

---

## Methodology

### Data Processing
- Chronological train/validation/test split (60/20/20)
- Rolling normalization to avoid lookahead bias
- Forward/backward filling for missing values
- Feature standardization and lag generation

### Models Evaluated
- Logistic Regression
- Random Forest
- XGBoost
- LSTM
- Transformer
- Stacking Ensemble (Random Forest + XGBoost with Logistic Regression meta-learner)

---

## Results

| Model                | Test Accuracy |
|---------------------|--------------|
| Logistic Regression | 58%          |
| Random Forest       | 59.4%        |
| XGBoost             | 66.4%        |
| LSTM                | 33%          |
| Transformer         | 36%          |
| Stacking Ensemble   | 67.79%       |

Key observations:
- Tree-based models significantly outperform deep learning on tabular financial data.
- The stacking ensemble achieves the best performance by combining complementary models.

---

## Feature Importance

SHAP analysis shows that sentiment features contribute approximately 44% of total model importance, comparable to technical indicators.

This indicates that sentiment is not merely correlated with market behavior but provides meaningful predictive signal.

---

## Backtesting

A regime-based portfolio allocation strategy was evaluated against a buy-and-hold benchmark.

### Results

| Metric            | Strategy | Buy & Hold |
|------------------|----------|------------|
| Total Return     | 194.25%  | 194.96%    |
| Volatility       | 13.14%   | 17.82%     |
| Sharpe Ratio     | 0.72     | 0.53       |
| Max Drawdown     | -24.16%  | -33.92%    |

The strategy achieves similar returns with significantly lower risk, improving risk-adjusted performance.

---

## Key Insights

- Sentiment provides a measurable early signal (approximately 5–10 days) before regime transitions.
- Tree-based ensemble methods are more effective than deep learning for structured financial datasets.
- Feature engineering has a larger impact on performance than model complexity.
- Combining multiple data modalities (text and numerical data) improves predictive capability.

---

## Tech Stack

- Python
- scikit-learn
- XGBoost
- TensorFlow
- SHAP
- yfinance
- pandas, numpy

---

## How to Run

```bash
git clone https://github.com/yourusername/market-regime-detection.git
cd market-regime-detection

pip install -r requirements.txt

jupyter notebook notebooks/603_FinalProject.ipynb



## Limitations

Backtesting assumes no transaction costs
Regime definitions depend on chosen thresholds
Class imbalance affects performance on rare regimes
Model performance may degrade over time without retraining

## Future Work
Incorporate macroeconomic indicators (GDP, unemployment, policy rates)
Extend to other asset classes (bonds, commodities, crypto)
Build a real-time prediction API
Integrate earnings call transcripts and long-form text data
Explore online learning for model adaptation

# Gautam Kumar Venkat Sreeram Bulusu
# University of Maryland, College Park
