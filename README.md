# Exchange-Specific XGBoost Framework for Commodity Forecasting

**Team 7** | Mitsui & Co. Commodity Prediction Challenge | JNU School of Engineering

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## Overview

A robust machine learning pipeline combining statistical preprocessing with gradient boosting for multi-asset commodity price forecasting across four global exchanges (US, LME, FX, JPX).


## Features

- ✅ **Statistical Testing Framework** - ADF, KPSS, Ljung-Box, normality tests
- ✅ **Log-Normal Transformation** - Variance stabilization and stationarity
- ✅ **ACF-Based Lag Selection** - Data-driven feature engineering
- ✅ **Technical Indicators** - CCI, RSI, MACD, Bollinger Bands
- ✅ **Exchange-Specific Models** - Captures market microstructure
- ✅ **Rolling Window Validation** - Prevents temporal data leakage

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main_pipeline.py --train data/train.csv \
                        --window 7 \
                        --indicator CCI \
                        --max_lag 30
```


## Methodology

1. **Statistical Preprocessing**
   - Log transformation → First-order differencing
   - Stationarity: 31% (raw) → 89% (transformed)

2. **Feature Engineering**
   - Optimal lag selection via ACF (threshold: 2/√n)
   - CCI indicator (12% performance gain)

3. **Modeling**
   - XGBoost with exchange-specific architecture
   - Rolling window (size=500) for temporal validation

4. **Evaluation**
   - RMSE, MAE, R² metrics, Sharpe-like ratio
   - Feature importance analysis

## Results

| Exchange | RMSE   | MAE    | R²   | Improvement |
|----------|--------|--------|------|-------------|
| US       | 0.0103 | 0.0084 | 0.87 | +14.9%      |
| LME      | 0.0112 | 0.0096 | 0.84 | +18.8%      |
| FX       | 0.0098 | 0.0079 | 0.89 | +14.0%      |
| JPX      | 0.0089 | 0.0071 | 0.91 | +18.3%      |
| **Mean** | **0.0100** | **0.0083** | **0.88** | **+16.5%** |

*Improvement vs. unified model approach*

## Requirements

```
Python >= 3.9
numpy >= 1.23.5
pandas >= 1.5.3
xgboost >= 1.7.3
scikit-learn >= 1.2.2
statsmodels >= 0.13.5
matplotlib >= 3.7.1
seaborn >= 0.12.2
scipy >= 1.10.1
```

## Configuration

Customize pipeline parameters in `main_pipeline.py`:

```python
pipeline = CommodityPredictionPipeline(
    rolling_window=7,      # Window size for predictions
    indicator='CCI',       # Technical indicator: CCI/RSI/MACD/BB
    max_lag=30            # Maximum lag for ACF analysis
)
```

## Output Files

**CSV Reports:**
- `statistical_tests_complete.csv` - All statistical test results
- `model_metrics.csv` - Performance metrics by exchange
- `exchange_wise_summary.csv` - Exchange-level analysis

**Documentation:**
- `research_report.txt` - Comprehensive findings
- `executive_summary_statistical_tests.txt` - Statistical summary


## Team

**Team 7** - JNU School of Engineering

- Sanchit Mishra (23/11/EC/034)
- Abhinav Mishra (23/11/EC/035)
- Karan Joshi (23/11/EC/036)
- Sambid Mallick (23/11/EC/037)
- Punit Sulakh (23/11/EE/040)
- Jyoti Sangwan (23/11/EE/041)
- Hemang Joshi (23/11/EE/042)

## Acknowledgments

- Kaggle and Mitsui & Co. for competition organization
- JNU School of Engineering faculty for guidance
- Open-source community (XGBoost, scikit-learn, statsmodels)

---

**Competition:** [Mitsui Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)  
**Contact:** sanchi74_soe@jnu.ac.in
