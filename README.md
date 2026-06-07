# NVDA Stock Movement Prediction

## Overview
This project applies machine learning to predict the next-day price movement direction of NVIDIA Corporation (NVDA) stock — specifically, whether the closing price will be higher or lower than the previous day's close. Five different algorithms are implemented and compared to evaluate whether technical price data contains exploitable predictive signal.

## Group Members & Contributions
| Name | Algorithm |
|------|-----------|
| Rico Bullock | Random Forest |
| Joy Chen | Logistic Regression |
| Angele He | Decision Trees |
| Blue Knutson | Nueral Network |
| Ubaidillah Mohammad Razali | SVM |

## Data
- **Source:** Yahoo Finance via `yfinance`
- **Ticker:** NVDA
- **Period:** June 2021 – June 2026
- **Instances:** ~1,256 trading days
- **Target variable:** Binary (1 = next-day close higher, 0 = otherwise)

## Features
Engineered from raw OHLCV data:
- Lagged returns (1, 2, 3, 5 days)
- Moving average signals (5-day, 20-day)
- Intraday high-low range
- Rolling volatility (5-day, 10-day)
- Volume ratio
- RSI (14-day)

## Models
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Neural Network

## Results

| Model | Test Accuracy | Test AUC |
|---|---|---|
| Naive Baseline | 0.532 | 0.500 |
| Random Forest (Rico Bullock) | 0.541 | 0.534 |
| Logistic Regression (Joy Chen) | 0.528 | — |
| Decision Tree (Angele He) | TBD | TBD |
| Neural Network (Blue Knutson) | 0.450 | 0.530 |
| SVM (Ubaidillah Mohammad Razali) | TBD | TBD |

## Reproducing Results
1. Clone the repo:
```bash
   git clone https://github.com/bullock11111/nvda-ml-prediction.git
   cd nvda-ml-prediction
```
2. Create the conda environment:
```bash
   conda env create -f environment.yml
   conda activate nvda-ml
```
3. Run the notebook:
```bash
   jupyter notebook notebook.ipynb
```

## Repository Structure
```
nvda-ml-prediction/
├── data/                  # Raw dataset
├── models/                # One .py file per algorithm
├── notebook.ipynb         # Main findings
├── environment.yml        # Conda environment
└── README.md
```
