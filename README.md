# Stock Prediction

This project demonstrates a stock price prediction model using historical stock data. It applies time series analysis, such as calculating moving averages and implementing the ARIMA model, to predict stock prices. The data is fetched from APIs like Yahoo Finance and Alpha Vantage to analyze Tesla (TSLA) and Apple (AAPL) stock prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Modeling and Forecasting](#modeling-and-forecasting)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to analyze stock prices and predict future values using time series models. The project covers:

1) **Data fetching** from Yahoo Finance and Alpha Vantage.
2) **Data analysis**, including moving averages, price changes, and volatility.
3) **Time series modeling** using ARIMA to predict future stock prices.

## Features

* **Data Downloading:** Fetches real-time historical stock data.
* **Moving Averages:** Computes 50-day moving averages to smooth out fluctuations in the stock price.
* **Hourly Stock Data:** Analyzes per-hour stock prices using the Alpha Vantage API.
* **ARIMA Modeling:** Predicts future stock prices using the ARIMA model.
* **Visualization:** Visualizes stock price trends and predictions.


## System Requirements

+ **Python:** 3.6 or higher
+ **Libraries:** 
+ + **yfinance:** 0.1.63 or higher
+ + **matplotlib:** 3.3 or higher
+ + **pandas:** 1.1.3 or higher
+ + **numpy:** 1.19.2 or higher
+ + **statsmodels:** 0.12 or higher
+ + **requests:** 2.24.0 or higher
+ + **altair_viewer:** 0.3.0 or higher


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Guruttamsv/Stock-Prediction.git
cd Stock-Prediction
```
2. Set up a virtual environment (optional but recommended):
```bash
conda create -n stock-prediction python=3.8
conda activate stock-prediction
```
3. Install required packages:
```bash
pip install yfinance matplotlib pandas numpy statsmodels requests altair_viewer
```


## Usage

### 1. Fetching Historical Data:
To fetch stock data for **Tesla (TSLA)** from Yahoo Finance and calculate the 50-day moving average:
```python
import yfinance as yf
import matplotlib.pyplot as plt

df = yf.download("TSLA", start="2010-01-01", end="2023-12-31")
df["MA_50"] = df["Adj Close"].rolling(window=50).mean()

# Plot stock price and moving average
plt.plot(df["Adj Close"], label="TSLA")
plt.plot(df["MA_50"], label="50-day MA")
plt.legend()
plt.show()
```

### 2. Fetching Per Hour Stock Data:
To fetch hourly stock data for Apple (AAPL) using Alpha Vantage:
```python
import requests
import pandas as pd

API_KEY = 'YOUR_API_KEY'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=60min&apikey={API_KEY}&outputsize=full'
response = requests.get(url)
data = response.json()

# Process and visualize hourly closing prices
df = pd.DataFrame(data['Time Series (60min)']).transpose()
df.columns = ['open', 'high', 'low', 'close', 'volume']
df['close'] = df['close'].astype(float)

plt.plot(df.index, df['close'], label='AAPL Closing Price')
plt.show()
```

### 3. ARIMA Modeling:
To forecast future stock prices using the ARIMA model:
```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Train the ARIMA model
model = ARIMA(train['close'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast and evaluate model
predictions = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test['close'], predictions)
print(f"MSE: {mse}")
```

## Data Collection

### Yahoo Finance:
Historical data for **Tesla (TSLA)** is fetched from Yahoo Finance:
```bash
!pip install yfinance
```

### Alpha Vantage:
Hourly data for Apple (AAPL) is fetched using the Alpha Vantage API. To use this, sign up at Alpha Vantage for an API key.
```python
API_KEY = 'YOUR_API_KEY'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=60min&apikey={API_KEY}'

```

## Model and Forecasting

This project utilizes the ARIMA model to forecast future stock prices:

+ **AR Order (p):** The number of lag observations included in the model.
+ **Differencing Order (d):** The number of times the data is differenced to make it stationary.
+ **MA Order (q):** The size of the moving average window.

The model is evaluated using the **Mean Squared Error (MSE)** to assess its accuracy.

## Results

+ **50-day Moving Average:** Smoothing out short-term fluctuations and highlighting long-term trends for TSLA stock.
+ **Hourly Closing Prices:** Visualized for Apple stock with detailed trends in price movements.
+ **ARIMA Model:** Produces stock price forecasts with a low Mean Squared Error (MSE), indicating effective performance for short-term predictions.

The model can be further improved by tuning hyperparameters such as the number of epochs, batch size, and the architecture of the LSTM layers.

## Limitations and Future Work

### Limitations
* **Limited Data:** Only a subset of stock data is used for predictions. A larger dataset may improve the accuracy.
* **Simple ARIMA Model:** More advanced models like LSTM or Prophet could improve predictions, especially for longer time periods.

### Future Work
* **Experiment with different models:** Explore other time series models like **LSTM** or **Facebook** Prophet for more robust forecasting.
* **Include More Features:** Incorporate additional features like trading volume, technical indicators (e.g., RSI, MACD) to enhance the model’s predictive power.
* **Use Larger Datasets:** Collect data from multiple stocks or extend the historical range to train a more generalized model.

## Acknowledgements

* **Yahoo Finance:** For providing stock price data.
* **Alpha Vantage:** For providing hourly stock data via their API.
* **Matplotlib:** For visualization support.
* **Statsmodels:** For providing ARIMA modeling functionality.

