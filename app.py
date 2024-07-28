import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

st.title('Stock Price Time Series Forecasting')

# User inputs for stock ticker symbol, forecast horizon, start date, and end date
ticker = st.text_input('Enter Stock Ticker Symbol', 'AAPL')
forecast_horizon = st.number_input('Enter Forecast Horizon (in days)', min_value=1, max_value=365, value=30)
start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('today'))

# Fetching data from yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Pre-processing
if 'Date' in data.columns:
    data.set_index('Date', inplace=True)
data.index = pd.to_datetime(data.index)
market = data['Adj Close']

# Calculating returns for volatility
returns = 100 * market.pct_change().dropna()

# Fit ARCH model
arch_model_fit = arch_model(returns, vol='ARCH').fit(disp='off')
arch_forecast = arch_model_fit.forecast(horizon=forecast_horizon)
arch_cond_vol = arch_forecast.variance.values[-1, :]

# Fit GARCH model
garch_model_fit = arch_model(returns, vol='Garch').fit(disp='off')
garch_forecast = garch_model_fit.forecast(horizon=forecast_horizon)
garch_cond_vol = garch_forecast.variance.values[-1, :]

# Plotting the conditional volatility for both models
fig, ax = plt.subplots()
ax.plot(range(1, forecast_horizon+1), np.sqrt(arch_cond_vol), label='ARCH Conditional Volatility')
ax.plot(range(1, forecast_horizon+1), np.sqrt(garch_cond_vol), label='GARCH Conditional Volatility')
ax.set_title('Conditional Volatility Forecast')
ax.set_xlabel('Days')
ax.set_ylabel('Volatility')
ax.legend()

# Display results
st.pyplot(fig)
st.subheader('ARCH Model Summary')
st.text(arch_model_fit.summary())

st.subheader('GARCH Model Summary')
st.text(garch_model_fit.summary())
