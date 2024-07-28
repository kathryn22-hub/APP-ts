import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Streamlit app title
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Stock Price Forecaster (ARCH/GARCH MODEL) by <a href='https://github.com/kathryn22-hub'>Kathryn Shaju</a></h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <p align="center">
      <a href="https://github.com/DenverCoder1/readme-typing-svg">
        <img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=yellow&size=30&center=true&vCenter=true&width=600&height=100&lines=Stock+Forecasts+Made+Simple!;ARCH/GARCH Model_analyser-1.0;" alt="Typing SVG">
      </a>
    </p>
    """,
    unsafe_allow_html=True
)


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
arch_resid = arch_model_fit.resid
arch_std_resid = arch_model_fit.std_resid

# Fit GARCH model
garch_model_fit = arch_model(returns, vol='Garch').fit(disp='off')
garch_forecast = garch_model_fit.forecast(horizon=forecast_horizon)
garch_cond_vol = garch_forecast.variance.values[-1, :]
garch_resid = garch_model_fit.resid
garch_std_resid = garch_model_fit.std_resid

# Display results

st.subheader('ARCH Model Summary')
st.text(arch_model_fit.summary())

st.subheader('GARCH Model Summary')
st.text(garch_model_fit.summary())

# Plotting the conditional volatility for both models
fig, ax = plt.subplots()
ax.plot(range(1, forecast_horizon+1), np.sqrt(arch_cond_vol), label='ARCH Conditional Volatility')
ax.plot(range(1, forecast_horizon+1), np.sqrt(garch_cond_vol), label='GARCH Conditional Volatility')
ax.set_title('Conditional Volatility Forecast')
ax.set_xlabel('Days')
ax.set_ylabel('Volatility')
ax.legend()
st.pyplot(fig)
 # Plotting the forecast residuals for both models
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(arch_resid, label='ARCH Residuals')
ax[0].plot(arch_std_resid, label='ARCH Standardized Residuals')
ax[0].set_title('ARCH Model Residuals')
ax[0].legend()

ax[1].plot(garch_resid, label='GARCH Residuals')
ax[1].plot(garch_std_resid, label='GARCH Standardized Residuals')
ax[1].set_title('GARCH Model Residuals')
ax[1].legend()
st.pyplot(fig)
