import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load data
data = pd.read_csv("sales_data.csv")

# Convert date column to datetime
data["date"] = pd.to_datetime(data["date"])

# Set date as index
data.set_index("date", inplace=True)

# Resample data to monthly
data = data.resample("M").sum()

# Plot original data
plt.plot(data)
plt.title("Monthly Sales")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.show()

# Decompose time series into trend, seasonality and residual
result = seasonal_decompose(data, model="multiplicative")
result.plot()
plt.show()

# Plot autocorrelation and partial autocorrelation plots
plot_acf(data)
plot_pacf(data)
plt.show()

# Fit SARIMA model
model = ARIMA(data, order=(1,0,0), seasonal_order=(1,1,1,12))
model_fit = model.fit()

# Forecast next year's sales
forecast = model_fit.forecast(steps=12)
forecast = pd.DataFrame(forecast[0], columns=["forecast"], index=data.index[-12:])

# Plot original data and forecast
plt.plot(data, label="Original data")
plt.plot(forecast, label="Forecast")
plt.title("Monthly Sales Forecast")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.legend()
plt.show()
