# src/modeling_arima.py

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

def auto_arima_order(series, seasonal=True, m=12):
    """
    Use auto_arima to find best SARIMA order
    """
    stepwise_fit = auto_arima(series, seasonal=seasonal, m=m, trace=True, suppress_warnings=True)
    print(stepwise_fit.summary())
    return stepwise_fit.order, stepwise_fit.seasonal_order

def train_sarima(train_series, order, seasonal_order):
    """
    Train SARIMA model
    """
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

def forecast_sarima(model_fit, steps):
    """
    Forecast with SARIMA model
    """
    forecast_result = model_fit.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()
    return forecast_mean, forecast_ci
