# src/evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(actual, predicted):
    """
    Compute MAE and RMSE
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)
    return mae, rmse

def plot_forecasts(test_index, actual, forecast_arima, forecast_prophet, ci=None):
    """
    Plot actual vs forecasts from ARIMA and Prophet
    """
    plt.figure(figsize=(12,6))
    plt.plot(test_index, actual, label='Actual', color='black')
    plt.plot(test_index, forecast_arima, label='ARIMA Forecast', color='blue')
    plt.plot(test_index, forecast_prophet, label='Prophet Forecast', color='green')
    
    # Plot confidence interval for ARIMA if available
    if ci is not None:
        plt.fill_between(test_index, ci.iloc[:,0], ci.iloc[:,1], color='blue', alpha=0.1)
    
    plt.title('Forecast vs Actual Inflation Rate')
    plt.xlabel('Date')
    plt.ylabel('Inflation Rate (%)')
    plt.legend()
    plt.show()
