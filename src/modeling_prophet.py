# src/modeling_prophet.py

from prophet import Prophet
import pandas as pd

def prepare_prophet_df(df):
    """
    Convert df to Prophet format: columns ['ds', 'y']
    """
    return df.reset_index().rename(columns={'date':'ds', 'inflation_rate':'y'})

def train_prophet(train_df):
    """
    Train Prophet model
    """
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)
    return model

def forecast_prophet(model, periods):
    """
    Make future dataframe & forecast
    """
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    return forecast
