# src/eda.py

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def plot_inflation_trend(df):
    """
    Plot inflation rate over time
    """
    plt.figure(figsize=(12,6))
    sns.lineplot(x=df.index, y=df['inflation_rate'])
    plt.title('Kenya Monthly Inflation Rate (2005–2025)')
    plt.ylabel('Inflation Rate (%)')
    plt.xlabel('Date')
    plt.show()

def plot_rolling_statistics(df, window=12):
    """
    Plot rolling mean and std to assess stationarity
    """
    rolling_mean = df['inflation_rate'].rolling(window=window).mean()
    rolling_std = df['inflation_rate'].rolling(window=window).std()

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['inflation_rate'], label='Original')
    plt.plot(df.index, rolling_mean, label=f'{window}-Month Rolling Mean')
    plt.plot(df.index, rolling_std, label=f'{window}-Month Rolling Std')
    plt.legend()
    plt.title('Rolling Statistics')
    plt.show()

def adf_test(df):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(df['inflation_rate'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
