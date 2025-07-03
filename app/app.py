# app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))




import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
sys.path.append('./src')  # make sure src folder is accessible

from data_preprocessing import load_and_preprocess
from modeling_arima import train_sarima, forecast_sarima, auto_arima_order
from modeling_prophet import prepare_prophet_df, train_prophet, forecast_prophet
from evaluation import compute_metrics

st.set_page_config(page_title="📊 Kenyan Inflation Forecast Dashboard", layout="wide")


# 🔹 1. Title & Description

st.title("📊 Kenyan Inflation Rates Forecast (2005–2025)")
st.markdown("""
Forecast Kenyan monthly inflation rates using ARIMA and Facebook Prophet models.
Explore trends, compare model accuracy, and download forecasts.
""")


# Load & preprocess data

df = load_and_preprocess('../data/Inflation Rates.csv')

# Split train/test
train = df[:'2023-12-01']
test = df['2024-01-01':]


# 🔹 2. Model Selector

model_choice = st.selectbox(
    "Select forecasting model:",
    options=["ARIMA", "Prophet"]
)


# Prepare forecasts

if model_choice == "ARIMA":
    # Auto ARIMA to get best order
    order, seasonal_order = auto_arima_order(train['inflation_rate'], seasonal=True, m=12)

    # Train SARIMA model
    sarima_model = train_sarima(train['inflation_rate'], order, seasonal_order)

    # Forecast
    forecast_mean, forecast_ci = forecast_sarima(sarima_model, steps=len(test))

    # Compute metrics
    mae, rmse = compute_metrics(test['inflation_rate'], forecast_mean)

    # Build dataframe for display
    forecast_df = pd.DataFrame({
        'Date': test.index,
        'Actual': test['inflation_rate'].values,
        'Forecast': forecast_mean.values,
        'Lower CI': forecast_ci.iloc[:,0].values,
        'Upper CI': forecast_ci.iloc[:,1].values
    })

elif model_choice == "Prophet":
    # Prepare data
    prophet_df = prepare_prophet_df(df)
    train_p = prophet_df[prophet_df['ds'] < '2024-01-01']
    test_p = prophet_df[prophet_df['ds'] >= '2024-01-01']

    # Train Prophet
    prophet_model = train_prophet(train_p)

    # Forecast
    forecast_p = forecast_prophet(prophet_model, periods=len(test))
    forecasted_prophet = forecast_p[-len(test):][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Compute metrics
    mae, rmse = compute_metrics(test_p['y'], forecasted_prophet['yhat'])

    # Build dataframe for display
    forecast_df = pd.DataFrame({
        'Date': forecasted_prophet['ds'],
        'Actual': test_p['y'].values,
        'Forecast': forecasted_prophet['yhat'].values,
        'Lower CI': forecasted_prophet['yhat_lower'].values,
        'Upper CI': forecasted_prophet['yhat_upper'].values
    })


# 🔹 3. Inflation Trend Chart

st.subheader("📈 Actual vs Forecast Inflation Rates")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Actual'],
    mode='lines', name='Actual',
    line=dict(color='black')
))
fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Forecast'],
    mode='lines', name='Forecast',
    line=dict(color='green')
))
fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Upper CI'],
    mode='lines', name='Upper CI',
    line=dict(width=0), showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Lower CI'],
    mode='lines', name='Lower CI',
    fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
    line=dict(width=0), showlegend=False
))
fig.update_layout(height=500, margin=dict(t=20, b=20))

st.plotly_chart(fig, use_container_width=True)


# 🔹 4. Forecast Table

st.subheader("📋 Forecast Table")
st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}", "Lower CI": "{:.2f}", "Upper CI": "{:.2f}"}))


# 🔹 5. Model Accuracy Metrics

st.subheader("✅ Model Accuracy")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")


# 🔹 6. Data Explorer (Optional)

with st.expander("🧪 Explore Raw Data"):
    st.dataframe(df.reset_index().rename(columns={'date':'Date', 'inflation_rate':'Inflation Rate'}))


# 🔹 7. Download Forecast as CSV (Optional)

csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Forecast as CSV",
    data=csv,
    file_name=f'{model_choice.lower()}_forecast.csv',
    mime='text/csv',
)


# 🧠 Sidebar: Methodology & assumptions

with st.sidebar:
    st.markdown("## 🧠 Methodology & Assumptions")
    st.markdown("""
- Models trained on data: 2005–2023
- Tested on last 24 months (2024–2025)
- Metrics: MAE & RMSE
- Prophet captures multiple seasonalities, ARIMA captures linear trends + seasonality
- Data source: World Bank / local statistics
""")
