# src/data_preprocessing.py

import pandas as pd

def load_and_preprocess(filepath):
    """
    Load monthly inflation data and preprocess:
    - Combine Year & Month into datetime
    - Sort and clean
    """
    df = pd.read_csv(filepath)

    # Combine Year & Month into datetime
    df['date'] = pd.to_datetime(df['Month'] + ' ' + df['Year'].astype(str))
    df = df.sort_values('date')

    # Keep only target column
    df = df[['date', '12-Month Inflation']].rename(columns={'12-Month Inflation': 'inflation_rate'})

    # Set date as index
    df = df.set_index('date')

    # Handle missing values if any
    df['inflation_rate'] = df['inflation_rate'].interpolate()

    return df
