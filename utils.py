from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


API_KEY = 'W3BJBJ2JNXO46ZPO'


def get_time_window() -> Tuple[str, str]:
    """Return 3-year time window ending today."""
    today = datetime.today()
    end_date = datetime.strftime(today, '%Y-%m-%d')
    start_date = datetime.strftime(
        datetime(today.year - 3, today.month, today.day) + timedelta(days=1),
        '%Y-%m-%d')
    return start_date, end_date


def get_exchange_rate(api_key: str, from_currency: str, to_currency: str,
    start_date: str, end_date: str) -> pd.DataFrame:
    # Alpha Vantage API endpoint for Forex Daily data
    url = 'https://www.alphavantage.co/query'

    # Define API parameters
    params = {
        'function': 'FX_DAILY',
        'from_symbol': from_currency,
        'to_symbol': to_currency,
        'apikey': api_key,
        'outputsize': 'full'  # Full will give us all the data
    }

    # Make the request to the API
    response = requests.get(url, params=params)
    data = response.json()

    # Check if the data is valid
    if 'Time Series FX (Daily)' not in data:
        print(f"Error: {data}")
        return None

    # Extract time series data
    time_series = data['Time Series FX (Daily)']

    # Convert the time series to a DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')

    # Rename columns for clarity
    df.columns = ['open', 'high', 'low', 'close']

    # Convert columns to float
    df = df.astype(float)

    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)

    # Filter by the start and end dates
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    return df



@st.cache_data
def make_exchange_rate_df(start_date: str, end_date: str) -> pd.DataFrame:
    """Return dataframe with exchange rate timeseries with date as index."""
    df_usd = get_exchange_rate(API_KEY, 'USD', 'EUR', start_date, end_date)
    df_usd = df_usd[['close']].rename(columns={'close': 'USD/EUR'})
    df_gbp = get_exchange_rate(API_KEY, 'GBP', 'EUR', start_date, end_date)
    df_gbp = df_gbp[['close']].rename(columns={'close': 'GBP/EUR'})
    df_jpy = get_exchange_rate(API_KEY, 'JPY', 'EUR', start_date, end_date)
    df_jpy = df_jpy[['close']].rename(columns={'close': 'JPY/EUR'})
    df = pd.concat([df_usd, df_gbp, df_jpy], axis=1)
    df = df.fillna(method='ffill')  # Impute missing values (forward filling)
    df = df.dropna()
    df = df.sort_index()
    return df


@st.cache_data
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to (log-)returns."""
    return pd.DataFrame(
        data=np.log(df.iloc[:-1].values / df.iloc[1:].values),
        index=df.iloc[1:].index,
        columns=df.columns)

