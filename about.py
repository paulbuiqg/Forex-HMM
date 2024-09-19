import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from utils import compute_returns, get_time_window, make_exchange_rate_df


RANDOM_STATE = check_random_state(33)


def forecast(X: np.ndarray, scaler: StandardScaler, model: GaussianHMM,
    horizon: int, n_forecast: int) -> np.ndarray:
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    ret_forecast = np.zeros((horizon, X.shape[1], n_forecast))
    for i in range(n_forecast):
        X_scaled_forecast = model.sample(horizon, random_state=RANDOM_STATE,
            currstate=states[-1])[0]
        ret_forecast[:,:,i] = scaler.inverse_transform(X_scaled_forecast)
    return ret_forecast

@st.cache_data
def foret2pri(df: pd.DataFrame, forecast: np.ndarray,
    horizon: int, n_forecast: int) -> np.ndarray:
    """Convert forecasted returns to prices."""
    last_price = df.iloc[-1].values
    last_price_tiled = np.transpose(
        np.tile(last_price, (n_forecast, horizon, 1)), (1, 2, 0))
    return last_price_tiled * np.exp(np.cumsum(forecast, axis=0))


start_date, end_date = get_time_window()
df = make_exchange_rate_df(start_date, end_date)
df.index = df.index.set_names('Date')

df_ret = compute_returns(df)
X = df_ret.values

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('model/hmm.pkl', 'rb') as file:
    hmm = pickle.load(file)

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
horizon = config['horizon']
n_forecast = config['n_forecast']

# Log-return forecast
ret_forecast = forecast(X, scaler, hmm, horizon, n_forecast)
# Price forecast
price_forecast = foret2pri(df, ret_forecast, horizon, n_forecast)


if 'data' not in st.session_state:
    st.session_state['data'] = df
if 'horizon' not in st.session_state:
    st.session_state['horizon'] = horizon
if 'price_forecast' not in st.session_state:
    st.session_state['price_forecast'] = price_forecast
