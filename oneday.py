import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from utils import compute_returns


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

@st.cache_data
def compute_forecast_stats(forecast: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_ahead = pd.DataFrame(
        columns=['USD/EUR', 'GBP/EUR', 'JPY/EUR'],
        data=forecast[0,:,:].T
    )
    df_stats = pd.DataFrame(
        index=['USD/EUR', 'GBP/EUR', 'JPY/EUR'],
        columns=[
            'Mean',
            'Std',
            '5%-percentile',
            '25%-percentile',
            'Median',
            '75%-percentile',
            '95%-percentile'
        ],
        data=[[
            df_ahead['USD/EUR'].mean(),
            df_ahead['USD/EUR'].std(),
            df_ahead['USD/EUR'].quantile(.05),
            df_ahead['USD/EUR'].quantile(.25),
            df_ahead['USD/EUR'].median(),
            df_ahead['USD/EUR'].quantile(.75),
            df_ahead['USD/EUR'].quantile(.95)
        ], [
            df_ahead['GBP/EUR'].mean(),
            df_ahead['GBP/EUR'].std(),
            df_ahead['GBP/EUR'].quantile(.05),
            df_ahead['GBP/EUR'].quantile(.25),
            df_ahead['GBP/EUR'].median(),
            df_ahead['GBP/EUR'].quantile(.75),
            df_ahead['GBP/EUR'].quantile(.95)
        ], [
            df_ahead['JPY/EUR'].mean(),
            df_ahead['JPY/EUR'].std(),
            df_ahead['JPY/EUR'].quantile(.05),
            df_ahead['JPY/EUR'].quantile(.25),
            df_ahead['JPY/EUR'].median(),
            df_ahead['JPY/EUR'].quantile(.75),
            df_ahead['JPY/EUR'].quantile(.95)
        ]]
    )
    return df_ahead, df_stats

@st.cache_data
def make_one_day_head_forecast_histogram(df: pd.DataFrame, pair: str
    ) -> go.Figure:
    fig = px.histogram(df[pair], x=pair, histnorm='percent', nbins=15)
    return fig


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
horizon = config['horizon']
n_forecast = config['n_forecast']

df = st.session_state.data
df_ret = compute_returns(df)
X = df_ret.values

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('model/hmm.pkl', 'rb') as file:
    hmm = pickle.load(file)

# Log-return forecast
ret_forecast = forecast(X, scaler, hmm, horizon, n_forecast)
# Price forecast
price_forecast = foret2pri(df, ret_forecast, horizon, n_forecast)

df_ahead, df_stats = compute_forecast_stats(price_forecast)

col1, col2, col3 = st.columns(3)

fig1 = make_one_day_head_forecast_histogram(df_ahead, 'USD/EUR')
fig2 = make_one_day_head_forecast_histogram(df_ahead, 'GBP/EUR')
fig3 = make_one_day_head_forecast_histogram(df_ahead, 'JPY/EUR')

with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)

st.write(df_stats.style.format(precision=6))

if 'horizon' not in st.session_state:
    st.session_state['horizon'] = horizon
if 'price_forecast' not in st.session_state:
    st.session_state['price_forecast'] = price_forecast
