
import pickle
from datetime import datetime, timedelta
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

from utils import compute_returns, get_time_window, make_exchange_rate_df


API_KEY = 'W3BJBJ2JNXO46ZPO'
RANDOM_STATE = check_random_state(33)


#####



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

# @st.cache_data
# def get_time_window() -> Tuple[str, str]:
#     today = datetime.today()
#     end_date = datetime.strftime(today, '%Y-%m-%d')
#     start_date = datetime.strftime(
#         datetime(today.year - 3, today.month, today.day) + timedelta(days=1),
#         '%Y-%m-%d')
#     return start_date, end_date

# @st.cache_data
# def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
#     """Convert prices to (log-)returns."""
#     return pd.DataFrame(
#         data=np.log(df.iloc[:-1].values / df.iloc[1:].values),
#         index=df.iloc[1:].index,
#         columns=df.columns)

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

@st.cache_data
def make_3_day_head_forecast_plot(df: pd.DataFrame, forecast: np.ndarray,
    pair: str, horizon: int) -> go.Figure:
    df.index.names = ['Date']
    forecast_uni = np.squeeze(forecast[:,df.columns == pair,:])
    #
    df_history = df[[pair]].iloc[-14:].reset_index()
    last_date = df.index[-1]
    last_value = df[pair].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(horizon + 1)]
    #
    # dfs_forecast = [pd.DataFrame({
    #     'Date': future_dates,
    #     pair: [last_value] + list(forecast_uni[:,i])
    # }) for i in range(n_forecast)]
    dfs_quantile = [pd.DataFrame({
        'Date': future_dates,
        pair: [last_value] + list(np.quantile(forecast_uni, q=q, axis=1))
    }) for q in [.05, .20, .35, .65, .80, .95]]

    #
    # Last values
    fig1 = px.line(df_history, x='Date', y=pair)
    # # Monte Carlo forecasts
    # figs = [px.line(dfs_forecast[i], x='Date', y=pair)
    #     for i in range(n_forecast)]
    # for f in figs:
    #     f.update_traces(opacity=.1)
    # Quantile forecasts
    fig_q05 = px.line(dfs_quantile[0], x='Date', y=pair)
    fig_q05.update_traces(line_color='green', line_dash='dash', opacity=.75)
    fig_q20 = px.line(dfs_quantile[1], x='Date', y=pair)
    fig_q20.update_traces(line_color='yellow', line_dash='dash', opacity=.75)
    fig_q35 = px.line(dfs_quantile[2], x='Date', y=pair)
    fig_q35.update_traces(line_color='orange', line_dash='dash', opacity=.75)
    fig_q65 = px.line(dfs_quantile[3], x='Date', y=pair)
    fig_q65.update_traces(line_color='orange', line_dash='dash', opacity=.75)
    fig_q80 = px.line(dfs_quantile[4], x='Date', y=pair)
    fig_q80.update_traces(line_color='yellow', line_dash='dash', opacity=.75)
    fig_q95 = px.line(dfs_quantile[5], x='Date', y=pair)
    fig_q95.update_traces(line_color='green', line_dash='dash', opacity=.75)
    #
    fig2 = go.Figure(data=fig1.data + fig_q05.data + fig_q20.data + fig_q35.data
        + fig_q65.data + fig_q80.data + fig_q95.data)
    # for i in range(n_forecast):
    #     fig2 = go.Figure(data=fig2.data + figs[i].data)
    fig2.update_layout(title=pair)
    return fig2

#####


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
max_n_state = config['max_n_state']
n_train_init = config['n_train_init']
horizon = config['horizon']
n_forecast = config['n_forecast']

st.title('Forex Forecaster')

start_date, end_date = get_time_window()
df = make_exchange_rate_df(start_date, end_date)

st.header('Exchange Rates')

st.write(df)

df.index = df.index.set_names('Date')

fig1 = px.line(df.reset_index(), x='Date', y='USD/EUR')
fig2 = px.line(df.reset_index(), x='Date', y='GBP/EUR')
fig3 = px.line(df.reset_index(), x='Date', y='JPY/EUR')

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

df_ret = compute_returns(df)

X = df_ret.values

with open('./model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('./model/hmm.pkl', 'rb') as file:
    hmm = pickle.load(file)

# Log-return forecasts
ret_forecast = forecast(X, scaler, hmm, horizon, n_forecast)
st.text('Forecast computed.')

price_forecast = foret2pri(df, ret_forecast, horizon, n_forecast)

st.header('1-day-ahead Forecast')

df_ahead, df_stats = compute_forecast_stats(price_forecast)

st.write(df_stats.style.format(precision=6))

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

st.header('3-day-ahead Forecast')

fig1 = make_3_day_head_forecast_plot(df, price_forecast, 'USD/EUR', horizon)
fig2 = make_3_day_head_forecast_plot(df, price_forecast, 'GBP/EUR', horizon)
fig3 = make_3_day_head_forecast_plot(df, price_forecast, 'JPY/EUR', horizon)

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
