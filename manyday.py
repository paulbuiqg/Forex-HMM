from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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
    dfs_quantile = [pd.DataFrame({
        'Date': future_dates,
        pair: [last_value] + list(np.quantile(forecast_uni, q=q, axis=1))
    }) for q in [.05, .20, .35, .65, .80, .95]]
    # Last values
    fig1 = px.line(df_history, x='Date', y=pair)
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
    fig2.update_layout(title=pair)
    return fig2


df = st.session_state.data
price_forecast = st.session_state.price_forecast
horizon = st.session_state.horizon

fig1 = make_3_day_head_forecast_plot(df, price_forecast, 'USD/EUR', horizon)
fig2 = make_3_day_head_forecast_plot(df, price_forecast, 'GBP/EUR', horizon)
fig3 = make_3_day_head_forecast_plot(df, price_forecast, 'JPY/EUR', horizon)

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
