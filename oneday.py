"""
One-day ahead forecast visualization page
"""


from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


### Functions ###

@st.cache_data
def compute_forecast_stats(forecast: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary statistics for 1-day-ahead exchange rate forecasts."""
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
    """
    Create a histogram of 1-day-ahead forecast values for a given currency
    pair.
    """
    fig = px.histogram(df[pair], x=pair, histnorm='percent', nbins=15)
    return fig

######

price_forecast = st.session_state.price_forecast

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
