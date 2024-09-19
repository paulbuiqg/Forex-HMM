import plotly.express as px
import streamlit as st

from utils import get_time_window, make_exchange_rate_df


start_date, end_date = get_time_window()
df = make_exchange_rate_df(start_date, end_date)

df.index = df.index.set_names('Date')

fig1 = px.line(df.reset_index(), x='Date', y='USD/EUR')
fig2 = px.line(df.reset_index(), x='Date', y='GBP/EUR')
fig3 = px.line(df.reset_index(), x='Date', y='JPY/EUR')

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

if 'data' not in st.session_state:
    st.session_state['data'] = df
