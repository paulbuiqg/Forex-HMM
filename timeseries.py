"""
Time series visualization page
"""


import plotly.express as px
import streamlit as st


df = st.session_state.data

fig1 = px.line(df.reset_index(), x='Date', y='USD/EUR')
fig2 = px.line(df.reset_index(), x='Date', y='GBP/EUR')
fig3 = px.line(df.reset_index(), x='Date', y='JPY/EUR')

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

if 'data' not in st.session_state:
    st.session_state['data'] = df
