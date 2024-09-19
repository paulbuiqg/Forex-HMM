import streamlit as st


st.title('Forex Forecaster')

about_page = st.Page('about.py', title='About')
timeseries_page = st.Page('timeseries.py', title='Time Series')
oneday_page = st.Page('oneday.py', title='1-day-ahead Forecast')
threeday_page = st.Page('threeday.py', title='3-day-ahead Forecast')

pg = st.navigation([about_page, timeseries_page, oneday_page, threeday_page])
pg.run()
