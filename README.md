**Forex Forecaster**

This is a Streamlit application to forecast foreign exchange (forex) rates using a [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM) based on historical
currency exchange rates. The application provides users with the ability to view 1-day and many-day forecasts for several currency pairs:

- USD/EUR
- GBP/EUR
- JPY/EUR.

The HMM models the multivariate (3-dimensional) forex rate time series. It is trained on the data using [maximum likelihood estimation](https://en.wikipedia.org/wiki/Hidden_Markov_model#Learning).
The number of hidden states (hyperparameter) is tuned using the [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion) (AIC) to prevent overfitting.

The fitted HMM generates probabilistic forecasts through Monte Carlo sampling (parametric bootstrap).

The application URL is https://forex-hmm.streamlit.app.
