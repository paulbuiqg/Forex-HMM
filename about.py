"""Information page"""


import json


with open('model/training_dates.json', 'r') as file:
    training_dates = json.load(file)
start_date = training_dates['start_date']
end_date = training_dates['end_date']

f"""
This application forecasts foreign exchange (forex) rates using a
[hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM)
based on historical currency exchange rates. The application provides users
with the ability to view 1-day and many-day forecasts for several currency
pairs: 

- USD/EUR
- GBP/EUR
- JPY/EUR.

The HMM models the multivariate (3-dimensional) forex rate time series. It is
trained on the data using [maximum likelihood estimation]
(https://en.wikipedia.org/wiki/Hidden_Markov_model#Learning).
The number of hidden states (hyperparameter) is tuned using the [Akaike information criterion]
(https://en.wikipedia.org/wiki/Akaike_information_criterion) (AIC) to prevent
overfitting.

The fitted HMM generates probabilistic forecasts through Monte Carlo sampling
(parametric bootstrap).

The HMM is trained on data ranging from {start_date} to {end_date}.
"""
