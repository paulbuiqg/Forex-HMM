"""Train hidden Markov model (HMM) on Forex exchange rate data"""


import json
import pickle
from datetime import datetime
from typing import Tuple

import numpy as np
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from utils import compute_returns, make_exchange_rate_df


RANDOM_STATE = check_random_state(33)


def train_model(X: np.ndarray, max_n_state: int, n_train_init: int
    ) -> Tuple[StandardScaler, GaussianHMM]:
    """Fit scaler and HMM to training data."""
    print('Model training...')
    best_aic = None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for n_state in range(1, max_n_state + 1):
        best_ll = None
        for _ in range(n_train_init):
            model = GaussianHMM(n_components=n_state, random_state=RANDOM_STATE)
            model.fit(X_scaled)
            ll = model.score(X_scaled)
            if not best_ll or best_ll < ll:
                best_ll = ll
                best_model = model
        aic = best_model.aic(X_scaled)
        print(f'# of hidden states: {n_state}, AIC: {aic}')
        if not best_aic or aic < best_aic:
            best_aic = aic
            final_model = best_model
    print('Model training... Done')
    return scaler, final_model


if __name__ == '__main__':

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    start_date = config['start_training_date']
    max_n_state = config['max_n_state']
    n_train_init = config['n_train_init']

    end_date = datetime.strftime(datetime.today(), '%Y-%m-%d')

    df = make_exchange_rate_df(start_date, end_date)
    df_ret = compute_returns(df)
    X = df_ret.values

    scaler, hmm = train_model(X, max_n_state, n_train_init)

    with open('model/training_dates.json', 'w') as file:
        json.dump({'start_date': start_date, 'end_date': end_date}, file)

    with open('model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    with open('model/hmm.pkl', 'wb') as file:
        pickle.dump(hmm, file)
