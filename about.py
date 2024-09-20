"""Information page"""


import json


training_dates = json.loads('model/training_dates.json')
start_date = training_dates['start_date']
end_date = training_dates['end_date']


f"""
Blablabla
*bla*


The HMM is trained with data from {start_date} to {end_date}.
"""