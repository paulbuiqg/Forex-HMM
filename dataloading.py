
import requests
import pandas as pd

def get_exchange_rate(api_key, from_currency, to_currency, start_date, end_date):
    # Alpha Vantage API endpoint for Forex Daily data
    url = 'https://www.alphavantage.co/query'

    # Define API parameters
    params = {
        'function': 'FX_DAILY',
        'from_symbol': from_currency,
        'to_symbol': to_currency,
        'apikey': api_key,
        'outputsize': 'full'  # Full will give us all the data
    }

    # Make the request to the API
    response = requests.get(url, params=params)
    data = response.json()

    # Check if the data is valid
    if 'Time Series FX (Daily)' not in data:
        print(f"Error: {data}")
        return None

    # Extract time series data
    time_series = data['Time Series FX (Daily)']

    # Convert the time series to a DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')

    # Rename columns for clarity
    df.columns = ['open', 'high', 'low', 'close']

    # Convert columns to float
    df = df.astype(float)

    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)

    # Filter by the start and end dates
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    return df

# # Example usage
# api_key = 'your_alpha_vantage_api_key'  # Replace with your Alpha Vantage API key
# from_currency = 'EUR'  # Base currency
# to_currency = 'USD'    # Quote currency
# start_date = '2021-01-01'
# end_date = '2021-12-31'

# # Get the exchange rate data
# df = get_exchange_rate(api_key, from_currency, to_currency, start_date, end_date)

# # Display the data
# if df is not None:
#     print(df)
#     # Plot the closing price
#     df['close'].plot(title=f'{from_currency}/{to_currency} Exchange Rate')
#     plt.ylabel('Exchange Rate')
#     plt.show()
