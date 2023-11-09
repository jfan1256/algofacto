from ib_insync import IB, Stock, Forex, util
import datetime
from functions.utils.func import *

live = True

past_date = '2023-11-03'
live_data = pd.read_parquet(get_parquet_dir(live) / 'data_price.parquet.brotli')
ticker = pd.read_parquet(get_parquet_dir(live) / 'data_ticker.parquet.brotli')
ticker_price = live_data.loc[live_data.index.get_level_values('date') == past_date]
ticker_price = ticker_price.merge(ticker, left_index=True, right_index=True, how='left')
ticker_price_list = ticker_price.ticker.tolist()

# Get first valid contract
def get_contract(symbol):
    contract = Stock(symbol, 'SMART', 'USD')
    contracts = ib.reqContractDetails(contract)
    if contracts:
        qualified_contract = contracts[0].contract
        print(f"Obtained qualified contract for {symbol}: {qualified_contract}")
        return qualified_contract
    else:
        print(f"No qualified contract found for {symbol}")
        return None

# Function to fetch the close price of a list of tickers at 3:40 PM
def get_close_prices_at_time(ib, tickers, date, time):
    close_prices = {}
    for i, ticker in enumerate(tickers):
        print("-"*60)
        # Define the contract
        contract = get_contract(ticker)

        # Request historical data for the minute at 3:40 PM
        historical_data = ib.reqHistoricalData(
            contract,
            endDateTime=f'{date} {time} America/New_York',  # Corrected date and time with time zone
            durationStr='1800 S',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        # Throttle requests to avoid hitting limits
        ib.sleep(0.0001)  # Adjust the sleep time based on IB's data request limits

        # Extract the close price from the historical data
        if historical_data:
            close_price = historical_data[0].close
            close_prices[ticker] = close_price
            print(f"{i+1} --> Successfully retrieved close price for {ticker} : {close_price}")
        else:
            print(f"{i+1} --> Failed retrieved close price for {ticker} : {close_price}")
            close_prices[ticker] = None

    return close_prices


# Initialize the IB object
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=123)  # Replace with your connection details

# Define the tickers and the time you want the data for
date = '20231106'  # YYYYMMDD format
time = '15:40:00'  # HH:MM:SS format

# Fetch the close prices
close_prices = get_close_prices_at_time(ib, ticker_price_list, date, time)

# Disconnect the session
ib.disconnect()

# Convert the dictionary into a list of tuples
data_tuples = [(ticker, date, price) for ticker, price in close_prices.items()]

# Create a DataFrame with the proper MultiIndex and column
df = pd.DataFrame(data_tuples, columns=['ticker', 'date', 'Close'])
print(df)