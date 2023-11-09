import asyncio
from ib_insync import *
from functions.utils.func import *

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------EXECUTE PRICES--------------------------------------------------------------------------------
# Execute get last price functionality
async def exec_price():
    # Get first valid contract
    async def get_contract(symbol):
        contract = Stock(symbol, 'SMART', 'USD')
        contracts = await ib.reqContractDetailsAsync(contract)
        if contracts:
            qualified_contract = contracts[0].contract
            print(f"Obtained qualified contract for {symbol}: {qualified_contract}")
            return qualified_contract
        else:
            print(f"No qualified contract found for {symbol}")
            return None

    # Get the last closing price of a stock
    async def get_market_data(stock):
        print("-" * 60)
        MAX_RETRIES = 10
        SLEEP_DURATION = 1

        for _ in range(MAX_RETRIES):
            market_data = ib.reqMktData(stock, '', False, False)
            await asyncio.sleep(SLEEP_DURATION)
            if market_data.last:
                print(f"Obtained {stock.symbol} last price")
                print("-" * 60)
                return market_data

        print(f"Failed to get market data for {stock.symbol} after {MAX_RETRIES} consecutive calls.")
        print("-" * 60)
        return None

    async def get_last_price(symbol):
        contract = await get_contract(symbol)
        if not contract:
            return symbol, None
        market_data = await get_market_data(contract)
        if market_data and market_data.last:
            return symbol, market_data.last
        return symbol, None


    # Get stocks
    live = True
    past_data = pd.read_parquet(get_parquet_dir(live) / 'data_price.parquet.brotli', columns=['Close'])
    ticker = pd.read_parquet(get_parquet_dir(live) / 'data_ticker.parquet.brotli')
    # Most up to "date" piece of data
    etf_ticker = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']
    past_date = past_data.index.get_level_values('date').max().strftime('%Y-%m-%d')
    ticker_price = past_data.loc[past_data.index.get_level_values('date') == past_date]
    ticker_price = ticker_price.merge(ticker, left_index=True, right_index=True, how='left')
    trade_ticker = ticker_price.ticker.tolist()
    all_stocks = trade_ticker + etf_ticker

    # Create a list of coroutines for each stock's closing price fetch
    tasks = [get_last_price(stock_symbol) for stock_symbol in all_stocks]
    # Run all fetch tasks concurrently and gather the symbol-price tuples
    symbol_price_tuples = await asyncio.gather(*tasks)
    # Filter out any tuples where the price is None (this should be None)
    symbol_price_tuples = [t for t in symbol_price_tuples if t[1] is not None]
    # Create DataFrame
    price_all = pd.DataFrame(symbol_price_tuples, columns=['ticker', 'Close'])
    price_all['date'] = pd.to_datetime(past_date)

    # Separate price data into ETF and Stocks to trade
    etf_data = price_all[price_all['ticker'].isin(etf_ticker)].set_index(['ticker', 'date'])
    trade_data = price_all[price_all['ticker'].isin(trade_ticker)]
    price_data = trade_data.copy(deep=True).set_index(['ticker', 'date'])


    # Add permno to trade_data and set_index
    permno_to_ticker_dict = ticker_price.reset_index(level='date')['ticker'].to_dict()
    trade_data['permno'] = trade_data['ticker'].map(permno_to_ticker_dict)
    trade_data = trade_data.set_index(['permno', 'date']).drop('ticker', axis=1)

    # Add live trade price to the end of the historical price dataset
    live_data = pd.concat([past_date, trade_data], axis=0).sort_index(level=['permno', 'date'])

    # Add live etf price to the end of the historical price dataset
    past_etf = pd.read_parquet(get_parquet_dir(live) / 'data_etf.parquet.brotli', columns=['Close'])
    # Add live trade price to the end of the historical price dataset
    live_etf = pd.concat([past_etf, etf_data], axis=0).sort_index(level=['ticker', 'date'])
    price_etf = live_etf.copy(deep=True)[['Close']]
    # Create returns and unstack dataframe to only have 'date' index and 'ETF ticker' columns
    T = [1]
    live_etf = create_return(live_etf, T)
    live_etf = live_etf.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
    live_etf = live_etf.unstack('ticker').swaplevel(axis=1)
    live_etf.columns = ['_'.join(col).strip() for col in live_etf.columns.values]
    live_etf = live_etf.fillna(0)

    # Disconnect from IB
    ib.disconnect()

    return live_data, live_etf, price_data, price_etf


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
# Execute trades
def exec_trade(long_stock, short_stock, etf_weight, price_data, price_etf):
    # Execute trades
    def create_moc_order(action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MOC"
        order.totalQuantity = quantity
        order.transmit = True
        return order

    # Callback to see if order has been filled
    def order_filled(trade, fill):
        print(f"Order has been filled for {trade.contract.symbol}")
        print(trade.order)
        print(fill)

    # Function to get a specific contract
    def get_contract(symbol):
        contracts = ib.qualifyContracts(Stock(symbol, 'SMART', 'USD'))
        if contracts:
            return contracts[0]
        else:
            print(f"Could not find a unique contract for {symbol}")
            return None

    # Fetch available capital
    print("Fetching available capital...")
    account_info = ib.accountValues()
    for item in account_info:
        if item.tag == 'NetLiquidation':
            capital = float(item.value)
            print(f"Available capital: ${capital}")
            break
    else:
        print("Could not fetch available capital. Exiting...")
        ib.disconnect()
        exit()

    available_capital = capital/2
    etf_stock = etf_weight.columns.tolist()

    order_num = 1
    # Buy long positions
    for stock_symbol in long_stock:
        print("-" * 60)
        capital_per_stock = available_capital * stock_symbol[1]
        stock_price = price_data.loc[price_data.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares

        stock = get_contract(stock_symbol[0])
        print(f"Buying LONG position for: {stock_symbol[0]}")
        action = 'BUY'
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {stock_symbol[0]}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Sell short positions
    for stock_symbol in short_stock:
        print("-" * 60)
        capital_per_stock = available_capital * stock_symbol[1]
        stock_price = price_data.loc[price_data.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares

        stock = get_contract(stock_symbol[0])
        print(f"Selling SHORT position for: {stock_symbol[0]}")
        action = 'SELL'
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {stock_symbol[0]}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Sell short positions
    for stock_symbol in short_stock:
        print("-" * 60)
        capital_per_stock = available_capital * stock_symbol[1]
        stock_price = price_data.loc[price_data.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares

        stock = get_contract(stock_symbol[0])
        print(f"Selling SHORT position for: {stock_symbol[0]}")
        action = 'SELL'
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {stock_symbol[0]}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Buy/Sell ETF positions
    for stock_symbol in etf_stock:
        print("-" * 60)
        weight = etf_weight[stock_symbol][0]
        capital_per_stock = available_capital * abs(weight)
        stock_price = price_etf.loc[price_etf.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares
        stock = get_contract(stock_symbol[0])

        if weight >= 0:
            # Buy position
            print(f"Buying LONG position for: {stock_symbol}")
            action = 'BUY'
            moc_order = create_moc_order(action, num_share)
            print(f"Placing MOC order to {action}: {num_share} of {stock_symbol}")
            trade_moc = ib.placeOrder(stock, moc_order)
            trade_moc.fillEvent += order_filled
            print(f"Order Number: {order_num}")
            order_num += 1
        else:
            # Short position
            print(f"Selling SHORT position for: {stock_symbol}")
            action = 'SELL'
            moc_order = create_moc_order(action, num_share)
            print(f"Placing MOC order to {action}: {num_share} of {stock_symbol}")
            trade_moc = ib.placeOrder(stock, moc_order)
            trade_moc.fillEvent += order_filled
            print(f"Order Number: {order_num}")
            order_num += 1


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------CONNECT------------------------------------------------------------------------------------
print("-------------------------------------------------------------------------------CONNECT------------------------------------------------------------------------------------")
# Connect to IB
print("Attempting to connect to IBKR TWS Workstation...")
ib = IB()
ib.connect(host='127.0.0.1', port=7497, clientId=1512)
print("Connected to IBKR TWS Workstation.")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------EXECUTE PRICES--------------------------------------------------------------------------------
print("----------------------------------------------------------------------------EXECUTE PRICES--------------------------------------------------------------------------------")
# Create an event loop
loop = asyncio.get_event_loop()
# Retrieve live prices
live_data, live_etf, price_data, price_etf = loop.run_until_complete(exec_price())
# Close the loop
loop.close()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------EXECUTE MEAN REVERSION----------------------------------------------------------------------------
print("------------------------------------------------------------------------EXECUTE MEAN REVERSION----------------------------------------------------------------------------")


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
print("-------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------")
exec_trade(long_stock=long_stock, short_stock=short_stock, etf_weight=etf_weight, price_data=price_data, price_etf=price_etf)

ib.disconnect()