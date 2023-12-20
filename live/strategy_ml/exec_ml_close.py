import pandas as pd

from functions.utils.func import *
from ib_insync import *
from live.callback import OrderCounter

def exec_ml_close(num_stocks):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------MARKET ORDER FUNCTIONS---------------------------------------------------------------------------
    # Create Market On Close Order
    def create_moc_order(action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MOC"
        order.totalQuantity = quantity
        order.transmit = True
        return order

    # Order Fill Callback
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

    # Retrieve stock list from stocks to trade live
    def strat_ml_stocks(target_date, num_stocks):
        filename = Path(get_strategy_ml() / f'trade_stock_{num_stocks}.csv')

        # Read the file
        df = pd.read_csv(filename)
        # Filter based on date
        date_data = df[df['date'] == target_date].squeeze()

        # If no data for the date
        if date_data.empty:
            print("No data for this date")
            return

        # Extract stocks from the columns
        long_cols = [col for col in df.columns if col.startswith('Long_')]
        short_cols = [col for col in df.columns if col.startswith('Short_')]
        long_stocks = date_data[long_cols].dropna().tolist()
        short_stocks = date_data[short_cols].dropna().tolist()
        long_tuples = [ast.literal_eval(item) for item in long_stocks]
        short_tuples = [ast.literal_eval(item) for item in short_stocks]
        return long_tuples, short_tuples

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------PARAMS-----------------------------------------------------------------------------------
    yesterday_date = (datetime.today() - pd.DateOffset(1)).strftime('%Y-%m-%d')
    long, short = strat_ml_stocks(yesterday_date, num_stocks)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------EXECUTE ML CLOSE----------------------------------------------------------------------------
    print("------------------------------------------------------------------------------EXECUTE ML CLOSE----------------------------------------------------------------------------")
    # Connect to IB
    print("Attempting to connect to IBKR TWS Workstation...")
    ib = IB()
    ib.connect(host='127.0.0.1', port=7497, clientId=123)
    print("Connected to IBKR TWS Workstation.")
    print("-" * 60)

    # Subscribe the class method to the newOrderEvent
    order_counter = OrderCounter()
    ib.newOrderEvent += order_counter.new_order_event_handler

    order_num = 1
    # Close long positions
    for stock_symbol in long:
        print("-" * 60)
        stock = get_contract(stock_symbol[0])
        portfolio = ib.portfolio()
        position = None

        for item in portfolio:
            if item.contract == stock:
                position = item
                break

        if not position:
            print(f"No position found for {stock_symbol}")
            continue

        print(f"Selling LONG position for: {stock_symbol}")
        action = 'SELL'
        moc_order = create_moc_order(action, abs(position.position))
        print(f"Placing MOC order to {action}: {abs(position.position)} of {stock_symbol}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Cover short positions
    for stock_symbol in short:
        print("-" * 60)
        stock = get_contract(stock_symbol[0])
        portfolio = ib.portfolio()
        position = None

        for item in portfolio:
            if item.contract == stock:
                position = item
                break

        if not position:
            print(f"No position found for {stock_symbol}")
            continue

        print(f"Buying back SHORT position for: {stock_symbol}")
        action = 'BUY'
        num_share = int(abs(position.position))
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {stock_symbol}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    print(f"----------------------------------------------------Total number of new orders placed: {order_counter.new_order_count}---------------------------------------------------")
    # Disconnect when done
    ib.disconnect()

