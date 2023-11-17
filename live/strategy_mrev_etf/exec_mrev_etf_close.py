import pandas as pd

from functions.utils.func import *
from ib_insync import *
from live.callback import OrderCounter

def exec_mrev_etf_close():
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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------PARAMS-----------------------------------------------------------------------------------
    yesterday_date = (datetime.today() - pd.DateOffset(1)).strftime('%Y-%m-%d')
    long, short, etf = strat_mrev_stocks(get_strategy_mrev_etf(), 'ETF', yesterday_date)

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

    # Close etf positions
    for stock_symbol in etf:
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

        if stock_symbol[1] >= 0:
            print(f"Selling LONG position for: {stock_symbol}")
            action = 'SELL'
            moc_order = create_moc_order(action, abs(position.position))
            print(f"Placing MOC order to {action}: {abs(position.position)} of {stock_symbol}")
            trade_moc = ib.placeOrder(stock, moc_order)
            trade_moc.fillEvent += order_filled
            print(f"Order Number: {order_num}")
            order_num += 1
        else:
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

