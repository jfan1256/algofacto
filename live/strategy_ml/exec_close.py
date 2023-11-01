import pandas as pd

from functions.utils.func import *
from ib_insync import *

def exec_close(num_stocks):
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

    # Create Market On Open Order
    def create_moo_order(action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.tif = "OPG"
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
    long, short = strat_ml_stocks(yesterday_date, num_stocks)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------EXECUTE ORDERS------------------------------------------------------------------------------
    print("------------------------------------------------------------------------------EXECUTE ORDERS------------------------------------------------------------------------------")
    # Establish connection with IBKR TWS Workstation (7497 is for TWS)
    ib = IB()
    ib.connect(host='127.0.0.1', port=7497, clientId=123)

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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------ALL ORDERS HAVE BEEN EXECUTED----------------------------------------------------------------------
    print("-----------------------------------------------------------------------ALL ORDERS HAVE BEEN EXECUTED----------------------------------------------------------------------")
    # Disconnect when done
    ib.disconnect()

