from functions.utils.func import *
from ib_insync import *

def exec_trade():
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
    # -------------------------------------------------------------------------------------------PARAMS------------------------------------------------------------------------------
    current_date = datetime.today().strftime('%Y-%m-%d')
    num_stocks = 50
    long, short = strat_ml_stocks(current_date, num_stocks)
    num_share = 1

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------EXECUTE ORDERS------------------------------------------------------------------------------
    print("------------------------------------------------------------------------------EXECUTE ORDERS------------------------------------------------------------------------------")
    # Establish connection with IBKR TWS Workstation (7497 is for TWS)
    ib = IB()
    ib.connect(host='127.0.0.1', port=7497, clientId=123)

    order_num = 1
    # Execute MOO and MOC orders for long positions
    for stock_symbol in long:
        print("-" * 60)
        stock = get_contract(stock_symbol[0])
        print(f"Placing orders for LONG position on: {stock_symbol}")

        # Placing MOO order
        moo_order = create_moo_order('BUY', num_share)
        print(f"Placing MOO order to BUY: {num_share} of {stock_symbol}")
        trade_moo = ib.placeOrder(stock, moo_order)
        trade_moo.fillEvent += order_filled

        # Placing MOC order
        moc_order = create_moc_order('BUY', num_share)
        print(f"Placing MOC order to BUY: {num_share} of {stock_symbol}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Execute MOO and MOC orders for short positions
    for stock_symbol in short:
        print("-" * 60)
        stock = get_contract(stock_symbol[0])
        print(f"Placing orders for SHORT position on: {stock_symbol}")

        # Placing MOO order
        moo_order = create_moo_order('SELL', num_share)
        print(f"Placing MOO order to SELL: {num_share} of {stock_symbol}")
        trade_moo = ib.placeOrder(stock, moo_order)
        trade_moo.fillEvent += order_filled

        # Placing MOC order
        moc_order = create_moc_order('SELL', num_share)
        print(f"Placing MOC order to SELL: {num_share} of {stock_symbol}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------ALL ORDERS HAVE BEEN EXECUTED----------------------------------------------------------------------
    print("-----------------------------------------------------------------------ALL ORDERS HAVE BEEN EXECUTED----------------------------------------------------------------------")
    # Let IB run as it fills orders
    ib.run()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------ALL ORDERS HAVE BEEN FILLED-----------------------------------------------------------------------
    print("------------------------------------------------------------------------ALL ORDERS HAVE BEEN FILLED-----------------------------------------------------------------------")
    # Disconnect when done
    ib.disconnect()

exec_trade()