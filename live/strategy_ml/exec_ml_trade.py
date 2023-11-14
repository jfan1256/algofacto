from ib_insync import *
from functions.utils.func import *
from live.callback import OrderCounter
import math
import asyncio


# Execute trades
async def exec_ml_trade(num_stocks, settlement, capital):
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
        SLEEP_DURATION = 3.0

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

    # Execute trade order
    async def execute_order(symbol, action, capital_per_stock, order_num):
        MAX_RETRIES = 20
        WAIT_TIME = 3.0 # Time in seconds

        print("-" * 60)
        print(f"Placing orders for {action} position on: {symbol}")
        stock = await get_contract(symbol)
        print(f"Requesting market data for {symbol}...")
        # ib.reqMarketDataType(3)

        retries = 0
        stock_price = None
        while retries < MAX_RETRIES:
            market_data = await get_market_data(stock)

            if not market_data or not market_data.last or math.isnan(market_data.last):
                retries += 1

                print(f"Attempt {retries} failed to fetch valid price for {symbol}. Retrying...")
                print("-" * 60)
                await asyncio.sleep(WAIT_TIME)
            else:
                stock_price = market_data.last
                break

        if stock_price is None:
            print(f"Failed to get valid price for {symbol} after {MAX_RETRIES} attempts. Skipping order.")
            print("-" * 60)
            return

        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares

        # Placing MOC order
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {symbol}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        print("-" * 60)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
    print("-------------------------------------------------------------------------EXECUTE TRADE ORDERS------------------------------------------------------------------------------")
    # Connect to IB
    print("Attempting to connect to IBKR TWS Workstation...")
    ib = IB()
    current_date = datetime.today().strftime('%Y-%m-%d')
    long, short = strat_ml_stocks(current_date, num_stocks)
    await ib.connectAsync(host='127.0.0.1', port=7497, clientId=1512)
    print("Connected to IBKR TWS Workstation.")
    print("-" * 60)

    # Fetch available capital
    print("Fetching available capital...")
    account_info = ib.accountValues()
    for item in account_info:
        if item.tag == 'NetLiquidation':
            available_capital = float(item.value)
            print(f"Available capital: ${available_capital}")
            break
    else:
        print("Could not fetch available capital. Exiting...")
        ib.disconnect()
        exit()

    # Calculations for EWP
    available_capital = available_capital * capital
    capital_per_stock = (available_capital / settlement) / num_stocks
    order_num = 1
    all_stocks = long + short

    # List to store all the tasks
    tasks = []
    # Subscribe the class method to the newOrderEvent
    order_counter = OrderCounter()
    ib.newOrderEvent += order_counter.new_order_event_handler

    for i, stock_symbol in enumerate(all_stocks, start=order_num):
        if stock_symbol in long:
            task = execute_order(stock_symbol[0], 'BUY', capital_per_stock, i)
            tasks.append(task)
        elif stock_symbol in short:  # Changed to elif to avoid possible duplication
            task = execute_order(stock_symbol[0], 'SELL', capital_per_stock, i)
            tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    print(f"----------------------------------------------------Total number of new orders placed: {order_counter.new_order_count}---------------------------------------------------")
    ib.disconnect()


# asyncio.run(exec_trade(num_stocks=50, settlement=3))

