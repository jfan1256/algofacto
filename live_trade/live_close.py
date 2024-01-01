import math
import asyncio

from ib_insync import *
from functions.utils.func import *
from callback import OrderCounter

class LiveClose:
    def __init__(self,
                 ibkr_server,
                 current_date):

        self.ibkr_server = ibkr_server
        self.current_date = current_date

    # Execute trades
    @staticmethod
    def _create_moc_order(action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MOC"
        order.totalQuantity = quantity
        order.transmit = True
        return order

    # Callback to see if order has been filled
    @staticmethod
    def _order_filled(trade, fill):
        print(f"Order has been filled for {trade.contract.symbol}")
        print(trade.order)
        print(fill)

    # Get first valid contract
    async def _get_contract(self, symbol):
        contract = Stock(symbol, 'SMART', 'USD')
        contracts = await self.ibkr_server.reqContractDetailsAsync(contract)
        if contracts:
            qualified_contract = contracts[0].contract
            print(f"Obtained qualified contract for {symbol}: {qualified_contract}")
            return qualified_contract
        else:
            print(f"No qualified contract found for {symbol}")
            return None

    # Get the last closing price of a stock
    async def _get_market_data(self, stock):
        print("-" * 60)
        MAX_RETRIES = 10
        SLEEP_DURATION = 3.0

        for _ in range(MAX_RETRIES):
            market_data = self.ibkr_server.reqMktData(stock, '', False, False)
            await asyncio.sleep(SLEEP_DURATION)
            if market_data.last:
                print(f"Obtained {stock.symbol} last price")
                print("-" * 60)
                return market_data

        print(f"Failed to get market data for {stock.symbol} after {MAX_RETRIES} consecutive calls.")
        print("-" * 60)
        return None

    # Execute trade order
    async def _execute_order(self, symbol, action, capital_per_stock, order_num):
        MAX_RETRIES = 20
        WAIT_TIME = 3

        print("-" * 60)
        print(f"Placing orders for {action} position on: {symbol}")
        stock = await self._get_contract(symbol)
        print(f"Requesting market data for {symbol}...")
        # ib.reqMarketDataType(3)

        retries = 0
        stock_price = None
        while retries < MAX_RETRIES:
            market_data = await self._get_market_data(stock)

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

        # Retrieve whole number of shares
        num_share = int(capital_per_stock / stock_price)

        # Placing MOC order
        moc_order = self._create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {symbol}")
        trade_moc = self.ibkr_server.placeOrder(stock, moc_order)
        trade_moc.fillEvent += self._order_filled
        print(f"Order Number: {order_num}")
        print("-" * 60)

    # Execute all orders
    async def exec_trade(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------")
        # Fetch available capital
        print("Fetching available capital...")
        account_info = self.ibkr_server.accountValues()
        for item in account_info:
            if item.tag == 'NetLiquidation':
                available_capital = float(item.value)
                print(f"Available capital: ${available_capital}")
                break

        # Get Stock Data
        ml_ret = pd.read_parquet(get_live_stock() / 'trade_stock_ml_ret.parquet.brotli')
        ml_trend = pd.read_parquet(get_live_stock() / 'trade_stock_ml_trend.parquet.brotli')
        port_iv = pd.read_parquet(get_live_stock() / 'trade_stock_port_iv.parquet.brotli')
        port_im = pd.read_parquet(get_live_stock() / 'trade_stock_port_im.parquet.brotli')
        port_id = pd.read_parquet(get_live_stock() / 'trade_stock_port_id.parquet.brotli')
        port_ivmd = pd.read_parquet(get_live_stock() / 'trade_stock_port_ivmd.parquet.brotli')
        trend_mls = pd.read_parquet(get_live_stock() / 'trade_stock_trend_mls.parquet.brotli')
        mrev_etf = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_etf.parquet.brotli')
        mrev_mkt = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_mkt.parquet.brotli')

        # Merge data by 'date', 'ticker', 'type'
        stock_data = pd.concat([ml_ret, ml_trend, port_iv, port_im, port_id, port_ivmd, trend_mls, mrev_etf, mrev_mkt], axis=0)
        stock_data = stock_data.groupby(level=['date', 'ticker', 'type']).sum()
        stock_data = stock_data.loc[stock_data.index.get_level_values('date') == self.current_date]

        # List to store all the tasks
        tasks = []
        # Subscribe the class method to the newOrderEvent
        order_counter = OrderCounter()
        self.ibkr_server.newOrderEvent += order_counter.new_order_event_handler

        for order_num, row in enumerate(stock_data.itertuples(), start=1):
            ticker = row.index.get_level_values('ticker')
            type = row.index.get_level_values('type')
            weight = row['weight']
            capital_per_stock = available_capital * weight

            if type == 'long':
                task = self._execute_order(symbol=ticker, action='BUY', capital_per_stock=capital_per_stock, order_num=order_num)
                tasks.append(task)
            elif type == 'short':
                task = self._execute_order(symbol=ticker, action='SELL', capital_per_stock=capital_per_stock, order_num=order_num)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        print(f"----------------------------------------------------Total number of orders placed: {order_counter.new_order_count}/{len(stock_data)}---------------------------------------------------")

