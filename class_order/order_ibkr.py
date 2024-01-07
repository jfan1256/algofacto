import math
import asyncio

from ib_insync import *

class OrderIBKR:
    def __init__(self,
                 ibkr_server):

        self.ibkr_server = ibkr_server

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