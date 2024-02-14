from ib_insync import *

class LiveOrder:
    def __init__(self,
                 ibkr_server):

        self.ibkr_server = ibkr_server

    # Execute MOC order
    @staticmethod
    def _create_moc_order(action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MOC"
        order.totalQuantity = quantity
        order.transmit = True
        return order

    # Execute Market order
    @staticmethod
    def _create_market_order(action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.transmit = True
        return order

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

    # Execute trade order
    async def _execute_order(self, stock_price, symbol, action, capital_per_stock, order_num, weight):
        print("-" * 60)
        print(f"Placing orders for {action} position on: {symbol}")
        zero_share = 0
        stock = await self._get_contract(symbol)

        # Retrieve whole number of shares
        num_share = int(capital_per_stock / stock_price)

        # Buy 1 share if num_share rounds to 0
        if num_share == 0:
            num_share = 1

        # Placing MOC order
        moc_order = self._create_moc_order(action, num_share)
        print(f"Trade: Placing MOC order to {action}: {num_share} of {symbol} (weight={weight}, capital={capital_per_stock}, price={stock_price})")
        self.ibkr_server.placeOrder(stock, moc_order)
        print(f"Order Number: {order_num}")
        print("-" * 60)

    # Execute close order
    async def _execute_close(self, symbol, action, order_num, instant):
        print("-" * 60)
        print(f"Placing orders for {action} position on: {symbol}")
        stock = await self._get_contract(symbol)

        # Get stock's num_share in portfolio
        portfolio = self.ibkr_server.portfolio()
        position = None
        for item in portfolio:
            if item.contract == stock:
                position = item
                break

        # Placing MOC order
        if instant == False:
            moc_order = self._create_moc_order(action, abs(position.position))
            print(f"Close: Placing MOC order to {action}: {abs(position.position)} of {symbol}")
            self.ibkr_server.placeOrder(stock, moc_order)
            print(f"Order Number: {order_num}")
            print("-" * 60)
        else:
            # Placing Market order for immediate execution
            market_order = self._create_market_order(action, abs(position.position))
            print(f"Close: Placing Market order to {action}: {abs(position.position)} of {symbol}")
            self.ibkr_server.placeOrder(stock, market_order)
            print(f"Order Number: {order_num}")
            print("-" * 60)