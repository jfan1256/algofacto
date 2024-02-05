import asyncio

from core.operation import *

from class_live.live_callback import OrderCounter
from class_order.order_ibkr import OrderIBKR

class LiveTrade:
    def __init__(self,
                 ibkr_server,
                 current_date,
                 settle_period):

        '''
        ibkr_server (ib_sync server): IBKR IB Sync server
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for model training)
        settle_period (int): IBKR Settlement Period (time it takes cash to settle after buy/sell for reuse)
        '''

        self.ibkr_server = ibkr_server
        self.current_date = current_date
        self.settle_period = settle_period

    # Execute all orders
    async def exec_trade(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------")
        # Create OrderIBKR Class
        order_ibkr = OrderIBKR(ibkr_server=self.ibkr_server)

        # Fetch available capital
        print("Fetching available capital...")
        account_info = self.ibkr_server.accountValues()
        for item in account_info:
            if item.tag == 'NetLiquidation':
                available_capital = float(item.value)
                settle_capital = available_capital / self.settle_period
                print(f"Total capital: ${available_capital}")
                print(f"Allocated capital (Settlement): ${settle_capital}")
                break

        # Get Stock Data
        ml_ret = pd.read_parquet(get_live_stock() / 'trade_stock_ml_ret.parquet.brotli')
        ml_trend = pd.read_parquet(get_live_stock() / 'trade_stock_ml_trend.parquet.brotli')
        port_iv = pd.read_parquet(get_live_stock() / 'trade_stock_port_iv.parquet.brotli')
        port_im = pd.read_parquet(get_live_stock() / 'trade_stock_port_im.parquet.brotli')
        port_id = pd.read_parquet(get_live_stock() / 'trade_stock_port_id.parquet.brotli')
        port_ivm = pd.read_parquet(get_live_stock() / 'trade_stock_port_ivm.parquet.brotli')
        trend_mls = pd.read_parquet(get_live_stock() / 'trade_stock_trend_mls.parquet.brotli')
        mrev_etf = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_etf.parquet.brotli')
        mrev_mkt = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_mkt.parquet.brotli')

        # Load Live Price
        permno_data = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        etf_data = pd.read_parquet(get_live_price() / 'data_etf_live.parquet.brotli')
        market_data = pd.read_parquet(get_live_price() / 'data_mkt_live.parquet.brotli')
        bond_data = pd.read_parquet(get_live_price() / 'data_bond_live.parquet.brotli')
        com_data = pd.read_parquet(get_live_price() / 'data_com_live.parquet.brotli')

        # Merge data by 'date', 'ticker', 'type'
        stock_data = pd.concat([ml_ret, ml_trend, port_iv, port_im, port_id, port_ivm, trend_mls, mrev_etf, mrev_mkt], axis=0)
        stock_data = stock_data.groupby(level=['date', 'ticker', 'type']).sum()
        stock_data = stock_data.loc[stock_data.index.get_level_values('date') == self.current_date]

        # Merge price data
        permno_data = permno_data.reset_index().set_index(['ticker', 'date'])
        price_data = pd.concat([permno_data, etf_data, market_data, bond_data, com_data], axis=0)
        price_data = price_data.loc[~price_data.duplicated(keep='last')]

        # Params (Note: IBKR has an Order Limit of 50 per second)
        tasks = []
        batch_size = 50
        order_num = 1
        # Subscribe the class method to the newOrderEvent
        order_counter = OrderCounter()
        self.ibkr_server.orderStatusEvent += order_counter.order_status_event_handler

        # Execute Trade Orders
        for row in stock_data.itertuples():
            # Row format: Pandas(Index=('date', 'ticker', 'type'), weight=float64)
            ticker = row[0][1]
            type = row[0][2]
            weight = row[1]
            stock_price = price_data.loc[price_data.index.get_level_values('ticker') == ticker]['Close'][0]
            capital_per_stock = settle_capital * weight

            # Create orders
            if type == 'long':
                task = order_ibkr._execute_order(stock_price=stock_price, symbol=ticker, action='BUY', capital_per_stock=capital_per_stock, order_num=order_num, weight=weight)
                tasks.append(task)
                order_num += 1
            elif type == 'short':
                task = order_ibkr._execute_order(stock_price=stock_price, symbol=ticker, action='SELL', capital_per_stock=capital_per_stock, order_num=order_num, weight=weight)
                tasks.append(task)
                order_num += 1

            # Execute Batch
            if order_num % batch_size == 0:
                batch_num = int(order_num/batch_size)
                print(f"----------------------------------------------------------------BATCH: {batch_num}----------------------------------------------------------------------------------")
                # Wait for current batch of tasks to complete
                await asyncio.gather(*tasks)
                tasks = []
                # Avoid Order Hit Rate Limit
                time.sleep(2)

        # Ensure any excess tasks are completed (this will only happen if len(stock_data) is not divisible by batch_size
        if tasks:
            print(f"----------------------------------------------------------------BATCH: EXCESS------------------------------------------------------------------------------------------")
            # Wait for current batch of tasks to complete
            await asyncio.gather(*tasks)
            # Ensure the event handler has time to transmit all orders between API and TWS
            await asyncio.sleep(5)

        # Display Order Counts
        print("----------------------------------------------------------------------ORDER METRIC------------------------------------------------------------------------------------------")
        print(f"Total stocks to trade: {len(stock_data)}")
        order_counter.display_metric()
