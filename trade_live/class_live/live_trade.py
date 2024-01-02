import asyncio

from core.operation import *

from callback import OrderCounter
from class_order.order_ibkr import OrderIBKR


class LiveTrade:
    def __init__(self,
                 ibkr_server,
                 current_date):

        '''
        ibkr_server (ib_sync server): IBKR IB Sync server
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for model training)
        '''

        self.ibkr_server = ibkr_server
        self.current_date = current_date

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

        # Execute Trade Orders
        for order_num, row in enumerate(stock_data.itertuples(), start=1):
            ticker = row.index.get_level_values('ticker')
            type = row.index.get_level_values('type')
            weight = row['weight']
            capital_per_stock = available_capital * weight

            if type == 'long':
                task = order_ibkr._execute_order(symbol=ticker, action='BUY', capital_per_stock=capital_per_stock, order_num=order_num)
                tasks.append(task)
            elif type == 'short':
                task = order_ibkr._execute_order(symbol=ticker, action='SELL', capital_per_stock=capital_per_stock, order_num=order_num)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        print(f"----------------------------------------------------Total number of orders placed: {order_counter.new_order_count}/{len(stock_data)}---------------------------------------------------")

