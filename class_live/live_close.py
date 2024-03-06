import asyncio

from core.operation import *

from class_live.live_order import LiveOrder
from class_live.live_callback import LiveCallback

class LiveClose:
    def __init__(self,
                 portfolio=None,
                 ibkr_server=None,
                 current_date=None,
                 capital=None
                 ):

        '''
        portfolio (list): List of portfolio strategy class names
        ibkr_server (ib_sync server): IBKR IB Sync server
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for model training)
        capital (int): Total capital to trade (this is not equal to portfolio cash)
        '''

        self.portfolio = portfolio
        self.ibkr_server = ibkr_server
        self.current_date = current_date
        self.capital = capital

    # Execute close orders
    async def exec_close(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE CLOSE ORDERS-----------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE CLOSE ORDERS-----------------------------------------------------------------------------")
        # Create LiveOrder Class
        live_order = LiveOrder(ibkr_server=self.ibkr_server)

        # Fetch available capital
        print("Fetching available capital...")
        account_info = self.ibkr_server.accountValues()
        for item in account_info:
            if item.tag == 'NetLiquidation':
                available_capital = float(item.value)
                print(f"Total capital to trade (margin): ${self.capital}")
                print(f"Total cash in portfolio: ${available_capital}")
                break

        # Get Stock Data
        stock_data = []
        try:
            if 'StratMLRet' in self.portfolio:
                ml_ret = pd.read_parquet(get_live() / 'data_ml_ret_store.parquet.brotli')
                stock_data.append(ml_ret)
            if 'StratMLTrend' in self.portfolio:
                ml_trend = pd.read_parquet(get_live() / 'data_ml_trend_store.parquet.brotli')
                stock_data.append(ml_trend)
            if 'StratPortIV' in self.portfolio:
                port_iv = pd.read_parquet(get_live() / 'data_port_iv_store.parquet.brotli')
                stock_data.append(port_iv)
            if 'StratPortID' in self.portfolio:
                port_id = pd.read_parquet(get_live() / 'data_port_id_store.parquet.brotli')
                stock_data.append(port_id)
            if 'StratPortIM' in self.portfolio:
                port_im = pd.read_parquet(get_live() / 'data_port_im_store.parquet.brotli')
                stock_data.append(port_im)
            if 'StratTrendMLS' in self.portfolio:
                trend_mls = pd.read_parquet(get_live() / 'data_trend_mls_store.parquet.brotli')
                stock_data.append(trend_mls)
            if 'StratMrevETF' in self.portfolio:
                mrev_etf = pd.read_parquet(get_live() / 'data_mrev_etf_store.parquet.brotli')
                stock_data.append(mrev_etf)
            if 'StratMrevMkt' in self.portfolio:
                mrev_mkt = pd.read_parquet(get_live() / 'data_mrev_etf_store.parquet.brotli')
                stock_data.append(mrev_mkt)
        except:
            print("No trades to be closed, which means today is the first day trading")
            return

        # Merge data by 'date', 'ticker', 'type'
        stock_data = pd.concat(stock_data, axis=0)
        stock_data = stock_data.groupby(level=['date', 'ticker', 'type']).sum()

        # Get yesterday's date stocks
        all_dates = stock_data.index.get_level_values('date').unique()
        all_dates = sorted(all_dates)
        yesterday_date = all_dates[-1]
        stock_data = stock_data.loc[stock_data.index.get_level_values('date') == yesterday_date]

        # Params (Note: IBKR has an Order Limit of 50 per second)
        tasks = []
        batch_size = 50
        order_num = 1
        # Subscribe the class method to the newOrderEvent
        live_callback = LiveCallback()
        self.ibkr_server.orderStatusEvent += live_callback.order_status_event_handler

        # Execute Trade Orders
        for row in stock_data.itertuples():
            # Row format: Pandas(Index=('date', 'ticker', 'type'), weight=float64)
            ticker = row[0][1]
            type = row[0][2]

            # Create orders
            if type == 'long':
                task = live_order._execute_close(symbol=ticker, action='SELL', order_num=order_num, instant=False)
                tasks.append(task)
                order_num += 1
            elif type == 'short':
                task = live_order._execute_close(symbol=ticker, action='BUY', order_num=order_num, instant=False)
                tasks.append(task)
                order_num += 1

            # Execute Batch
            if order_num % batch_size == 0:
                batch_num = int(order_num / batch_size)
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
        print(f"Total stocks to close: {len(stock_data)}")
        print(f"Skipped Orders: {len(live_order.skip_close)}")
        print(f"    Cause: no position, which means there was an error in execution the prior day (most likely because IBKR could not find security to execute the trade)")
        print(f"    Symbols: {', '.join(live_order.skip_close)}")
        live_callback.display_metric()