import asyncio

from core.operation import *

from class_order.order_ibkr import OrderIBKR
from class_live.live_callback import OrderCounter

class LiveClose:
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
    async def exec_close(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE CLOSE ORDERS-----------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE CLOSE ORDERS-----------------------------------------------------------------------------")
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
        ml_ret = pd.read_parquet(get_live() / 'data_ml_ret_store.parquet.brotli')
        ml_trend = pd.read_parquet(get_live() / 'data_ml_trend_store.parquet.brotli')
        port_iv = pd.read_parquet(get_live() / 'data_port_iv_store.parquet.brotli')
        port_im = pd.read_parquet(get_live() / 'data_port_im_store.parquet.brotli')
        port_id = pd.read_parquet(get_live() / 'data_port_id_store.parquet.brotli')
        port_ivmd = pd.read_parquet(get_live() / 'data_port_ivmd_store.parquet.brotli')
        trend_mls = pd.read_parquet(get_live() / 'data_port_trend_mls_store.parquet.brotli')
        mrev_etf = pd.read_parquet(get_live() / 'data_port_mrev_etf_store.parquet.brotli')
        mrev_mkt = pd.read_parquet(get_live() / 'data_port_mrev_etf_store.parquet.brotli')

        # Merge data by 'date', 'ticker', 'type'
        stock_data = pd.concat([ml_ret, ml_trend, port_iv, port_im, port_id, port_ivmd, trend_mls, mrev_etf, mrev_mkt], axis=0)
        stock_data = stock_data.groupby(level=['date', 'ticker', 'type']).sum()

        # Get yesterday's date stocks
        all_dates = stock_data.index.get_level_values('date').unique()
        all_dates = sorted(all_dates)
        yesterday_date = all_dates[-2]
        stock_data = stock_data.loc[stock_data.index.get_level_values('date') == yesterday_date]

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

            # Create orders
            if type == 'long':
                task = order_ibkr._execute_close(symbol=ticker, action='SELL', order_num=order_num)
                tasks.append(task)
                order_num += 1
            elif type == 'short':
                task = order_ibkr._execute_close(symbol=ticker, action='BUY', order_num=order_num)
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
        order_counter.display_metric()