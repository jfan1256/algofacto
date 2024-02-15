import asyncio

from core.operation import *

from class_live.live_callback import LiveCallback
from class_live.live_order import LiveOrder

class LiveTrade:
    def __init__(self,
                 ibkr_server=None,
                 current_date=None,
                 capital=None,
                 settle_period=None):

        '''
        ibkr_server (ib_sync server): IBKR IB Sync server
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for model training)
        capital (int): Total capital to trade (this is not equal to portfolio cash)
        settle_period (int): IBKR Settlement Period (time it takes cash to settle after buy/sell for reuse)
        '''

        self.ibkr_server = ibkr_server
        self.current_date = current_date
        self.capital = capital
        self.settle_period = settle_period

    # Execute all orders
    async def exec_trade(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------")
        # Create LiveOrder Class
        live_order = LiveOrder(ibkr_server=self.ibkr_server)

        # Fetch available capital
        print("Fetching available capital...")
        account_info = self.ibkr_server.accountValues()
        for item in account_info:
            if item.tag == 'NetLiquidation':
                available_capital = float(item.value)
                settle_capital = self.capital / self.settle_period
                print(f"Total capital to trade (margin): ${self.capital}")
                print(f"Total cash in portfolio: ${available_capital}")
                print(f"Allocated capital (Settlement): ${settle_capital}")
                break

        # Get Stock Data
        ml_ret = pd.read_parquet(get_live_stock() / 'trade_stock_ml_ret.parquet.brotli')
        ml_trend = pd.read_parquet(get_live_stock() / 'trade_stock_ml_trend.parquet.brotli')
        port_iv = pd.read_parquet(get_live_stock() / 'trade_stock_port_iv.parquet.brotli')
        port_id = pd.read_parquet(get_live_stock() / 'trade_stock_port_id.parquet.brotli')
        port_ivm = pd.read_parquet(get_live_stock() / 'trade_stock_port_ivm.parquet.brotli')
        trend_mls = pd.read_parquet(get_live_stock() / 'trade_stock_trend_mls.parquet.brotli')
        mrev_etf = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_etf.parquet.brotli')
        mrev_mkt = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_mkt.parquet.brotli')

        # Load Live Price
        permno_data = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        mrev_etf_hedge_data = pd.read_parquet(get_live_price() / 'data_mrev_etf_hedge_live.parquet.brotli')
        mrev_mkt_hedge_data = pd.read_parquet(get_live_price() / 'data_mrev_mkt_hedge_live.parquet.brotli')
        trend_mls_com_data = pd.read_parquet(get_live_price() / 'data_trend_mls_com_live.parquet.brotli')
        trend_mls_bond_data = pd.read_parquet(get_live_price() / 'data_trend_mls_bond_live.parquet.brotli')
        ml_trend_re_data = pd.read_parquet(get_live_price() / 'data_ml_trend_re_live.parquet.brotli')
        ml_trend_bond_data = pd.read_parquet(get_live_price() / 'data_ml_trend_bond_live.parquet.brotli')

        # Merge data by 'date', 'ticker', 'type' to calculate total weight per type per stock
        stock_data = pd.concat([ml_ret, ml_trend, port_iv, port_id, port_ivm, trend_mls, mrev_etf, mrev_mkt], axis=0)
        stock_data = stock_data.groupby(level=['date', 'ticker', 'type']).sum()
        stock_data = stock_data.loc[stock_data.index.get_level_values('date') == self.current_date]

        # Convert 'type' into positive and negative weights (long and short)
        stock_data['signed_weight'] = np.where(stock_data.index.get_level_values('type') == 'long', stock_data['weight'], -stock_data['weight'])
        # Sum signed weights by 'date' and 'ticker'
        net_weights = stock_data.groupby(['date', 'ticker'])['signed_weight'].sum()
        # Determine the 'type' based on the sign of the net weight
        net_weights = net_weights.reset_index()
        net_weights['type'] = np.where(net_weights['signed_weight'] > 0, 'long', 'short')
        # Assign absolute values to the new weight column
        net_weights['weight'] = net_weights['signed_weight'].abs()
        del net_weights['signed_weight']
        # Create final stock_data dataframe with net weights across tickers (ensures that no stock enters both a long and short position)
        stock_data = net_weights.set_index(['date', 'ticker', 'type'])

        # Merge price data
        permno_data = permno_data.reset_index().set_index(['ticker', 'date'])
        price_data = pd.concat([permno_data, mrev_etf_hedge_data, mrev_mkt_hedge_data, trend_mls_bond_data, trend_mls_com_data, ml_trend_re_data, ml_trend_bond_data], axis=0)
        price_data = price_data.loc[~price_data.index.duplicated(keep='last')]

        # Params (Note: IBKR has an Order Limit of 50 per second)
        tasks = []
        batch_size = 50
        order_num = 1
        nan_tickers = []
        nan_num = 0
        zero_share = []
        # Subscribe the class method to the newOrderEvent
        live_callback = LiveCallback()
        self.ibkr_server.orderStatusEvent += live_callback.order_status_event_handler

        # Execute Trade Orders
        for row in stock_data.itertuples():
            # Row format: Pandas(Index=('date', 'ticker', 'type'), weight=float64)
            ticker = row[0][1]
            type = row[0][2]
            weight = row[1]
            capital_per_stock = settle_capital * weight

            # Fetch Live Price
            try:
                stock_price = price_data.loc[price_data.index.get_level_values('ticker') == ticker, 'Close'].iloc[0]

                if pd.isna(stock_price) or stock_price < 0:
                    nan_num+=1
                    nan_tickers.append(ticker)
                    continue
            except IndexError:
                nan_num+=1
                nan_tickers.append(ticker)
                continue

            # Check for zero share stocks (not enough capital)
            if int(capital_per_stock / stock_price) == 0:
                zero_share.append(ticker)

            # Create orders
            if type == 'long':
                task = live_order._execute_order(stock_price=stock_price, symbol=ticker, action='BUY', capital_per_stock=capital_per_stock, order_num=order_num, weight=weight)
                tasks.append(task)
                order_num += 1
            elif type == 'short':
                task = live_order._execute_order(stock_price=stock_price, symbol=ticker, action='SELL', capital_per_stock=capital_per_stock, order_num=order_num, weight=weight)
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
        print(f"Skipped Orders: {nan_num}")
        print(f"    Cause: no live price data")
        print(f"    Symbols: {', '.join(nan_tickers)}")
        print(f"Zero Share Orders: {len(zero_share)}")
        print(f"    Cause: not enough capital to buy one share (buy 1 by default)")
        print(f"    Symbols: {', '.join(zero_share)}")
        live_callback.display_metric()
