import asyncio
import os

import pandas as pd
from ib_insync import *
from core.operation import *

class LivePrice:
    def __init__(self,
                 ibkr_server,
                 current_date):

        '''
        ibkr_server (ib_sync server): IBKR IB Sync server
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for model training)
        '''

        self.ibkr_server = ibkr_server
        self.current_date = current_date


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
        SLEEP_DURATION = 5

        for _ in range(MAX_RETRIES):
            market_data = self.ibkr_server.reqMktData(stock, '', False, False)
            await asyncio.sleep(SLEEP_DURATION)
            if market_data.last:
                print(f"Obtained {stock.symbol} last price: {market_data.last}")
                print("-" * 60)
                self.ibkr_server.cancelMktData(stock)
                return market_data.last

        print(f"Failed to get market data for {stock.symbol} after {MAX_RETRIES} consecutive calls.")
        print("-" * 60)
        self.ibkr_server.cancelMktData(stock)
        return None

    # Get most recent closing price
    async def _get_last_price(self, symbol):
        contract = await self._get_contract(symbol)
        if not contract:
            return symbol, None
        last_price = await self._get_market_data(contract)
        print(f"Saving {symbol} last price: {last_price}")
        print("-" * 60)
        return symbol, last_price

    # Split the all_stocks list into chunks of batch_size
    async def _get_prices_in_batches(self, all_stocks, batch_size):
        batches = [all_stocks[i:i + batch_size] for i in range(0, len(all_stocks), batch_size)]
        symbol_price_tuples = []
        for i, batch in enumerate(batches, 1):
            print(f"----------------------------------------------------------------BATCH: {i}/{len(batches)}---------------------------------------------------------------------------")
            tasks = [self._get_last_price(stock_symbol) for stock_symbol in batch]
            batch_results = await asyncio.gather(*tasks)
            # Filter and extend the main list
            symbol_price_tuples.extend([t for t in batch_results if t[1] is not None])
            # Sleep to avoid hitting rate limits, if necessary
            await asyncio.sleep(1)
        return symbol_price_tuples

    # Get live prices and export the data for easy access
    async def exec_live_price(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE LIVE PRICES------------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE LIVE PRICES------------------------------------------------------------------------------")
        # All Permno Ticker
        live = True
        historical_data = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli', columns=['Close'])
        all_ticker = pd.read_parquet(get_parquet(live) / 'data_ticker.parquet.brotli')
        latest_date = historical_data.index.get_level_values('date').max().strftime('%Y-%m-%d')
        latest_data = historical_data.loc[historical_data.index.get_level_values('date') == latest_date]
        latest_data = latest_data.merge(all_ticker, left_index=True, right_index=True, how='left')
        permno_ticker = latest_data.ticker.tolist()

        # MREV ETF Ticker
        mrev_etf_hedge_ticker = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']

        # MREV Market Ticker
        mrev_mkt_hedge_ticker = ['SPY', 'MDY', 'VEA', 'EEM', 'VNQ', 'DBC']

        # Trend MLS Commodity Ticker
        trend_mls_com_ticker = ['GLD', 'SLV', 'PDBC', 'USO', 'AMLP']

        # Trend MLS Bond Ticker
        trend_mls_bond_ticker = ['LQD', 'HYG', 'TLT', 'BNDX', 'MUB']

        # ML Trend Real Estate Ticker
        ml_trend_re_ticker = ['VNQ', 'IYR', 'SCHH', 'RWR', 'USRT']

        # ML Trend Bond Ticker
        ml_trend_bond_ticker = ['LQD', 'HYG', 'TLT', 'BNDX', 'MUB']

        # Combine all tickers
        all_stocks = permno_ticker + mrev_etf_hedge_ticker + mrev_mkt_hedge_ticker + trend_mls_com_ticker + trend_mls_bond_ticker + ml_trend_re_ticker + ml_trend_bond_ticker

        # Get stock prices in batches (or else it will hit ticker request rate limit ~ 250 request per 5 seconds)
        batch_size = 75
        symbol_price_tuples = await self._get_prices_in_batches(all_stocks, batch_size)

        # Create DataFrame
        all_price_data = pd.DataFrame(symbol_price_tuples, columns=['ticker', 'Close'])
        all_price_data['date'] = pd.to_datetime(self.current_date)

        # Separate all price data into respective ticker datasets
        permno_data = all_price_data[all_price_data['ticker'].isin(permno_ticker)]
        mrev_etf_hedge_data = all_price_data[all_price_data['ticker'].isin(mrev_etf_hedge_ticker)].set_index(['ticker', 'date']).sort_index(level=['ticker', 'date'])
        mrev_etf_hedge_data = mrev_etf_hedge_data[~mrev_etf_hedge_data.index.duplicated(keep='last')]
        mrev_mkt_hedge_data = all_price_data[all_price_data['ticker'].isin(mrev_mkt_hedge_ticker)].set_index(['ticker', 'date']).sort_index(level=['ticker', 'date'])
        mrev_mkt_hedge_data = mrev_mkt_hedge_data[~mrev_mkt_hedge_data.index.duplicated(keep='last')]
        trend_mls_com_data = all_price_data[all_price_data['ticker'].isin(trend_mls_com_ticker)].set_index(['ticker', 'date']).sort_index(level=['ticker', 'date'])
        trend_mls_com_data = trend_mls_com_data[~trend_mls_com_data.index.duplicated(keep='last')]
        trend_mls_bond_data = all_price_data[all_price_data['ticker'].isin(trend_mls_bond_ticker)].set_index(['ticker', 'date']).sort_index(level=['ticker', 'date'])
        trend_mls_bond_data = trend_mls_bond_data[~trend_mls_bond_data.index.duplicated(keep='last')]
        ml_trend_re_data = all_price_data[all_price_data['ticker'].isin(ml_trend_re_ticker)].set_index(['ticker', 'date']).sort_index(level=['ticker', 'date'])
        ml_trend_re_data = ml_trend_re_data[~ml_trend_re_data.index.duplicated(keep='last')]
        ml_trend_bond_data = all_price_data[all_price_data['ticker'].isin(ml_trend_bond_ticker)].set_index(['ticker', 'date']).sort_index(level=['ticker', 'date'])
        ml_trend_bond_data = ml_trend_bond_data[~ml_trend_bond_data.index.duplicated(keep='last')]

        # Add permnos to permno_data and keep 'ticker' column (this will be used for easy conversion when identifying stocks to long/short for different strategies)
        permno_to_ticker_dict = latest_data.reset_index(level='date')['ticker'].to_dict()
        ticker_to_permno_dict = {v: k for k, v in permno_to_ticker_dict.items()}
        permno_data['permno'] = permno_data['ticker'].map(ticker_to_permno_dict)
        permno_data = permno_data.set_index(['permno', 'date']).sort_index(level=['permno', 'date'])
        permno_data = permno_data[~permno_data.index.duplicated(keep='last')]

        # Adjust close price using previous day's dividend adjustment factor
        adj_factor_trade = pd.read_parquet(get_adj() / 'data_adj_permno_live.parquet.brotli')
        permno_data = permno_data.reset_index().set_index(['ticker', 'date'])
        permno_data = permno_data.merge(adj_factor_trade[['adj_factor']], left_index=True, right_index=True, how='left')
        permno_data['adj_factor'] = permno_data['adj_factor'].fillna(1.0)
        permno_data = permno_data.reset_index().set_index(['permno', 'date'])
        permno_data = permno_data.sort_values('ticker')
        permno_data['Close'] = permno_data['Close'] / permno_data['adj_factor']
        permno_data = permno_data.drop('adj_factor', axis=1)
        permno_data = permno_data.sort_index(level=['permno', 'date'])

        adj_factor_trade = pd.read_parquet(get_adj() / 'data_adj_mrev_etf_hedge_live.parquet.brotli')
        mrev_etf_hedge_data = mrev_etf_hedge_data.merge(adj_factor_trade[['adj_factor']], left_index=True, right_index=True, how='left')
        mrev_etf_hedge_data['adj_factor'] = mrev_etf_hedge_data['adj_factor'].fillna(1.0)
        mrev_etf_hedge_data['Close'] = mrev_etf_hedge_data['Close'] / mrev_etf_hedge_data['adj_factor']
        mrev_etf_hedge_data = mrev_etf_hedge_data.drop('adj_factor', axis=1)

        adj_factor_trade = pd.read_parquet(get_adj() / 'data_adj_mrev_mkt_hedge_live.parquet.brotli')
        mrev_mkt_hedge_data = mrev_mkt_hedge_data.merge(adj_factor_trade[['adj_factor']], left_index=True, right_index=True, how='left')
        mrev_mkt_hedge_data['adj_factor'] = mrev_mkt_hedge_data['adj_factor'].fillna(1.0)
        mrev_mkt_hedge_data['Close'] = mrev_mkt_hedge_data['Close'] / mrev_mkt_hedge_data['adj_factor']
        mrev_mkt_hedge_data = mrev_mkt_hedge_data.drop('adj_factor', axis=1)

        adj_factor_trade = pd.read_parquet(get_adj() / 'data_adj_trend_mls_com_live.parquet.brotli')
        trend_mls_com_data = trend_mls_com_data.merge(adj_factor_trade[['adj_factor']], left_index=True, right_index=True, how='left')
        trend_mls_com_data['adj_factor'] = trend_mls_com_data['adj_factor'].fillna(1.0)
        trend_mls_com_data['Close'] = trend_mls_com_data['Close'] / trend_mls_com_data['adj_factor']
        trend_mls_com_data = trend_mls_com_data.drop('adj_factor', axis=1)

        adj_factor_trade = pd.read_parquet(get_adj() / 'data_adj_trend_mls_bond_live.parquet.brotli')
        trend_mls_bond_data = trend_mls_bond_data.merge(adj_factor_trade[['adj_factor']], left_index=True, right_index=True, how='left')
        trend_mls_bond_data['adj_factor'] = trend_mls_bond_data['adj_factor'].fillna(1.0)
        trend_mls_bond_data['Close'] = trend_mls_bond_data['Close'] / trend_mls_bond_data['adj_factor']
        trend_mls_bond_data = trend_mls_bond_data.drop('adj_factor', axis=1)

        adj_factor_trade = pd.read_parquet(get_adj() / 'data_adj_ml_trend_re_live.parquet.brotli')
        ml_trend_re_data = ml_trend_re_data.merge(adj_factor_trade[['adj_factor']], left_index=True, right_index=True, how='left')
        ml_trend_re_data['adj_factor'] = ml_trend_re_data['adj_factor'].fillna(1.0)
        ml_trend_re_data['Close'] = ml_trend_re_data['Close'] / ml_trend_re_data['adj_factor']
        ml_trend_re_data = ml_trend_re_data.drop('adj_factor', axis=1)

        adj_factor_trade = pd.read_parquet(get_adj() / 'data_adj_ml_trend_bond_live.parquet.brotli')
        ml_trend_bond_data = ml_trend_bond_data.merge(adj_factor_trade[['adj_factor']], left_index=True, right_index=True, how='left')
        ml_trend_bond_data['adj_factor'] = ml_trend_bond_data['adj_factor'].fillna(1.0)
        ml_trend_bond_data['Close'] = ml_trend_bond_data['Close'] / ml_trend_bond_data['adj_factor']
        ml_trend_bond_data = ml_trend_bond_data.drop('adj_factor', axis=1)

        # Export Data
        permno_data.to_parquet(get_live_price() / 'data_permno_live.parquet.brotli', compression='brotli')
        mrev_etf_hedge_data.to_parquet(get_live_price() / 'data_mrev_etf_hedge_live.parquet.brotli', compression='brotli')
        mrev_mkt_hedge_data.to_parquet(get_live_price() / 'data_mrev_mkt_hedge_live.parquet.brotli', compression='brotli')
        trend_mls_com_data.to_parquet(get_live_price() / 'data_trend_mls_com_live.parquet.brotli', compression='brotli')
        trend_mls_bond_data.to_parquet(get_live_price() / 'data_trend_mls_bond_live.parquet.brotli', compression='brotli')
        ml_trend_re_data.to_parquet(get_live_price() / 'data_ml_trend_re_live.parquet.brotli', compression='brotli')
        ml_trend_bond_data.to_parquet(get_live_price() / 'data_ml_trend_bond_live.parquet.brotli', compression='brotli')

    # Store live price and live stock data in a recurring dataset
    def exec_live_store(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE LIVE STORE-------------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE LIVE STORE-------------------------------------------------------------------------------")
        def add_store(data, filename):
            # Check if file exists
            if os.path.exists(filename):
                existing_df = pd.read_parquet(filename)
                # Check if the current_date already exists in the existing_df
                if self.current_date in existing_df.index.get_level_values('date').values:
                    existing_df = existing_df[existing_df.index.get_level_values('date') != self.current_date]
                updated_df = pd.concat([existing_df, data], axis=0)
                updated_df.to_parquet(filename, compression='brotli')
            else:
                data.to_parquet(filename, compression='brotli')

        # Load Live Price
        permno_data = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        mrev_etf_hedge_data = pd.read_parquet(get_live_price() / 'data_mrev_etf_hedge_live.parquet.brotli')
        mrev_mkt_hedge_data = pd.read_parquet(get_live_price() / 'data_mrev_mkt_hedge_live.parquet.brotli')
        trend_mls_bond_data = pd.read_parquet(get_live_price() / 'data_trend_mls_com_live.parquet.brotli')
        trend_mls_com_data = pd.read_parquet(get_live_price() / 'data_trend_mls_bond_live.parquet.brotli')
        ml_trend_re_data = pd.read_parquet(get_live_price() / 'data_ml_trend_re_live.parquet.brotli')
        ml_trend_bond_data = pd.read_parquet(get_live_price() / 'data_ml_trend_bond_live.parquet.brotli')

        # Load Live Stock
        ml_ret = pd.read_parquet(get_live_stock() / 'trade_stock_ml_ret.parquet.brotli')
        ml_trend = pd.read_parquet(get_live_stock() / 'trade_stock_ml_trend.parquet.brotli')
        port_iv = pd.read_parquet(get_live_stock() / 'trade_stock_port_iv.parquet.brotli')
        port_id = pd.read_parquet(get_live_stock() / 'trade_stock_port_id.parquet.brotli')
        port_im = pd.read_parquet(get_live_stock() / 'trade_stock_port_im.parquet.brotli')
        trend_mls = pd.read_parquet(get_live_stock() / 'trade_stock_trend_mls.parquet.brotli')
        mrev_mkt = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_mkt.parquet.brotli')
        mrev_etf = pd.read_parquet(get_live_stock() / 'trade_stock_mrev_etf.parquet.brotli')

        # Store Live Price
        add_store(data=permno_data, filename=get_live() / 'data_permno_store.parquet.brotli')
        add_store(data=mrev_etf_hedge_data, filename=get_live() / 'data_mrev_etf_hedge_store.parquet.brotli')
        add_store(data=mrev_mkt_hedge_data, filename=get_live() / 'data_mrev_mkt_hedge_store.parquet.brotli')
        add_store(data=trend_mls_bond_data, filename=get_live() / 'data_trend_mls_bond_store.parquet.brotli')
        add_store(data=trend_mls_com_data, filename=get_live() / 'data_trend_mls_com_store.parquet.brotli')
        add_store(data=ml_trend_re_data, filename=get_live() / 'data_ml_trend_re_store.parquet.brotli')
        add_store(data=ml_trend_bond_data, filename=get_live() / 'data_ml_trend_bond_store.parquet.brotli')

        # Store Live Stock
        add_store(data=ml_ret, filename=get_live() / 'data_ml_ret_store.parquet.brotli')
        add_store(data=ml_trend, filename=get_live() / 'data_ml_trend_store.parquet.brotli')
        add_store(data=port_iv, filename=get_live() / 'data_port_iv_store.parquet.brotli')
        add_store(data=port_id, filename=get_live() / 'data_port_id_store.parquet.brotli')
        add_store(data=port_im, filename=get_live() / 'data_port_im_store.parquet.brotli')
        add_store(data=trend_mls, filename=get_live() / 'data_trend_mls_store.parquet.brotli')
        add_store(data=mrev_mkt, filename=get_live() / 'data_mrev_mkt_store.parquet.brotli')
        add_store(data=mrev_etf, filename=get_live() / 'data_mrev_etf_store.parquet.brotli')




