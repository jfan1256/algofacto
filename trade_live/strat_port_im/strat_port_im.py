import os

from class_port.port_factor import PortFactor
from class_model.prep_factor import PrepFactor

from core.operation import *


class StratPortIM:
    def __init__(self,
                 allocate=None,
                 current_date=None,
                 start_date=None,
                 threshold=None,
                 num_stocks=None,
                 window_port=None):

        '''
        allocate (float): Percentage of capital to allocate for this strategy
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for backtest period)
        start_date (str: YYYY-MM-DD): Start date for backtest period
        num_stocks (int): Number of stocks to long/short
        threshold (int): Market cap threshold to determine if a stock is buyable/shortable
        window_port (int): Rolling window size to calculate inverse volatility
        '''

        self.allocate = allocate
        self.current_date = current_date
        self.start_date = start_date
        self.threshold = threshold
        self.num_stocks = num_stocks
        self.window_port = window_port

    def backtest_port_im(self):
        print("-----------------------------------------------------------------BACKTEST PORT IM---------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Create returns and resample fund_q date index to daily
        ret_price = create_return(historical_price, [1])
        ret_price = ret_price.groupby('permno').shift(-1)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Momentum
        mom_season = PrepFactor(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        mom_season6 = PrepFactor(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        load_ret = PrepFactor(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        mom_season_short = PrepFactor(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, mom_season, left_index=True, right_index=True, how='left')
                       .merge(mom_season6, left_index=True, right_index=True, how='left')
                       .merge(mom_season_short, left_index=True, right_index=True, how='left')
                       .merge(load_ret, left_index=True, right_index=True, how='left')
                       .merge(market, left_index=True, right_index=True, how='left'))

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            'load_ret_1',
            'load_ret_2',
            'load_ret_3',
            'load_ret_4',
            'load_ret_5',
            "mom_season",
            "mom_season_short",
            "mom_season_6"
        ]

        filname = f"port_im_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_port_im() / 'report' / filname

        long_short_stocks = PortFactor(data=factor_data, window=self.window_port, num_stocks=self.num_stocks, factors=factors,
                                       threshold=self.threshold, backtest=True, dir_path=dir_path).create_factor_port()

    def exec_port_im(self):
        print("-------------------------------------------------------------------EXEC PORT IM-----------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')
        window_date = (pd.to_datetime(self.current_date) - BDay(252)).strftime('%Y-%m-%d')

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        live_price = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Concat historical price and live price datasets
        price = pd.concat([historical_price, live_price], axis=0)

        # Create returns crop into window data
        ret_price = create_return(price, [1])
        ret_price = window_data(data=ret_price, date=self.current_date, window=self.window_port * 2)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Momentum
        mom_season = PrepFactor(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=window_date, end=self.current_date,
                                save=False).prep()
        mom_season6 = PrepFactor(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=window_date, end=self.current_date,
                                 save=False).prep()
        load_ret = PrepFactor(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=window_date, end=self.current_date,
                              save=False).prep()
        mom_season_short = PrepFactor(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=window_date, end=self.current_date,
                                      save=False).prep()

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, mom_season, left_index=True, right_index=True, how='left')
                       .merge(mom_season6, left_index=True, right_index=True, how='left')
                       .merge(mom_season_short, left_index=True, right_index=True, how='left')
                       .merge(load_ret, left_index=True, right_index=True, how='left')
                       .merge(market, left_index=True, right_index=True, how='left'))

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            'load_ret_1',
            'load_ret_2',
            'load_ret_3',
            'load_ret_4',
            'load_ret_5',
            "mom_season",
            "mom_season_short",
            "mom_season_6"
        ]

        filname = f"port_im_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_port_im() / 'report' / filname

        latest_window_data = window_data(data=factor_data, date=self.current_date, window=self.window_port)
        long_short_stocks = PortFactor(data=latest_window_data, window=self.window_port, num_stocks=self.num_stocks, factors=factors,
                                       threshold=self.threshold, backtest=False, dir_path=dir_path).create_factor_port()

        # Separate into long/short from current_date data
        latest_long_short = long_short_stocks.loc[long_short_stocks.index.get_level_values('date') == self.current_date]
        long = latest_long_short.loc[latest_long_short['final_weight'] >= 0]
        short = latest_long_short.loc[latest_long_short['final_weight'] < 0]
        long_ticker = long['ticker']
        short_ticker = short['ticker']
        long_weight = long['final_weight'] * self.allocate
        short_weight = short['final_weight'] * self.allocate * -1

        # Long Stock Dataframe
        long_df = pd.DataFrame({
            'date': [self.current_date] * len(long),
            'ticker': long_ticker,
            'weight': long_weight,
            'type': 'long'
        })
        # Short Stock Dataframe
        short_df = pd.DataFrame({
            'date': [self.current_date] * len(short),
            'ticker': short_ticker,
            'weight': short_weight,
            'type': 'short'
        })

        # Combine long and short dataframes
        combined_df = pd.concat([long_df, short_df], axis=0)
        combined_df = combined_df.set_index(['date', 'ticker', 'type']).sort_index(level=['date', 'ticker', 'type'])
        filename = get_live_stock() / 'trade_stock_port_im.parquet.brotli'

        # Check if file exists
        if os.path.exists(filename):
            existing_df = pd.read_parquet(filename)
            # Check if the current_date already exists in the existing_df
            if self.current_date in existing_df.index.get_level_values('date').values:
                existing_df = existing_df[existing_df.index.get_level_values('date') != self.current_date]
            updated_df = pd.concat([existing_df, combined_df], axis=0)
            updated_df.to_parquet(filename, compression='brotli')
        else:
            combined_df.to_parquet(filename, compression='brotli')