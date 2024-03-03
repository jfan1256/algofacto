from class_port.port_factor import PortFactor
from class_model.model_prep import ModelPrep
from class_strat.strat import Strategy
from core.operation import *

class StratPortIM(Strategy):
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

        super().__init__(allocate, current_date, threshold)
        self.allocate = allocate
        self.current_date = current_date
        self.start_date = start_date
        self.threshold = threshold
        self.num_stocks = num_stocks
        self.window_port = window_port

    def exec_backtest(self):
        print("-------------------------------------------------------------------BACKTEST PORT IM-------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        fund_q = pd.read_parquet(get_parquet(live) / 'data_fund_raw_q.parquet.brotli', columns=['ltq', 'ceqq', 'prccq', 'cshoq', 'niq', 'dpq', 'xintq'])
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Create returns and resample fund_q date index to daily
        ret_price = create_return(historical_price, [1])
        date_index = historical_price.drop(historical_price.columns, axis=1)
        fund_q = fund_q.groupby('permno').shift(3)
        fund_q = date_index.merge(fund_q, left_index=True, right_index=True, how='left').groupby('permno').ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Momentum
        mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        mom_season6 = ModelPrep(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        mom_season_short = ModelPrep(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, mom_season, left_index=True, right_index=True, how='left')
                       .merge(mom_season6, left_index=True, right_index=True, how='left')
                       .merge(mom_season_short, left_index=True, right_index=True, how='left')
                       .merge(market, left_index=True, right_index=True, how='left'))

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            "mom_season",
            "mom_season_short",
            "mom_season_6"
        ]

        filename = f"port_im_{date.today().strftime('%Y%m%d')}.html"
        dir_path = get_strat_port_im() / 'report' / filename

        long_short_stocks = PortFactor(data=factor_data, window=self.window_port, num_stocks=self.num_stocks, factors=factors,
                                       threshold=self.threshold, backtest=True, dir_path=dir_path).create_factor_port()

    def exec_live(self):
        print("---------------------------------------------------------------------EXEC PORT IM--------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')
        window_date = (pd.to_datetime(self.current_date) - BDay(252)).strftime('%Y-%m-%d')
        
        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        historical_price = historical_price.loc[historical_price.index.get_level_values('date') != self.current_date]
        live_price = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        fund_q = pd.read_parquet(get_parquet(live) / 'data_fund_raw_q.parquet.brotli', columns=['ltq', 'ceqq', 'prccq', 'cshoq', 'niq', 'dpq', 'xintq'])
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Concat historical price and live price datasets
        price = pd.concat([historical_price, live_price], axis=0)

        # Create returns crop into window data
        ret_price = create_return(price, [1])
        ret_price = window_data(data=ret_price, date=self.current_date, window=126 * 2)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------CREATE FACTOR DATA-----------------------------------------------------------------------------------
        print("-------------------------------------------------------------------CREATE FACTOR DATA-----------------------------------------------------------------------------------")
        # Create Factor Data
        factor_data = ret_price.copy(deep=True)

        # Momentum
        # Momentum Season
        def compute_mom(group):
            for n in range(23, 60, 12):
                group[f'temp{n}'] = group['RET_01'].shift(n)
            group['retTemp1'] = group[[col for col in group.columns if 'temp' in col]].sum(axis=1, skipna=True)
            group['retTemp2'] = group[[col for col in group.columns if 'temp' in col]].count(axis=1)
            group['mom_season'] = group['retTemp1'] / group['retTemp2']
            return group

        factor_data = factor_data.groupby('permno').apply(compute_mom).reset_index(level=0, drop=True)

        # Momentum Season 6
        def compute_mom_6(group):
            for n in range(71, 121, 12):
                group[f'temp{n}'] = group['RET_01'].shift(n)
            group['retTemp1'] = group[[col for col in group.columns if 'temp' in col]].sum(axis=1, skipna=True)
            group['retTemp2'] = group[[col for col in group.columns if 'temp' in col]].count(axis=1)
            group['mom_season_6'] = group['retTemp1'] / group['retTemp2']
            return group

        factor_data = factor_data.groupby('permno').apply(compute_mom_6).reset_index(level=0, drop=True)

        # Momentum Season Short
        def compute_mom_short(group):
            group['mom_season_short'] = group['RET_01'].shift(21)
            return group

        factor_data = factor_data.groupby('permno').apply(compute_mom_short).reset_index(level=0, drop=True)

        # Merge into one dataframe
        factor_data = (pd.merge(factor_data, market, left_index=True, right_index=True, how='left'))

        factor_data['market_cap'] = factor_data.groupby('permno')['market_cap'].ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            "mom_season",
            "mom_season_short",
            "mom_season_6"
        ]

        # Forward Fill Factors
        factor_data[factors] = factor_data.groupby('permno')[factors].ffill()

        filename = f"port_im_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_port_im() / 'report' / filename

        latest_window_data = window_data(data=factor_data, date=self.current_date, window=self.window_port*2)
        long_short_stocks = PortFactor(data=latest_window_data, window=self.window_port, num_stocks=self.num_stocks, factors=factors,
                                       threshold=self.threshold, backtest=False, dir_path=dir_path).create_factor_port()

        # Separate into long/short from current_date data
        latest_long_short = long_short_stocks.loc[long_short_stocks.index.get_level_values('date') == self.current_date]
        long = latest_long_short.loc[latest_long_short['final_weight'] > 0]
        short = latest_long_short.loc[latest_long_short['final_weight'] < 0]
        long_ticker = long['ticker'].tolist()
        short_ticker = short['ticker'].tolist()
        long_weight = (long['final_weight'] * self.allocate).tolist()
        short_weight = (short['final_weight'] * self.allocate * -1).tolist()

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
        combined_df.to_parquet(filename, compression='brotli')