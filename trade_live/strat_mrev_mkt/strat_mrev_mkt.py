import quantstats as qs

from class_mrev.mrev_sd_epsil import MrevSDEpsil
from core.operation import *
from class_strat.strat import Strategy

class StratMrevMkt(Strategy):
    def __init__(self,
                 allocate=None,
                 current_date=None,
                 start_date=None,
                 threshold=None,
                 window_epsil=None,
                 sbo=None,
                 sso=None,
                 sbc=None,
                 ssc=None):

        '''
        allocate (float): Percentage of capital to allocate for this strategy
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for backtest period)
        start_date (str: YYYY-MM-DD): Start date for backtest period
        threshold (int): Market cap threshold to determine if a stock is buyable/shortable
        window_epsil (int): Rolling window size to calculate standardized s-score
        window_port (int): Rolling window size to calculate inverse volatility for main portfolio
        sbo (float): Threshold to determine buy signal
        sso (float): Threshold to determine sell signal
        sbc (float): Threshold to determine close buy signal
        ssc (float): Threshold to determine close sell signal
        '''


        super().__init__(allocate, current_date, threshold)
        self.allocate = allocate
        self.current_date = current_date
        self.start_date = start_date
        self.threshold = threshold
        self.window_epsil = window_epsil
        self.sbo = sbo
        self.sso = sso
        self.sbc = sbc
        self.ssc = ssc

    def exec_backtest(self):
        print("-----------------------------------------------------------------BACKTEST MREV MKT--------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        # Create MrevSDEpsil Class
        mrev_sd_epsil = MrevSDEpsil(name='mkt', threshold=self.threshold, sbo=self.sbo, sso=self.sso, sbc=self.sbc, ssc=self.ssc)

        # Params
        live = True
        hedge_ticker = ['SPY', 'MDY', 'VEA', 'EEM', 'VNQ', 'DBC']

        # Load in hedge dataset
        hedge_ret = get_data_fmp(ticker_list=hedge_ticker, start=self.start_date, current_date=self.current_date)
        hedge_ret = hedge_ret[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
        hedge_ret = hedge_ret.rename(columns={'Adj Close': 'Close'})
        hedge_ret = hedge_ret.loc[~hedge_ret.index.duplicated(keep='first')]

        # Export hedge dataset
        hedge_ret.to_parquet(get_strat_mrev_mkt() / 'data' / 'data_hedge.parquet.brotli', compression='brotli')

        # Create returns and unstack dataframe to only have 'date' index and 'ticker' columns
        hedge_ret = create_return(hedge_ret, [1])
        hedge_ret = hedge_ret.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        hedge_ret = hedge_ret.unstack('ticker').swaplevel(axis=1)
        hedge_ret.columns = ['_'.join(col).strip() for col in hedge_ret.columns.values]
        hedge_ret = hedge_ret.fillna(0)

        # Load in datasets
        stock = read_stock(get_large(live) / 'permno_live.csv')
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')

        # Create returns
        price = create_return(historical_price, [1])
        price = price.fillna(0)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------PERFORM ROLLING OLS-----------------------------------------------------------------------------------
        print("------------------------------------------------------------------PERFORM ROLLING OLS-----------------------------------------------------------------------------------")
        # Params
        ret = 'RET_01'
        factor_col = hedge_ret.columns.tolist()
        name = 'mkt_01'

        # Execute Rolling LR
        beta_data = rolling_ols_parallel(data=price, ret=ret, factor_data=hedge_ret, factor_cols=factor_col, window=self.window_epsil, name=name)

        # Retrieve required data
        beta_data = beta_data[beta_data.columns[1:len(factor_col) + 2]]
        beta_data = beta_data.fillna(0)

        # Calculate rolling mean, standard deviation, and z-score
        rolling_mean = beta_data.groupby('permno')[f'epsil_{name}_{self.window_epsil:02}'].rolling(window=self.window_epsil).mean().reset_index(level=0, drop=True)
        rolling_std = beta_data.groupby('permno')[f'epsil_{name}_{self.window_epsil:02}'].rolling(window=self.window_epsil).std().reset_index(level=0, drop=True)
        beta_data['s_score'] = (beta_data[f'epsil_{name}_{self.window_epsil:02}'] - rolling_mean) / rolling_std

        # Convert Hedge Dataframe to multi-index
        hedge_data = mrev_sd_epsil._create_multi_index(hedge_ret, stock)

        # Merge beta_data and hedge_multi
        comb_data = beta_data.merge(hedge_data, left_index=True, right_index=True, how='left')
        comb_data = comb_data.merge(price[['RET_01']], left_index=True, right_index=True, how='left')
        comb_data = comb_data.fillna(0)

        # Retrieve required data
        ret_columns = [col for col in comb_data.columns if "RET_01" in col]
        comb_data = comb_data[['s_score'] + ret_columns]

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------CREATE SIGNALS--------------------------------------------------------------------------------------
        print("--------------------------------------------------------------------CREATE SIGNALS--------------------------------------------------------------------------------------")
        # Add Market Cap data
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])
        comb_data = comb_data.merge(market, left_index=True, right_index=True, how='left')

        # Create Signals
        signal_data = mrev_sd_epsil._create_signal(comb_data)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------BACKTEST STRATEGY-----------------------------------------------------------------------------------
        print("--------------------------------------------------------------------BACKTEST STRATEGY-----------------------------------------------------------------------------------")
        # Shift returns for backtest results
        signal_data['RET_01'] = signal_data.groupby('permno')['RET_01'].shift(-1)
        hedge_ret = hedge_ret.shift(-1)

        # Calculate total returns and weights
        total_ret, beta_weight, stock_weight = mrev_sd_epsil.calc_total_ret(signal_data, hedge_ret)
        # Export backtest result
        filename = f"mrev_mkt_{date.today().strftime('%Y%m%d')}.html"
        dir_path = get_strat_mrev_mkt() / 'report' / filename
        qs.reports.html(total_ret, 'SPY', output=dir_path)

    def exec_live(self):
        print("-------------------------------------------------------------------EXEC MREV MKT----------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        # Create MrevSDEpsil Class
        mrev_sd_epsil = MrevSDEpsil(name='mkt', threshold=self.threshold, sbo=self.sbo, sso=self.sso, sbc=self.sbc, ssc=self.ssc)

        # Params
        live = True

        # Load in datasets
        stock = read_stock(get_large(live) / 'permno_live.csv')
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        historical_price = historical_price.loc[historical_price.index.get_level_values('date') != self.current_date]
        live_price = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        historical_hedge = pd.read_parquet(get_strat_mrev_mkt() / 'data' / 'data_hedge.parquet.brotli', columns=['Close'])
        historical_hedge = historical_hedge.loc[historical_hedge.index.get_level_values('date') != self.current_date]
        live_hedge = pd.read_parquet(get_live_price() / 'data_mkt_live.parquet.brotli')

        # Merge historical dataset and live dataset
        price = pd.concat([historical_price, live_price], axis=0)
        hedge = pd.concat([historical_hedge, live_hedge], axis=0)

        # Create returns
        price = create_return(price, [1])
        price = price.fillna(0)

        # Create returns
        hedge_ret = create_return(hedge, [1])
        hedge_ret = hedge_ret.drop(['Close'], axis=1)
        hedge_ret = hedge_ret.unstack('ticker').swaplevel(axis=1)
        hedge_ticker = [col[0] for col in hedge_ret.columns.values]
        hedge_ret.columns = ['_'.join(col).strip() for col in hedge_ret.columns.values]
        hedge_ret = hedge_ret.fillna(0)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------PERFORM ROLLING OLS-----------------------------------------------------------------------------------
        print("------------------------------------------------------------------PERFORM ROLLING OLS-----------------------------------------------------------------------------------")
        # Params
        ret = 'RET_01'
        factor_col = hedge_ret.columns.tolist()
        name = 'mkt_01'

        # Execute Rolling LR
        window_price = window_data(data=price, date=self.current_date, window=self.window_epsil*3)
        beta_data = rolling_ols_parallel(data=window_price, ret=ret, factor_data=hedge_ret, factor_cols=factor_col, window=self.window_epsil, name=name)

        # Retrieve required data
        beta_data = beta_data[beta_data.columns[1:len(factor_col) + 2]]
        beta_data = beta_data.fillna(0)

        # Calculate rolling mean, standard deviation, and z-score
        rolling_mean = beta_data.groupby('permno')[f'epsil_{name}_{self.window_epsil:02}'].rolling(window=self.window_epsil).mean().reset_index(level=0, drop=True)
        rolling_std = beta_data.groupby('permno')[f'epsil_{name}_{self.window_epsil:02}'].rolling(window=self.window_epsil).std().reset_index(level=0, drop=True)
        beta_data['s_score'] = (beta_data[f'epsil_{name}_{self.window_epsil:02}'] - rolling_mean) / rolling_std

        # Convert Hedge Dataframe to multi-index
        hedge_data = mrev_sd_epsil._create_multi_index(hedge_ret, stock)

        # Merge beta_data and hedge_multi
        comb_data = beta_data.merge(hedge_data, left_index=True, right_index=True, how='left')
        comb_data = comb_data.merge(window_price[['ticker', 'RET_01']], left_index=True, right_index=True, how='left')
        comb_data = comb_data.fillna(0)

        # Retrieve required data
        ret_columns = [col for col in comb_data.columns if "RET_01" in col]
        comb_data = comb_data[['ticker', 's_score'] + ret_columns]

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------CREATE SIGNALS--------------------------------------------------------------------------------------
        print("--------------------------------------------------------------------CREATE SIGNALS--------------------------------------------------------------------------------------")
        # Add Market Cap data
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])
        comb_data = comb_data.merge(market, left_index=True, right_index=True, how='left')
        comb_data['market_cap'] = comb_data.groupby('permno')['market_cap'].ffill()

        # Create Signals
        window_comb = window_data(data=comb_data, date=self.current_date, window=21*2)
        signal_data = mrev_sd_epsil._create_signal(window_comb)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------GET STOCKS----------------------------------------------------------------------------------------
        print("----------------------------------------------------------------------GET STOCKS----------------------------------------------------------------------------------------")
        # Calculate total returns and weights
        total_ret, beta_weight, stock_weight = mrev_sd_epsil.calc_total_ret(signal_data, hedge_ret)

        # Separate into long/short from current_date data
        latest_long_short = stock_weight.loc[stock_weight.index.get_level_values('date') == self.current_date]
        long = latest_long_short.loc[latest_long_short['normalized_weight'] > 0]
        short = latest_long_short.loc[latest_long_short['normalized_weight'] < 0]
        long_ticker = long['ticker'].tolist()
        short_ticker = short['ticker'].tolist()
        long_weight = (long['normalized_weight'] * self.allocate).tolist()
        short_weight = (short['normalized_weight'] * self.allocate * -1).tolist()
        hedge_ticker = hedge_ticker
        hedge_weight = (beta_weight.iloc[-1] * self.allocate).tolist()

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
        # Hedge Stock Dataframe
        hedge_df = pd.DataFrame({
            'date': [self.current_date] * len(hedge_ticker),
            'ticker': hedge_ticker,
            'weight': [abs(w) for w in hedge_weight],
            'type': ['short' if w < 0 else 'long' for w in hedge_weight]
        })

        # Combine long and short dataframes
        combined_df = pd.concat([long_df, short_df, hedge_df], axis=0)
        combined_df = combined_df.set_index(['date', 'ticker', 'type']).sort_index(level=['date', 'ticker', 'type'])
        filename = get_live_stock() / 'trade_stock_mrev_mkt.parquet.brotli'
        combined_df.to_parquet(filename, compression='brotli')


