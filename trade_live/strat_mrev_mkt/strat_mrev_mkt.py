import os
import quantstats as qs

from core.operation import *

class StratMrevMkt:
    def __init__(self,
                 allocate=None,
                 current_date=None,
                 start_date=None,
                 threshold=None,
                 num_stocks=None,
                 window_epsil=None,
                 window_port=None,
                 sbo=None,
                 sso=None,
                 sbc=None,
                 ssc=None):

        '''
        allocate (float): Percentage of capital to allocate for this strategy
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for backtest period)
        start_date (str: YYYY-MM-DD): Start date for backtest period
        num_stocks (int): Number of stocks to long/short
        threshold (int): Market cap threshold to determine if a stock is buyable/shortable
        window_epsil (int): Rolling window size to calculate standardized s-score
        window_port (int): Rolling window size to calculate inverse volatility for main portfolio
        sbo (float): Threshold to determine buy signal
        sso (float): Threshold to determine sell signal
        sbc (float): Threshold to determine close buy signal
        ssc (float): Threshold to determine close sell signal
        '''

        self.allocate = allocate
        self.current_date = current_date
        self.start_date = start_date
        self.threshold = threshold
        self.num_stocks = num_stocks
        self.window_epsil = window_epsil
        self.window_port = window_port
        self.sbo = sbo
        self.sso = sso
        self.sbc = sbc
        self.ssc = ssc


    # Retrieves the top self.num_stocks stocks with the greatest inverse volatility weight
    def _top_inv_vol(self, df):
        long_df = df[df['position'] == 'long']
        short_df = df[df['position'] == 'short']
        top_long = long_df.nlargest(self.num_stocks, 'inv_vol')
        top_short = short_df.nlargest(self.num_stocks, 'inv_vol')
        comb = pd.concat([top_long, top_short], axis=0)
        return comb

    # Creates a multiindex of (permno, date) for a dataframe with only a date index
    @staticmethod
    def _create_multi_index(factor_data, stock):
        factor_values = pd.concat([factor_data] * len(stock), ignore_index=True).values
        multi_index = pd.MultiIndex.from_product([stock, factor_data.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=factor_data.columns, index=multi_index)
        multi_index_factor.index = multi_index_factor.index.set_names(['permno', 'date'])
        return multi_index_factor

    # Create signals
    def _create_signal(self, data):
        def apply_rules(group):
            # Initialize signals and positions
            signals = [None] * len(group)
            positions = [None] * len(group)
            # Create masks for conditions
            open_long_condition = (group['s_score'] < -self.sbo) & (group['market_cap'] > self.threshold)
            open_short_condition = (group['s_score'] > self.sso) & (group['market_cap'] > self.threshold)
            close_long_condition = group['s_score'] > -self.ssc
            close_short_condition = group['s_score'] < self.sbc
            # Flag to check if any position is open
            position_open = False
            current_position = None

            for i in range(len(group)):
                if position_open:
                    if positions[i - 1] == 'long' and close_long_condition.iloc[i]:
                        signals[i] = 'close long'
                        positions[i] = None
                        position_open = False
                        current_position = None
                    elif positions[i - 1] == 'short' and close_short_condition.iloc[i]:
                        signals[i] = 'close short'
                        positions[i] = None
                        position_open = False
                        current_position = None
                    else:
                        signals[i] = 'hold'
                        positions[i] = current_position
                else:
                    if open_long_condition.iloc[i]:
                        positions[i] = 'long'
                        signals[i] = 'buy to open'
                        current_position = 'long'
                        position_open = True
                    elif open_short_condition.iloc[i]:
                        positions[i] = 'short'
                        signals[i] = 'sell to open'
                        position_open = True
                        current_position = 'short'

            return pd.DataFrame({'signal': signals, 'position': positions}, index=group.index)

        # Sort data
        data = data.sort_index(level=['permno', 'date'])
        # Group by permno and apply the rules for each group
        results = data.groupby('permno').apply(apply_rules).reset_index(level=0, drop=True)
        # Flatten the results and assign back to the data
        data = data.join(results)
        return data

    # Calculate weights and total portfolio return
    @staticmethod
    def calc_total_ret(df, etf_returns):
        print("Get hedge weights...")
        mask_long = df['position'] == 'long'
        mask_short = df['position'] == 'short'
        df['hedge_weight'] = np.where(mask_long, -1, np.where(mask_short, 1, 0))

        # Get net hedge betas
        print("Get net hedge betas...")
        beta_columns = [col for col in df.columns if '_sector_' in col]
        weighted_betas = df[beta_columns].multiply(df['hedge_weight'], axis=0)
        net_hedge_betas = weighted_betas.groupby('date').sum()

        # Combine and normalize weights
        print("Normalize weights...")
        df['stock_weight'] = np.where(mask_long, 1, np.where(mask_short, -1, 0))

        # Normalize net hedge betas and stock weights combined
        df['abs_stock_weight'] = df['stock_weight'].abs()
        combined_weights = df.groupby('date')['abs_stock_weight'].sum() + net_hedge_betas.abs().sum(axis=1)
        df['normalized_weight'] = df['stock_weight'].div(combined_weights, axis=0)
        normalized_net_hedge_betas = net_hedge_betas.div(combined_weights, axis=0)

        # Get net hedge return
        print("Get net hedge returns...")
        net_hedge_returns = pd.DataFrame(index=normalized_net_hedge_betas.index)
        for beta in beta_columns:
            etf_return_column = beta.split('_sector_')[0]
            if etf_return_column in etf_returns.columns:
                net_hedge_returns[beta] = normalized_net_hedge_betas[beta] * etf_returns[etf_return_column]

        # Get total hedge return
        print("Get total hedge return...")
        net_hedge_return_total = net_hedge_returns.sum(axis=1)

        print("Get daily returns...")
        daily_returns = (df['RET_01'] * df['normalized_weight']).groupby('date').sum()

        print("Get total returns...")
        total_returns = daily_returns + net_hedge_return_total

        return total_returns, normalized_net_hedge_betas, df[['normalized_weight']]

    def backtest_mrev_mkt(self):
        print("-----------------------------------------------------------------BACKTEST MREV MKT--------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
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
        hedge_data = self._create_multi_index(hedge_ret, stock)

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
        signal_data = self._create_signal(comb_data)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------BACKTEST STRATEGY-----------------------------------------------------------------------------------
        print("--------------------------------------------------------------------BACKTEST STRATEGY-----------------------------------------------------------------------------------")
        # Calculate Inverse Volatility
        signal_data['vol'] = signal_data.groupby('ticker')['RET_01'].rolling(self.window_port).std().reset_index(level=0, drop=True)
        signal_data['inv_vol'] = 1 / signal_data['vol']

        # Retrieve top self.num_stocks based off inverse volatility
        signal_data = signal_data.groupby('date').apply(self._top_inv_vol).reset_index(level=0, drop=True)

        # Shift returns for backtest results
        signal_data['RET_01'] = signal_data.groupby('permno')['RET_01'].shift(-1)
        hedge_ret = hedge_ret.shift(-1)

        # Calculate total returns and weights
        total_ret, beta_weight, stock_weight = self.calc_total_ret(signal_data, hedge_ret)

        # Export backtest result
        filname = f"mrev_mkt_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_mrev_mkt() / 'report' / filname
        qs.reports.html(total_ret, 'SPY', output=dir_path)

    def exec_mrev_mkt(self):
        print("-------------------------------------------------------------------EXEC MREV MKT----------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        # Params
        live = True

        # Load in datasets
        stock = read_stock(get_large(live) / 'permno_live.csv')
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        live_price = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        historical_hedge = pd.read_parquet(get_strat_mrev_mkt() / 'data_hedge.parquet.brotli', columns=['Close'])
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
        hedge_ret.columns = [col[1] for col in hedge_ret.columns.values]
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
        hedge_data = self._create_multi_index(hedge_ret, stock)

        # Merge beta_data and hedge_multi
        comb_data = beta_data.merge(hedge_data, left_index=True, right_index=True, how='left')
        comb_data = comb_data.merge(window_price[['RET_01']], left_index=True, right_index=True, how='left')
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
        window_comb = window_data(data=comb_data, date=self.current_date, window=self.window_port*2)
        signal_data = self._create_signal(window_comb)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------GET STOCKS----------------------------------------------------------------------------------------
        print("----------------------------------------------------------------------GET STOCKS----------------------------------------------------------------------------------------")
        # Calculate Inverse Volatility
        signal_data['vol'] = signal_data.groupby('ticker')['RET_01'].rolling(self.window_port).std().reset_index(level=0, drop=True)
        signal_data['inv_vol'] = 1 / signal_data['vol']

        # Retrieve top self.num_stocks based off inverse volatility
        signal_data = signal_data.groupby('date').apply(self._top_inv_vol).reset_index(level=0, drop=True)

        # Calculate total returns and weights
        total_ret, beta_weight, stock_weight = self.calc_total_ret(signal_data, hedge_ret)

        # Separate into long/short from current_date data
        latest_long_short = stock_weight.loc[stock_weight.index.get_level_values('date') == self.current_date]
        long = latest_long_short.loc[latest_long_short['normalized_weight'] >= 0]
        short = latest_long_short.loc[latest_long_short['normalized_weight'] < 0]
        long_ticker = long['ticker']
        short_ticker = short['ticker']
        long_weight = long['normalized_weight'] * self.allocate
        short_weight = short['normalized_weight'] * self.allocate * -1
        hedge_ticker = hedge_ret.columns.tolist()
        hedge_weight = beta_weight[-1] * self.allocate

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
            'weight': hedge_weight,
            'type': 'long'
        })

        # Combine long and short dataframes
        combined_df = pd.concat([long_df, short_df, hedge_df], axis=0)
        combined_df = combined_df.set_index(['date', 'ticker', 'type']).sort_index(level=['date', 'ticker', 'type'])
        filename = get_live_stock() / 'trade_stock_mrev_mkt.parquet.brotli'

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


