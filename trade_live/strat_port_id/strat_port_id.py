from class_port.port_factor import PortFactor
from class_model.model_prep import ModelPrep
from class_strat.strat import Strategy
from core.operation import *

class StratPortID(Strategy):
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
        print("-----------------------------------------------------------------BACKTEST PORT ID---------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')

        # Params
        hedge_ticker = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']

        # Load in hedge dataset
        hedge_ret = get_data_fmp(ticker_list=hedge_ticker, start=self.start_date, current_date=self.current_date)
        hedge_ret = hedge_ret[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
        hedge_ret = hedge_ret.rename(columns={'Adj Close': 'Close'})
        hedge_ret = hedge_ret.loc[~hedge_ret.index.duplicated(keep='first')]

        # Export hedge dataset
        hedge_ret.to_parquet(get_strat_port_id() / 'data' / 'data_hedge.parquet.brotli', compression='brotli')

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Create returns and resample fund_q date index to daily
        ret_price = create_return(historical_price, [1])

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Defensive
        sb_sector = ModelPrep(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, sb_sector, left_index=True, right_index=True, how='left')
                       .merge(market, left_index=True, right_index=True, how='left'))

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            "XLB_RET_01_sector_01_126",
            "XLE_RET_01_sector_01_126",
            "XLF_RET_01_sector_01_126",
            "XLI_RET_01_sector_01_126",
            "XLK_RET_01_sector_01_126",
            "XLP_RET_01_sector_01_126",
            "XLU_RET_01_sector_01_126",
            "XLV_RET_01_sector_01_126",
            "XLY_RET_01_sector_01_126",
        ]

        filname = f"port_id_{date.today().strftime('%Y%m%d')}.html"
        dir_path = get_strat_port_id() / 'report' / filname

        long_short_stocks = PortFactor(data=factor_data, window=self.window_port, num_stocks=self.num_stocks, factors=factors,
                                       threshold=self.threshold, backtest=True, dir_path=dir_path).create_factor_port()

    def exec_live(self):
        print("-------------------------------------------------------------------EXEC PORT ID-----------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')
        window_date = (pd.to_datetime(self.current_date) - BDay(252)).strftime('%Y-%m-%d')

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        historical_price = historical_price.loc[historical_price.index.get_level_values('date') != self.current_date]
        live_price = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Concat historical price and live price datasets
        price = pd.concat([historical_price, live_price], axis=0)

        # Create returns crop into window data
        ret_price = create_return(price, [1])
        ret_price = window_data(data=ret_price, date=self.current_date, window=126 * 2)
        ret_price = ret_price.merge(market, left_index=True, right_index=True, how='left')
        ret_price['market_cap'] = ret_price.groupby('permno')['market_cap'].ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------CREATE FACTOR DATA-----------------------------------------------------------------------------------
        print("-------------------------------------------------------------------CREATE FACTOR DATA-----------------------------------------------------------------------------------")
        # Defensive
        factor_data = ret_price.copy(deep=True)

        # Smart Beta Sector
        def compute_sb_sector(data):
            # Load sector dataset
            historical_sector = pd.read_parquet(get_strat_port_id() / 'data' / 'data_hedge.parquet.brotli', columns=['Close'])
            historical_sector = historical_sector.loc[historical_sector.index.get_level_values('date') != self.current_date]
            live_sector = pd.read_parquet(get_live_price() / 'data_port_id_etf_live.parquet.brotli')
            # Merge historical dataset and live dataset
            sector = pd.concat([historical_sector, live_sector], axis=0)
            # Create returns
            sector_ret = create_return(sector, [1])
            sector_ret = sector_ret.drop(['Close'], axis=1)
            sector_ret = sector_ret.unstack('ticker').swaplevel(axis=1)
            sector_ret.columns = ['_'.join(col).strip() for col in sector_ret.columns.values]

            # Load risk-free rate
            risk_free = pd.read_parquet(get_parquet(True) / 'data_rf.parquet.brotli')

            # Create factor dataset
            sector_data = pd.concat([sector_ret, risk_free['RF']], axis=1)
            sector_data = sector_data.loc[data.index.get_level_values('date').min():data.index.get_level_values('date').max()]
            sector_data['RF'] = sector_data['RF'].ffill()
            sector_data = sector_data.fillna(0)
            factor_col = sector_data.columns[:-1]

            # Execute Rolling LR
            T = [1]
            for t in T:
                ret = f'RET_{t:02}'
                windows = [126]
                for window in windows:
                    betas = rolling_ols_parallel(data=data, ret=ret, factor_data=sector_data, factor_cols=factor_col.tolist(), window=window, name=f'sector_{t:02}')
                    data = data.join(betas)

            return data

        factor_data = compute_sb_sector(factor_data)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            "XLB_RET_01_sector_01_126",
            "XLE_RET_01_sector_01_126",
            "XLF_RET_01_sector_01_126",
            "XLI_RET_01_sector_01_126",
            "XLK_RET_01_sector_01_126",
            "XLP_RET_01_sector_01_126",
            "XLU_RET_01_sector_01_126",
            "XLV_RET_01_sector_01_126",
            "XLY_RET_01_sector_01_126",
        ]

        filename = f"port_id_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_port_id() / 'report' / filename

        latest_window_data = window_data(data=factor_data, date=self.current_date, window=self.window_port * 2)
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
        filename = get_live_stock() / 'trade_stock_port_id.parquet.brotli'
        combined_df.to_parquet(filename, compression='brotli')