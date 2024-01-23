import talib
import quantstats as qs

from fredapi import Fred
from core.operation import *
from class_trend.trend_helper import TrendHelper
from class_strat.strat import Strategy

class StratTrendMLS(Strategy):
    def __init__(self,
                 allocate=None,
                 current_date=None,
                 start_date=None,
                 threshold=None,
                 num_stocks=None,
                 window_hedge=None,
                 window_port=None):

        '''
        allocate (float): Percentage of capital to allocate for this strategy
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for backtest period)
        start_date (str: YYYY-MM-DD): Start date for backtest period
        num_stocks (int): Number of stocks to long
        threshold (int): Market cap threshold to determine if a stock is buyable/shortable
        window_hedge (int): Rolling window size to calculate inverse volatility for hedge portfolio
        window_port (int): Rolling window size to calculate inverse volatility for trend portfolio
        '''

        super().__init__(allocate, current_date, threshold)
        self.allocate = allocate
        self.current_date = current_date
        self.start_date = start_date
        self.threshold = threshold
        self.num_stocks = num_stocks
        self.window_hedge = window_hedge
        self.window_port = window_port

        with open(get_config() / 'api_key.json') as f:
            config = json.load(f)
            fred_key = config['fred_key']

        self.fred_key = fred_key

    def exec_backtest(self):
        print("-----------------------------------------------------------------BACKTEST TREND MLS-------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        # Create Trend Helper Class
        trend_helper = TrendHelper(current_date=self.current_date, start_date=self.start_date, num_stocks=self.num_stocks)

        # Params
        live = True

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Create returns and resample fund_q date index to daily
        price = create_return(historical_price, [1])
        price['RET_01'] = price.groupby('permno')['RET_01'].shift(-1)
        price = price.merge(market, left_index=True, right_index=True, how='left')

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------CREATE BOND+COMMODITY PORT-------------------------------------------------------------------------------
        print("---------------------------------------------------------------CREATE BOND+COMMODITY PORT-------------------------------------------------------------------------------")
        # Commodities
        com_ticker = ['GLD', 'SLV', 'PDBC', 'USO', 'AMLP', 'XOP']
        com = trend_helper._get_ret(com_ticker)
        com.to_parquet(get_strat_trend_mls() / 'data' / 'data_com.parquet.brotli', compression='brotli')

        # Bonds
        bond_ticker = ['BND', 'AGG', 'BNDX', 'VCIT', 'MUB', 'VCSH', 'BSV', 'VTEB', 'IEF', 'MBB', 'GOVT', 'VGSH', 'IUSB', 'TIP']
        bond = trend_helper._get_ret(bond_ticker)
        bond.to_parquet(get_strat_trend_mls() / 'data' / 'data_bond.parquet.brotli', compression='brotli')

        # Create portfolio
        bond_com_port = pd.concat([bond, com], axis=0)
        bond_com_port['vol'] = bond_com_port.groupby('ticker')['RET_01'].transform(lambda x: x.rolling(self.window_hedge).std().shift(1))
        bond_com_port['inv_vol'] = 1 / bond_com_port['vol']
        bond_com_port['norm_inv_vol'] = bond_com_port.groupby('date')['inv_vol'].apply(lambda x: x / x.sum()).reset_index(level=0, drop=True)
        bond_com_port['RET_01'] = bond_com_port['RET_01'].groupby('ticker').shift(-1)
        bond_com_port['weighted_ret'] = bond_com_port['RET_01'] * bond_com_port['norm_inv_vol']
        bond_com_port = bond_com_port.groupby('date')['weighted_ret'].sum()
        bond_com_port = bond_com_port.to_frame()
        bond_com_port.columns = ['bond_comm_ret']

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------GET MACRO DATA-------------------------------------------------------------------------------------
        print("---------------------------------------------------------------------GET MACRO DATA-------------------------------------------------------------------------------------")
        # Date Index
        date_index = bond_com_port.index
        date_index = date_index.to_frame().drop('date', axis=1).reset_index()

        # 5-Year Inflation Rate
        fred = Fred(api_key=self.fred_key)
        inflation = fred.get_series("T5YIE").to_frame()
        inflation.columns = ['5YIF']
        inflation = inflation.shift(1)
        inflation = inflation.reset_index()
        inflation = pd.merge_asof(date_index, inflation, left_on='date', right_on='index', direction='backward')
        inflation = inflation.set_index('date').drop('index', axis=1)
        inflation = inflation.ffill()

        # Unemployment Rate
        fred = Fred(api_key=self.fred_key)
        unemploy = fred.get_series("UNRATE").to_frame()
        unemploy.columns = ['UR']
        unemploy = unemploy.shift(1)
        unemploy = unemploy.reset_index()
        unemploy = pd.merge_asof(date_index, unemploy, left_on='date', right_on='index', direction='backward')
        unemploy = unemploy.set_index('date').drop('index', axis=1)
        unemploy = unemploy.ffill()

        # 10-year vs. 2-year Yield Curve
        fred = Fred(api_key=self.fred_key)
        yield_curve = fred.get_series("T10Y2Y").to_frame()
        yield_curve.columns = ['YIELD']
        yield_curve = yield_curve.shift(1)
        yield_curve = yield_curve.reset_index()
        yield_curve = pd.merge_asof(date_index, yield_curve, left_on='date', right_on='index', direction='backward')
        yield_curve = yield_curve.set_index('date').drop('index', axis=1)
        yield_curve = yield_curve.ffill()

        # Export Macro Data
        inflation.to_parquet(get_strat_trend_mls() / 'data'/ 'data_if.parquet.brotli', compression='brotli')
        unemploy.to_parquet(get_strat_trend_mls() / 'data'/ 'data_ur.parquet.brotli', compression='brotli')
        yield_curve.to_parquet(get_strat_trend_mls() / 'data'/ 'data_yield.parquet.brotli', compression='brotli')

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------CREATE SIGNALS-------------------------------------------------------------------------------------
        print("---------------------------------------------------------------------CREATE SIGNALS-------------------------------------------------------------------------------------")
        # Exponential Moving Averages
        for t in [60, 252]:
            price[f'ema_{t}'] = (price.groupby('permno', group_keys=False).apply(lambda x: talib.EMA(x.Close, timeperiod=t)))

        ema_buy = (price['ema_60'] > price['ema_252'])

        # Relative Strength Index
        for t in [5]:
            price[f'rsi_{t}'] = (price.groupby('permno', group_keys=False).apply(lambda x: talib.RSI(x.Close, timeperiod=t)))

        rsi_buy = (price['rsi_5'] < 30)

        # Macro Trend
        macro = pd.concat([inflation, unemploy, yield_curve], axis=1)
        macro['5YIF_z'] = (macro['5YIF'] - macro['5YIF'].mean()) / macro['5YIF'].std()
        macro['UR_z'] = (macro['UR'] - macro['UR'].mean()) / macro['UR'].std()
        macro['YIELD_z'] = (macro['YIELD'] - macro['YIELD'].mean()) / macro['YIELD'].std()
        macro['mt'] = macro[['5YIF_z', 'UR_z', 'YIELD_z']].mean(axis=1)

        for t in [21, 60]:
            macro[f'mt_{t}'] = macro['mt'].rolling(t).mean()

        macro_buy = (macro['mt_21'] > macro['mt_60'])
        macro_buy_df = macro_buy.to_frame()
        macro_buy_df.columns = ['macro_buy']

        # Market Threshold
        market_buy = (price['market_cap'] > self.threshold)

        # Volume Trend
        volume_buy = (price['Volume'] > price['Volume'].rolling(window=60).mean())

        # Create signal column
        price['signal'] = 0
        price.loc[ema_buy & volume_buy & rsi_buy & market_buy, 'signal'] = 1
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------BACKTEST----------------------------------------------------------------------------------------
        print("------------------------------------------------------------------------BACKTEST----------------------------------------------------------------------------------------")
        # Create trend portfolio
        price['current_ret'] = price.groupby('permno')['RET_01'].shift(1)
        price['vol'] = price.groupby('permno')['current_ret'].transform(lambda x: x.rolling(self.window_port).std().shift(1))
        price['inv_vol'] = 1 / price['vol']
        trend_port = price.groupby('date').apply(trend_helper._top_inv_vol).reset_index(level=0, drop=True)
        trend_port['norm_inv_vol'] = trend_port.groupby('date')['inv_vol'].apply(lambda x: x / x.sum()).reset_index(level=0, drop=True)
        trend_port['weighted_ret'] = trend_port['RET_01'] * trend_port['norm_inv_vol'] * trend_port['signal']
        trend_ret = trend_port.groupby('date')['weighted_ret'].sum()
        trend_ret = trend_ret.to_frame()
        trend_ret.columns = ['inv_vol_ret']

        # Combine trend portfolio + hedge portfolio
        total_ret = pd.merge(trend_ret, bond_com_port, left_index=True, right_index=True, how='left')
        total_ret = total_ret.merge(macro_buy_df, left_index=True, right_index=True, how='left')
        col1, col2 = total_ret.columns[0], total_ret.columns[1]
        total_ret['total_ret'] = total_ret.apply(trend_helper._calc_total_port, args=(col1, col2), axis=1)
        total_daily_ret = total_ret['total_ret']

        # Export backtest result
        filename = f"trend_mls_{date.today().strftime('%Y%m%d')}.html"
        dir_path = get_strat_trend_mls() / 'report' / filename
        qs.reports.html(total_daily_ret, 'SPY', output=dir_path)

    def exec_live(self):
        print("-------------------------------------------------------------------EXEC TREND MLS---------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        # Create Trend Helper Class
        trend_helper = TrendHelper(current_date=self.current_date, start_date=self.start_date, num_stocks=self.num_stocks)

        # Params
        live = True

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        historical_price = historical_price.loc[historical_price.index.get_level_values('date') != self.current_date]
        live_price = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        historical_com = pd.read_parquet(get_strat_trend_mls() / 'data' / 'data_com.parquet.brotli', columns=['Close'])
        historical_com = historical_com.loc[historical_com.index.get_level_values('date') != self.current_date]
        live_com = pd.read_parquet(get_live_price() / 'data_com_live.parquet.brotli')
        historical_bond = pd.read_parquet(get_strat_trend_mls() / 'data' / 'data_bond.parquet.brotli', columns=['Close'])
        historical_bond = historical_bond.loc[historical_bond.index.get_level_values('date') != self.current_date]
        live_bond = pd.read_parquet(get_live_price() / 'data_bond_live.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Merge historical dataset and live dataset
        price = pd.concat([historical_price, live_price], axis=0)
        com = pd.concat([historical_com, live_com], axis=0)
        bond = pd.concat([historical_bond, live_bond], axis=0)

        # Create returns
        price = create_return(price, [1])
        price = price.merge(market, left_index=True, right_index=True, how='left')
        price['market_cap'] = price.groupby('permno')['market_cap'].ffill()
        com = create_return(com, [1])
        bond = create_return(bond, [1])

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------CREATE BOND+COMMODITY PORT-------------------------------------------------------------------------------
        print("---------------------------------------------------------------CREATE BOND+COMMODITY PORT-------------------------------------------------------------------------------")
        # Create portfolio
        bond_com_port = pd.concat([bond, com], axis=0)
        window_bond_com_port = window_data(data=bond_com_port, date=self.current_date, window=self.window_hedge*2)
        window_bond_com_port['vol'] = window_bond_com_port.groupby('ticker')['RET_01'].transform(lambda x: x.rolling(self.window_hedge).std().shift(1))
        window_bond_com_port['inv_vol'] = 1 / window_bond_com_port['vol']
        window_bond_com_port['weight'] = window_bond_com_port.groupby('date')['inv_vol'].apply(lambda x: x / x.sum()).reset_index(level=0, drop=True)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------GET MACRO DATA-------------------------------------------------------------------------------------
        print("---------------------------------------------------------------------GET MACRO DATA-------------------------------------------------------------------------------------")
        # Date Index
        date_index = window_bond_com_port.drop(window_bond_com_port.columns, axis=1)

        # 5-Year Inflation Rate
        inflation = pd.read_parquet(get_strat_trend_mls() / 'data' / 'data_if.parquet.brotli')
        inflation = pd.merge(date_index, inflation, left_index=True, right_index=True, how='left')
        inflation = inflation.ffill()

        # Unemployment Rate
        unemploy = pd.read_parquet(get_strat_trend_mls() / 'data' / 'data_ur.parquet.brotli')
        unemploy = pd.merge(date_index, unemploy, left_index=True, right_index=True, how='left')
        unemploy = unemploy.ffill()

        # 10-year vs. 2-year Yield Curve
        yield_curve = pd.read_parquet(get_strat_trend_mls() / 'data' / 'data_yield.parquet.brotli')
        yield_curve = pd.merge(date_index, yield_curve, left_index=True, right_index=True, how='left')
        yield_curve = yield_curve.ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------CREATE SIGNALS-------------------------------------------------------------------------------------
        print("---------------------------------------------------------------------CREATE SIGNALS-------------------------------------------------------------------------------------")
        # Window Data
        window_price = window_data(data=price, date=self.current_date, window=252*2)

        # Exponential Moving Averages
        for t in [60, 252]:
            window_price[f'ema_{t}'] = (window_price.groupby('permno', group_keys=False).apply(lambda x: talib.EMA(x.Close, timeperiod=t)))

        ema_buy = (window_price['ema_60'] > window_price['ema_252'])

        # Relative Strength Index
        for t in [5]:
            window_price[f'rsi_{t}'] = (window_price.groupby('permno', group_keys=False).apply(lambda x: talib.RSI(x.Close, timeperiod=t)))

        rsi_buy = (window_price['rsi_5'] < 30)

        # Macro Trend
        macro = pd.concat([inflation, unemploy, yield_curve], axis=1)
        window_macro = macro.tail(60*2)
        window_macro['5YIF_z'] = (window_macro['5YIF'] - window_macro['5YIF'].mean()) / window_macro['5YIF'].std()
        window_macro['UR_z'] = (window_macro['UR'] - window_macro['UR'].mean()) / window_macro['UR'].std()
        window_macro['YIELD_z'] = (window_macro['YIELD'] - window_macro['YIELD'].mean()) / window_macro['YIELD'].std()
        window_macro['mt'] = window_macro[['5YIF_z', 'UR_z', 'YIELD_z']].mean(axis=1)

        for t in [21, 60]:
            window_macro[f'mt_{t}'] = window_macro['mt'].rolling(t).mean()

        macro_buy = (window_macro['mt_21'] > window_macro['mt_60'])
        macro_buy_df = macro_buy.to_frame()
        macro_buy_df.columns = ['macro_buy']

        # Market Threshold
        market_buy = (window_price['market_cap'] > self.threshold)

        # Volume Trend
        volume_buy = (window_price['Volume'] > window_price['Volume'].rolling(window=60).mean())

        # Create signal column
        window_price['signal'] = 0
        window_price.loc[ema_buy & volume_buy & rsi_buy & market_buy, 'signal'] = 1
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------GET STOCKS--------------------------------------------------------------------------------------
        print("------------------------------------------------------------------------GET STOCKS--------------------------------------------------------------------------------------")
        # Create trend portfolio
        window_price['vol'] = window_price.groupby('permno')['RET_01'].transform(lambda x: x.rolling(self.window_port).std().shift(1))
        window_price['inv_vol'] = 1 / window_price['vol']
        trend_port = window_price.groupby('date').apply(trend_helper._top_inv_vol).reset_index(level=0, drop=True)
        trend_port['weight'] = trend_port.groupby('date')['inv_vol'].apply(lambda x: x / x.sum()).reset_index(level=0, drop=True)

        # Total portfolio allocation weights
        macro_buy_df = macro_buy_df.loc[macro_buy_df.index.get_level_values('date') == self.current_date]
        if macro_buy_df.values[0]:
            trend_factor = 0.5
            hedge_factor = 0.5
        else:
            trend_factor = 0.25
            hedge_factor = 0.75

        # Get long weights and tickers from trend portfolio and hedge portfolio
        latest_trend_port = trend_port.loc[trend_port.index.get_level_values('date') == self.current_date]
        latest_bond_com_port = window_bond_com_port.loc[window_bond_com_port.index.get_level_values('date') == self.current_date]
        trend_ticker = latest_trend_port['ticker'].tolist()
        trend_weight = (latest_trend_port['weight'] * trend_factor * self.allocate).tolist()
        hedge_ticker = latest_bond_com_port.index.get_level_values('ticker').unique().tolist()
        hedge_weight = (latest_bond_com_port['weight'] * hedge_factor * self.allocate).tolist()
        long_ticker = trend_ticker + hedge_ticker
        long_weight = trend_weight + hedge_weight

        # Long Stock Dataframe
        long_df = pd.DataFrame({
            'date': [self.current_date] * len(long_ticker),
            'ticker': long_ticker,
            'weight': long_weight,
            'type': 'long'
        })

        # Combine long and short dataframes
        long_df = long_df.set_index(['date', 'ticker', 'type']).sort_index(level=['date', 'ticker', 'type'])
        filename = get_live_stock() / 'trade_stock_trend_mls.parquet.brotli'
        long_df.to_parquet(filename, compression='brotli')