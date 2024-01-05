import os

from class_port.port_factor import PortFactor
from class_model.prep_factor import PrepFactor

from core.operation import *

class StratPortIV:
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

    def backtest_port_iv(self):
        print("-----------------------------------------------------------------BACKTEST PORT IV---------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')

        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        fund_q = pd.read_parquet(get_parquet(live) / 'data_fund_raw_q.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Create returns and resample fund_q date index to daily
        ret_price = create_return(historical_price, [1])
        ret_price = ret_price.groupby('permno').shift(-1)
        date_index = historical_price.drop(historical_price.columns, axis=1)
        fund_q = fund_q.groupby('permno').shift(3)
        fund_q = date_index.merge(fund_q, left_index=True, right_index=True, how='left').groupby('permno').ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Fundamental
        accrual = PrepFactor(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        accrual = accrual.groupby('permno').shift(3)
        comp_debt = PrepFactor(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        comp_debt = comp_debt.groupby('permno').shift(3)
        inv_growth = PrepFactor(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        inv_growth = inv_growth.groupby('permno').shift(3)
        pcttoacc = PrepFactor(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        pcttoacc = pcttoacc.groupby('permno').shift(3)
        chtax = PrepFactor(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        chtax = chtax.groupby('permno').shift(3)
        net_debt_finance = PrepFactor(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        net_debt_finance = net_debt_finance.groupby('permno').shift(3)
        noa = PrepFactor(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        noa = noa.groupby('permno').shift(3)
        invest_ppe = PrepFactor(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        invest_ppe = invest_ppe.groupby('permno').shift(3)
        cheq = PrepFactor(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        cheq = cheq.groupby('permno').shift(3)
        xfin = PrepFactor(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        xfin = xfin.groupby('permno').shift(3)
        emmult = PrepFactor(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        emmult = emmult.groupby('permno').shift(3)
        grcapx = PrepFactor(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        grcapx = grcapx.groupby('permno').shift(3)
        fund_q['ev_to_ebitda'] = (fund_q['ltq'] + fund_q['ceqq'] + (fund_q['prccq'] * fund_q['cshoq'])) / (fund_q['niq'] + fund_q['dpq'] + fund_q['xintq'])
        fund_q = fund_q.replace([np.inf, -np.inf], np.nan)
        fund_factor = fund_q[['ev_to_ebitda']]

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, accrual, left_index=True, right_index=True, how='left')
                       .merge(comp_debt, left_index=True, right_index=True, how='left')
                       .merge(inv_growth, left_index=True, right_index=True, how='left')
                       .merge(pcttoacc, left_index=True, right_index=True, how='left')
                       .merge(chtax, left_index=True, right_index=True, how='left')
                       .merge(net_debt_finance, left_index=True, right_index=True, how='left')
                       .merge(noa, left_index=True, right_index=True, how='left')
                       .merge(invest_ppe, left_index=True, right_index=True, how='left')
                       .merge(cheq, left_index=True, right_index=True, how='left')
                       .merge(xfin, left_index=True, right_index=True, how='left')
                       .merge(emmult, left_index=True, right_index=True, how='left')
                       .merge(grcapx, left_index=True, right_index=True, how='left')
                       .merge(fund_factor, left_index=True, right_index=True, how='left')
                       .merge(market, left_index=True, right_index=True, how='left'))

        factor_data['accruals'] = factor_data.groupby('permno')['accruals'].ffill()
        factor_data['comp_debt_iss'] = factor_data.groupby('permno')['comp_debt_iss'].ffill()
        factor_data['inv_growth'] = factor_data.groupby('permno')['inv_growth'].ffill()
        factor_data['chtax'] = factor_data.groupby('permno')['chtax'].ffill()
        factor_data['net_debt_fin'] = factor_data.groupby('permno')['net_debt_fin'].ffill()
        factor_data['noa'] = factor_data.groupby('permno')['noa'].ffill()
        factor_data['invest_ppe_inv'] = factor_data.groupby('permno')['invest_ppe_inv'].ffill()
        factor_data['cheq'] = factor_data.groupby('permno')['cheq'].ffill()
        factor_data['pct_tot_acc'] = factor_data.groupby('permno')['pct_tot_acc'].ffill()
        factor_data['xfin'] = factor_data.groupby('permno')['xfin'].ffill()
        factor_data['emmult'] = factor_data.groupby('permno')['emmult'].ffill()
        factor_data['grcapx'] = factor_data.groupby('permno')['grcapx'].ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            "accruals",
            "inv_growth",
            "comp_debt_iss",
            "pct_tot_acc",
            'chtax',
            'net_debt_fin',
            'noa',
            'invest_ppe_inv',
            'cheq',
            'xfin',
            'emmult',
            'grcapx',
            'ev_to_ebitda'
        ]

        filname = f"port_iv_{date.today().strftime('%Y%m%d')}.html"
        dir_path = get_strat_port_iv() / 'report' / filname

        long_short_stocks = PortFactor(data=factor_data, window=self.window_port, num_stocks=self.num_stocks, factors=factors,
                                       threshold=self.threshold, backtest=True, dir_path=dir_path).create_factor_port()

    def exec_port_iv(self):
        print("-------------------------------------------------------------------EXEC PORT IV----------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large(live) / 'permno_live.csv')
        window_date = (pd.to_datetime(self.current_date) - BDay(252)).strftime('%Y-%m-%d')
        
        # Load in datasets
        historical_price = pd.read_parquet(get_parquet(live) / 'data_price.parquet.brotli')
        live_price = pd.read_parquet(get_live_price() / 'data_permno_live.parquet.brotli')
        fund_q = pd.read_parquet(get_parquet(live) / 'data_fund_raw_q.parquet.brotli')
        market = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Concat historical price and live price datasets
        price = pd.concat([historical_price, live_price], axis=0)

        # Create returns crop into window data
        ret_price = create_return(price, [1])
        ret_price = window_data(data=ret_price, date=self.current_date, window=self.window_port * 2)

        # Resample fund_q date index to daily and crop into window data
        date_index = price.drop(price.columns, axis=1)
        fund_q = date_index.merge(fund_q, left_index=True, right_index=True, how='left').groupby('permno').ffill()
        fund_q = window_data(data=fund_q, date=self.current_date, window=self.window_port * 2)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Fundamental
        accrual = PrepFactor(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        comp_debt = PrepFactor(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        inv_growth = PrepFactor(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        pcttoacc = PrepFactor(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        chtax = PrepFactor(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        net_debt_finance = PrepFactor(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        noa = PrepFactor(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        invest_ppe = PrepFactor(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        cheq = PrepFactor(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        xfin = PrepFactor(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        emmult = PrepFactor(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        grcapx = PrepFactor(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        fund_q['ev_to_ebitda'] = (fund_q['ltq'] + fund_q['ceqq'] + (fund_q['prccq'] * fund_q['cshoq'])) / (fund_q['niq'] + fund_q['dpq'] + fund_q['xintq'])
        fund_q = fund_q.replace([np.inf, -np.inf], np.nan)
        fund_factor = fund_q[['ev_to_ebitda']]

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, accrual, left_index=True, right_index=True, how='left')
                       .merge(comp_debt, left_index=True, right_index=True, how='left')
                       .merge(inv_growth, left_index=True, right_index=True, how='left')
                       .merge(pcttoacc, left_index=True, right_index=True, how='left')
                       .merge(chtax, left_index=True, right_index=True, how='left')
                       .merge(net_debt_finance, left_index=True, right_index=True, how='left')
                       .merge(noa, left_index=True, right_index=True, how='left')
                       .merge(invest_ppe, left_index=True, right_index=True, how='left')
                       .merge(cheq, left_index=True, right_index=True, how='left')
                       .merge(xfin, left_index=True, right_index=True, how='left')
                       .merge(emmult, left_index=True, right_index=True, how='left')
                       .merge(grcapx, left_index=True, right_index=True, how='left')
                       .merge(fund_factor, left_index=True, right_index=True, how='left')
                       .merge(market, left_index=True, right_index=True, how='left'))

        factor_data['accruals'] = factor_data.groupby('permno')['accruals'].ffill()
        factor_data['comp_debt_iss'] = factor_data.groupby('permno')['comp_debt_iss'].ffill()
        factor_data['inv_growth'] = factor_data.groupby('permno')['inv_growth'].ffill()
        factor_data['chtax'] = factor_data.groupby('permno')['chtax'].ffill()
        factor_data['net_debt_fin'] = factor_data.groupby('permno')['net_debt_fin'].ffill()
        factor_data['noa'] = factor_data.groupby('permno')['noa'].ffill()
        factor_data['invest_ppe_inv'] = factor_data.groupby('permno')['invest_ppe_inv'].ffill()
        factor_data['cheq'] = factor_data.groupby('permno')['cheq'].ffill()
        factor_data['pct_tot_acc'] = factor_data.groupby('permno')['pct_tot_acc'].ffill()
        factor_data['xfin'] = factor_data.groupby('permno')['xfin'].ffill()
        factor_data['emmult'] = factor_data.groupby('permno')['emmult'].ffill()
        factor_data['grcapx'] = factor_data.groupby('permno')['grcapx'].ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------
        print("-----------------------------------------------------------------------GET RANKINGS-------------------------------------------------------------------------------------")
        factors = [
            "accruals",
            "inv_growth",
            "comp_debt_iss",
            "pct_tot_acc",
            'chtax',
            'net_debt_fin',
            'noa',
            'invest_ppe_inv',
            'cheq',
            'xfin',
            'emmult',
            'grcapx',
            'ev_to_ebitda'
        ]

        filname = f"port_iv_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_port_iv() / 'report' / filname

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
        filename = get_live_stock() / 'trade_stock_port_iv.parquet.brotli'

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