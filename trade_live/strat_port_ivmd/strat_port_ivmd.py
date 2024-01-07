from class_port.port_factor import PortFactor
from class_model.model_prep import ModelPrep
from class_strat.strat import Strategy
from core.operation import *

class StratPortIVMD(Strategy):
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
        print("-----------------------------------------------------------------BACKTEST PORT IVMD-------------------------------------------------------------------------------------")
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
        ret_price = ret_price.groupby('permno').shift(-2)
        date_index = historical_price.drop(historical_price.columns, axis=1)
        fund_q = fund_q.groupby('permno').shift(3)
        fund_q = date_index.merge(fund_q, left_index=True, right_index=True, how='left').groupby('permno').ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Fundamental
        accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        accrual = accrual.groupby('permno').shift(3)
        comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        comp_debt = comp_debt.groupby('permno').shift(3)
        inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        inv_growth = inv_growth.groupby('permno').shift(3)
        pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        pcttoacc = pcttoacc.groupby('permno').shift(3)
        chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        chtax = chtax.groupby('permno').shift(3)
        net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        net_debt_finance = net_debt_finance.groupby('permno').shift(3)
        noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        noa = noa.groupby('permno').shift(3)
        invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        invest_ppe = invest_ppe.groupby('permno').shift(3)
        cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        cheq = cheq.groupby('permno').shift(3)
        xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        xfin = xfin.groupby('permno').shift(3)
        emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        emmult = emmult.groupby('permno').shift(3)
        grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        grcapx = grcapx.groupby('permno').shift(3)
        fund_q['ev_to_ebitda'] = (fund_q['ltq'] + fund_q['ceqq'] + (fund_q['prccq'] * fund_q['cshoq'])) / (fund_q['niq'] + fund_q['dpq'] + fund_q['xintq'])
        fund_q = fund_q.replace([np.inf, -np.inf], np.nan)
        fund_factor = fund_q[['ev_to_ebitda']]

        # Momentum
        mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        mom_season6 = ModelPrep(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        mom_season_short = ModelPrep(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()

        # Defensive
        sb_sector = ModelPrep(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        sb_pca = ModelPrep(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, sb_sector, left_index=True, right_index=True, how='left')
                       .merge(sb_pca, left_index=True, right_index=True, how='left')
                       .merge(accrual, left_index=True, right_index=True, how='left')
                       .merge(comp_debt, left_index=True, right_index=True, how='left')
                       .merge(inv_growth, left_index=True, right_index=True, how='left')
                       .merge(pcttoacc, left_index=True, right_index=True, how='left')
                       .merge(mom_season, left_index=True, right_index=True, how='left')
                       .merge(mom_season6, left_index=True, right_index=True, how='left')
                       .merge(chtax, left_index=True, right_index=True, how='left')
                       .merge(net_debt_finance, left_index=True, right_index=True, how='left')
                       .merge(noa, left_index=True, right_index=True, how='left')
                       .merge(invest_ppe, left_index=True, right_index=True, how='left')
                       .merge(cheq, left_index=True, right_index=True, how='left')
                       .merge(xfin, left_index=True, right_index=True, how='left')
                       .merge(emmult, left_index=True, right_index=True, how='left')
                       .merge(grcapx, left_index=True, right_index=True, how='left')
                       .merge(mom_season_short, left_index=True, right_index=True, how='left')
                       .merge(load_ret, left_index=True, right_index=True, how='left')
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
            "XLB_RET_01_sector_01_126",
            "XLE_RET_01_sector_01_126",
            "XLF_RET_01_sector_01_126",
            "XLI_RET_01_sector_01_126",
            "XLK_RET_01_sector_01_126",
            "XLP_RET_01_sector_01_126",
            "XLU_RET_01_sector_01_126",
            "XLV_RET_01_sector_01_126",
            "XLY_RET_01_sector_01_126",
            'PCA_Return_1_ret_pca_01_126',
            'PCA_Return_2_ret_pca_01_126',
            'PCA_Return_3_ret_pca_01_126',
            'PCA_Return_4_ret_pca_01_126',
            'PCA_Return_5_ret_pca_01_126',
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
            'ev_to_ebitda',
            'load_ret_1',
            'load_ret_2',
            'load_ret_3',
            'load_ret_4',
            'load_ret_5',
            "mom_season",
            "mom_season_short",
            "mom_season_6"
        ]

        filname = f"port_ivmd_{date.today().strftime('%Y%m%d')}.html"
        dir_path = get_strat_port_ivmd() / 'report' / filname

        long_short_stocks = PortFactor(data=factor_data, window=self.window_port, num_stocks=self.num_stocks, factors=factors,
                                       threshold=self.threshold, backtest=True, dir_path=dir_path).create_factor_port()

    def exec_live(self):
        print("-------------------------------------------------------------------EXEC PORT IVMD--------------------------------------------------------------------------------------")
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
        ret_price = window_data(data=ret_price, date=self.current_date, window=self.window_port * 2)

        # Resample fund_q date index to daily and crop into window data
        year_price = window_data(data=price, date=self.current_date, window=252 * 2)
        date_index = year_price.drop(year_price.columns, axis=1)
        fund_q = date_index.merge(fund_q, left_index=True, right_index=True, how='left').groupby('permno').ffill()
        fund_q = window_data(data=fund_q, date=self.current_date, window=self.window_port * 2)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Fundamental
        accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        fund_q['ev_to_ebitda'] = (fund_q['ltq'] + fund_q['ceqq'] + (fund_q['prccq'] * fund_q['cshoq'])) / (fund_q['niq'] + fund_q['dpq'] + fund_q['xintq'])
        fund_q = fund_q.replace([np.inf, -np.inf], np.nan)
        fund_factor = fund_q[['ev_to_ebitda']]

        # Momentum
        mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        mom_season6 = ModelPrep(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        mom_season_short = ModelPrep(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()

        # Defensive
        sb_sector = ModelPrep(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()
        sb_pca = ModelPrep(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=window_date, end=self.current_date, save=False).prep()

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, sb_sector, left_index=True, right_index=True, how='left')
                          .merge(sb_pca, left_index=True, right_index=True, how='left')
                          .merge(accrual, left_index=True, right_index=True, how='left')
                          .merge(comp_debt, left_index=True, right_index=True, how='left')
                          .merge(inv_growth, left_index=True, right_index=True, how='left')
                          .merge(pcttoacc, left_index=True, right_index=True, how='left')
                          .merge(mom_season, left_index=True, right_index=True, how='left')
                          .merge(mom_season6, left_index=True, right_index=True, how='left')
                          .merge(chtax, left_index=True, right_index=True, how='left')
                          .merge(net_debt_finance, left_index=True, right_index=True, how='left')
                          .merge(noa, left_index=True, right_index=True, how='left')
                          .merge(invest_ppe, left_index=True, right_index=True, how='left')
                          .merge(cheq, left_index=True, right_index=True, how='left')
                          .merge(xfin, left_index=True, right_index=True, how='left')
                          .merge(emmult, left_index=True, right_index=True, how='left')
                          .merge(grcapx, left_index=True, right_index=True, how='left')
                          .merge(mom_season_short, left_index=True, right_index=True, how='left')
                          .merge(load_ret, left_index=True, right_index=True, how='left')
                          .merge(fund_factor, left_index=True, right_index=True, how='left')
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
            'PCA_Return_1_ret_pca_01_126',
            'PCA_Return_2_ret_pca_01_126',
            'PCA_Return_3_ret_pca_01_126',
            'PCA_Return_4_ret_pca_01_126',
            'PCA_Return_5_ret_pca_01_126',
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
            'ev_to_ebitda',
            'load_ret_1',
            'load_ret_2',
            'load_ret_3',
            'load_ret_4',
            'load_ret_5',
            "mom_season",
            "mom_season_short",
            "mom_season_6"
        ]

        # Forward Fill Factors
        factor_data[factors] = factor_data.groupby('permno')[factors].ffill()

        filname = f"port_ivmd_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_port_ivmd() / 'report' / filname

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
        filename = get_live_stock() / 'trade_stock_port_ivmd.parquet.brotli'
        combined_df.to_parquet(filename, compression='brotli')