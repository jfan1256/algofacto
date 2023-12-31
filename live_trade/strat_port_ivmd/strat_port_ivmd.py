from port_class.port_factor import PortFactor
from model_class.prep_factor import PrepFactor

from functions.utils.func import *

class StratPortIVMD:
    def __init__(self,
                 allocate=None,
                 current_date=None,
                 start_date=None,
                 threshold=None,
                 num_stocks=None):

        '''
        allocate (float): Percentage of capital to allocate for this strategy
        num_stocks (int): Number of stocks to long/short
        threshold (int): Market cap threshold to determine if a stock is buyable/shortable
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for backtest period)
        start_date (str: YYYY-MM-DD): Start date for backtest period
        '''

        self.allocate = allocate
        self.threshold = threshold
        self.num_stocks = num_stocks
        self.current_date = current_date
        self.start_date = start_date


    def exec_port_ivmd(self):
        print("-------------------------------------------------------------------EXEC PORT IVMD--------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------DATA--------------------------------------------------------------------------------------------
        live = True
        stock = read_stock(get_large_dir(live) / 'permno_live.csv')
        
        # Load in datasets
        price = pd.read_parquet(get_parquet_dir(live) / 'data_price.parquet.brotli')
        fund_q = pd.read_parquet(get_parquet_dir(live) / 'data_fund_raw_q.parquet.brotli')
        market = pd.read_parquet(get_parquet_dir(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Create returns and resample fund_q date index to daily
        ret_price = create_return(price, [1])
        date_index = price.drop(price.columns, axis=1)
        fund_q = date_index.merge(fund_q, left_index=True, right_index=True, how='left').groupby('permno').ffill()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------
        print("-------------------------------------------------------------------LOAD FACTOR DATA-------------------------------------------------------------------------------------")
        # Fundamental
        accrual = PrepFactor(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        comp_debt = PrepFactor(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        inv_growth = PrepFactor(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        pcttoacc = PrepFactor(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        chtax = PrepFactor(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        net_debt_finance = PrepFactor(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        noa = PrepFactor(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        invest_ppe = PrepFactor(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        cheq = PrepFactor(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        xfin = PrepFactor(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        emmult = PrepFactor(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        grcapx = PrepFactor(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        fund_q['ev_to_ebitda'] = (fund_q['ltq'] + fund_q['ceqq'] + (fund_q['prccq'] * fund_q['cshoq'])) / (fund_q['niq'] + fund_q['dpq'] + fund_q['xintq'])
        fund_q = fund_q.replace([np.inf, -np.inf], np.nan)
        fund_factor = fund_q[['ev_to_ebitda']]

        # Momentum
        mom_season = PrepFactor(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        load_ret = PrepFactor(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        mom_season_short = PrepFactor(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()

        # Defensive
        sb_sector = PrepFactor(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()
        sb_pca = PrepFactor(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_date, end=self.current_date, save=False).prep()

        # Merge into one dataframe
        factor_data = (pd.merge(ret_price, sb_sector, left_index=True, right_index=True, how='left')
                          .merge(sb_pca, left_index=True, right_index=True, how='left')
                          .merge(accrual, left_index=True, right_index=True, how='left')
                          .merge(comp_debt, left_index=True, right_index=True, how='left')
                          .merge(inv_growth, left_index=True, right_index=True, how='left')
                          .merge(pcttoacc, left_index=True, right_index=True, how='left')
                          .merge(mom_season, left_index=True, right_index=True, how='left')
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
                          .mege(fund_factor, left_index=True, right_index=True, how='left')
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

        filname = f"port_ivmd_{date.today().strftime('%Y%m%d')}"
        dir_path = get_strat_port_ivmd() / 'report' / filname

        top_bottom_stocks = PortFactor(data=factor_data, num_stocks=self.num_stocks, factors=factors, threshold=self.threshold, dir_path=dir_path)