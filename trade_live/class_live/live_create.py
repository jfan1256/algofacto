from core.operation import *

from class_factor.factor_ind import FactorInd
from class_factor.factor_ind_mom import FactorIndMom
from class_factor.factor_ret import FactorRet
from class_factor.factor_volatility import FactorVolatility
from class_factor.factor_volume import FactorVolume
from class_factor.factor_time import FactorTime
from class_factor.factor_talib import FactorTalib
from class_factor.factor_load_ret import FactorLoadRet
from class_factor.factor_clust_ret import FactorClustRet
from class_factor.factor_clust_load_ret import FactorClustLoadRet
from class_factor.factor_clust_ind_mom import FactorClustIndMom
from class_factor.factor_load_volume import FactorLoadVolume
from class_factor.factor_sb_sector import FactorSBSector
from class_factor.factor_sb_pca import FactorSBPCA
from class_factor.factor_ind_fama import FactorIndFama
from class_factor.factor_ind_mom_fama import FactorIndMomFama
from class_factor.factor_clust_ind_mom_fama import FactorClustIndMomFama
from class_factor.factor_age_mom import FactorAgeMom
from class_factor.factor_net_debt_finance import FactorNetDebtFinance
from class_factor.factor_chtax import FactorCHTax
from class_factor.factor_asset_growth import FactorAssetGrowth
from class_factor.factor_mom_season_short import FactorMomSeasonShort
from class_factor.factor_mom_season import FactorMomSeason
from class_factor.factor_noa import FactorNOA
from class_factor.factor_invest_ppe_inv import FactorInvestPPEInv
from class_factor.factor_inv_growth import FactorInvGrowth
from class_factor.factor_trend_factor import FactorTrendFactor
from class_factor.factor_ind_sub import FactorIndSub
from class_factor.factor_ind_mom_sub import FactorIndMomSub
from class_factor.factor_clust_ind_mom_sub import FactorClustIndMomSub
from class_factor.factor_comp_debt import FactorCompDebt
from class_factor.factor_mom_vol import FactorMomVol
from class_factor.factor_mom_season6 import FactorMomSeason6
from class_factor.factor_mom_season11 import FactorMomSeason11
from class_factor.factor_int_mom import FactorIntMom
from class_factor.factor_cheq import FactorChEQ
from class_factor.factor_xfin import FactorXFIN
from class_factor.factor_emmult import FactorEmmult
from class_factor.factor_accrual import FactorAccrual
from class_factor.factor_frontier import FactorFrontier
from class_factor.factor_ret_comp import FactorRetComp
from class_factor.factor_mom_off_season import FactorMomOffSeason
from class_factor.factor_mom_season16 import FactorMomSeason16
from class_factor.factor_mom_season9 import FactorMomSeason9
from class_factor.factor_accrual_bm import FactorAccrualBM
from class_factor.factor_pcttotacc import FactorPctTotAcc
from class_factor.factor_rds import FactorRDS
from class_factor.factor_hire import FactorHire
from class_factor.factor_mom_rev import FactorMomRev
from class_factor.factor_grcapx import FactorGrcapx
from class_factor.factor_earning_streak import FactorEarningStreak
from class_factor.factor_sign_ret import FactorSignRet
from class_factor.factor_cond_ret import FactorCondRet
from class_factor.factor_dividend import FactorDividend
from class_factor.factor_vol_comp import FactorVolComp
from class_factor.factor_sign_volume import FactorSignVolume
from class_factor.factor_sign_volatility import FactorSignVolatility
from class_factor.factor_load_volatility import FactorLoadVolatility
from class_factor.factor_ind_vwr import FactorIndVWR
from class_factor.factor_ind_vwr_fama import FactorIndVWRFama
from class_factor.factor_ind_vwr_sub import FactorIndVWRSub
from class_factor.factor_ms import FactorMS
from class_factor.factor_gradexp import FactorGradexp
from class_factor.factor_size import FactorSize
from class_factor.factor_ret_max import FactorRetMax
from class_factor.factor_ret_skew import FactorRetSkew
from class_factor.factor_mom_off_season6 import FactorMomOffSeason6
from class_factor.factor_mom_off_season11 import FactorMomOffSeason11
from class_factor.factor_sb_inverse import FactorSBInverse
from class_factor.factor_clust_volatility import FactorClustVolatility
from class_factor.factor_clust_volume import FactorClustVolume
from class_factor.factor_clust_load_volume import FactorClustLoadVolume

from trade_live.class_live.live_data import LiveData

class LiveCreate:
    def __init__(self,
                 threshold,
                 set_length,
                 update_crsp_price,
                 current_date,
                 start_data,
                 start_factor):
        
        '''
        threshold (int): Market cap threshold for preprocessing stocks
        set_length (int): Number of years required for stock to be considered for trading
        update_crsp_price (bool): Used to determine whether to update CRSP Price Dataset (should only be done at the start of every new year)
        current_date (str: YYYY-MM-DD): Current date of today (the end date for data retrieval)
        start_data (str: YYYY-MM-DD): Start date for data retrieval
        start_facdtor (str: YYYY-MM-DD): Start data fore factor creation
        '''

        self.threshold = threshold
        self.set_length = set_length
        self.update_crsp_price = update_crsp_price
        self.current_date = current_date
        self.start_data = start_data
        self.start_factor = start_factor

    def exec_data(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------GET LIVE DATA----------------------------------------------------------------------------------
        print("---------------------------------------------------------------------------GET LIVE DATA----------------------------------------------------------------------------------")
        live = True
        start_time = time.time()
        live_data = LiveData(live=live, start_date=self.start_data, current_date=self.current_date)

        if self.update_crsp_price:
            live_data.create_crsp_price(self.threshold, self.set_length)
        live_data.create_compustat_quarterly()
        live_data.create_compustat_annual()
        live_data.create_stock_list()
        live_data.create_live_price()
        live_data.create_misc()
        live_data.create_compustat_pension()
        live_data.create_industry()
        live_data.create_macro()
        live_data.create_risk_rate()
        live_data.create_etf()

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Total time to get live_trade data: {int(minutes)}:{int(seconds):02}")
        print("-" * 60)

    def exec_factor(self):
        print("----------------------------------------------------------------------EXEC FACTORS----------------------------------------------------------------------------------------")
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------CREATE FACTORS---------------------------------------------------------------------------------------
        live = True

        stock = read_stock(get_large(live) / 'permno_live.csv')

        ray.init(num_cpus=16, ignore_reinit_error=True)
        start_time = time.time()

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------GENERAL----------------------------------------------------------------------------------------
        FactorRet(live=live, file_name='factor_ret', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorRetComp(live=live, file_name='factor_ret_comp', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorVolatility(live=live, file_name='factor_volatility', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorVolume(live=live, file_name='factor_volume', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorTime(live=live, file_name='factor_time', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorTalib(live=live, file_name='factor_talib', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorSignRet(live=live, file_name='factor_sign_ret', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorVolComp(live=live, file_name='factor_vol_comp', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorSignVolume(live=live, file_name='factor_sign_volume', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorSignVolatility(live=live, file_name='factor_sign_volatility', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------
        FactorLoadRet(live=live, file_name='factor_load_ret', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
        FactorLoadVolume(live=live, file_name='factor_load_volume', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
        FactorLoadVolatility(live=live, file_name='factor_load_volatility', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------CONDITION-------------------------------------------------------------------------------------
        FactorCondRet(live=live, file_name='factor_cond_ret', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------INDUSTRY--------------------------------------------------------------------------------------
        FactorInd(live=live, file_name='factor_ind', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndFama(live=live, file_name='factor_ind_fama', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndSub(live=live, file_name='factor_ind_sub', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndMom(live=live, file_name='factor_ind_mom', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndMomFama(live=live, file_name='factor_ind_mom_fama', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndMomSub(live=live, file_name='factor_ind_mom_sub', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndVWR(live=live, file_name='factor_ind_vwr', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndVWRFama(live=live, file_name='factor_ind_vwr_fama', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorIndVWRSub(live=live, file_name='factor_ind_vwr_sub', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------OPEN ASSET------------------------------------------------------------------------------------
        FactorAccrual(live=live, file_name='factor_accrual', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorNetDebtFinance(live=live, file_name='factor_net_debt_finance', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorCHTax(live=live, file_name='factor_chtax', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorHire(live=live, file_name='factor_hire', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorAssetGrowth(live=live, file_name='factor_asset_growth', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorNOA(live=live, file_name='factor_noa', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorInvestPPEInv(live=live, file_name='factor_invest_ppe_inv', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorInvGrowth(live=live, file_name='factor_inv_growth', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorCompDebt(live=live, file_name='factor_comp_debt', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorChEQ(live=live, file_name='factor_cheq', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorXFIN(live=live, file_name='factor_xfin', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorEmmult(live=live, file_name='factor_emmult', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorAccrualBM(live=live, file_name='factor_accrual_bm', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorPctTotAcc(live=live, file_name='factor_pcttotacc', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorRDS(live=live, file_name='factor_rds', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorFrontier(live=live, file_name='factor_frontier', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorAgeMom(live=live, file_name='factor_age_mom', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomVol(live=live, file_name='factor_mom_vol', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorMomSeasonShort(live=live, file_name='factor_mom_season_short', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomSeason(live=live, file_name='factor_mom_season', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomSeason6(live=live, file_name='factor_mom_season6', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomSeason9(live=live, file_name='factor_mom_season9', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomSeason11(live=live, file_name='factor_mom_season11', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomSeason16(live=live, file_name='factor_mom_season16', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorIntMom(live=live, file_name='factor_int_mom', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorTrendFactor(live=live, file_name='factor_trend_factor', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorMomOffSeason(live=live, file_name='factor_mom_off_season', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomRev(live=live, file_name='factor_mom_rev', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorGrcapx(live=live, file_name='factor_grcapx', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorEarningStreak(live=live, file_name='factor_earning_streak', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorRetSkew(live=live, file_name='factor_ret_skew', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorDividend(live=live, file_name='factor_dividend', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMS(live=live, file_name='factor_ms', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorGradexp(live=live, file_name='factor_gradexp', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorSize(live=live, file_name='factor_size', skip=True, stock=stock, start=self.start_factor, end=self.current_date).create_factor()
        FactorRetMax(live=live, file_name='factor_ret_max', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomOffSeason6(live=live, file_name='factor_mom_off_season6', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorMomOffSeason11(live=live, file_name='factor_mom_off_season11', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------BETAS-----------------------------------------------------------------------------------------
        FactorSBSector(live=live, file_name='factor_sb_sector', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorSBPCA(live=live, file_name='factor_sb_pca', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        FactorSBInverse(live=live, file_name='factor_sb_inverse', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='permno').create_factor()
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------CLUSTER---------------------------------------------------------------------------------------
        FactorClustRet(live=live, file_name='factor_clust_ret', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
        FactorClustLoadRet(live=live, file_name='factor_clust_load_ret', skip=False, stock='all', start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
        FactorClustIndMom(live=live, file_name='factor_clust_ind_mom', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
        FactorClustIndMomSub(live=live, file_name='factor_clust_ind_mom_sub', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
        FactorClustIndMomFama(live=live, file_name='factor_clust_ind_mom_fama', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
        FactorClustLoadVolume(live=live, file_name='factor_clust_load_volume', skip=False, stock='all', start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
        FactorClustVolatility(live=live, file_name='factor_clust_volatility', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
        FactorClustVolume(live=live, file_name='factor_clust_volume', skip=False, stock=stock, start=self.start_factor, end=self.current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Total time to create all factors: {int(minutes)}:{int(seconds):02}")
        print("-" * 60)
        ray.shutdown()