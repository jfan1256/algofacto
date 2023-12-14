from functions.utils.func import *

from factor_class.factor_ind import FactorInd
from factor_class.factor_ind_mom import FactorIndMom
from factor_class.factor_ret import FactorRet
from factor_class.factor_volatility import FactorVolatility
from factor_class.factor_volume import FactorVolume
from factor_class.factor_time import FactorTime
from factor_class.factor_talib import FactorTalib
from factor_class.factor_load_ret import FactorLoadRet
from factor_class.factor_clust_ret import FactorClustRet
from factor_class.factor_clust_load_ret import FactorClustLoadRet
from factor_class.factor_clust_ind_mom import FactorClustIndMom
from factor_class.factor_load_volume import FactorLoadVolume
from factor_class.factor_sb_sector import FactorSBSector
from factor_class.factor_sb_pca import FactorSBPCA
from factor_class.factor_ind_fama import FactorIndFama
from factor_class.factor_ind_mom_fama import FactorIndMomFama
from factor_class.factor_clust_ind_mom_fama import FactorClustIndMomFama
from factor_class.factor_age_mom import FactorAgeMom
from factor_class.factor_net_debt_finance import FactorNetDebtFinance
from factor_class.factor_chtax import FactorCHTax
from factor_class.factor_asset_growth import FactorAssetGrowth
from factor_class.factor_mom_season_short import FactorMomSeasonShort
from factor_class.factor_mom_season import FactorMomSeason
from factor_class.factor_noa import FactorNOA
from factor_class.factor_invest_ppe_inv import FactorInvestPPEInv
from factor_class.factor_inv_growth import FactorInvGrowth
from factor_class.factor_trend_factor import FactorTrendFactor
from factor_class.factor_ind_sub import FactorIndSub
from factor_class.factor_ind_mom_sub import FactorIndMomSub
from factor_class.factor_clust_ind_mom_sub import FactorClustIndMomSub
from factor_class.factor_comp_debt import FactorCompDebt
from factor_class.factor_mom_vol import FactorMomVol
from factor_class.factor_mom_season6 import FactorMomSeason6
from factor_class.factor_mom_season11 import FactorMomSeason11
from factor_class.factor_int_mom import FactorIntMom
from factor_class.factor_cheq import FactorChEQ
from factor_class.factor_xfin import FactorXFIN
from factor_class.factor_emmult import FactorEmmult
from factor_class.factor_accrual import FactorAccrual
from factor_class.factor_frontier import FactorFrontier
from factor_class.factor_ret_comp import FactorRetComp
from factor_class.factor_mom_off_season import FactorMomOffSeason
from factor_class.factor_mom_season16 import FactorMomSeason16
from factor_class.factor_mom_season9 import FactorMomSeason9
from factor_class.factor_accrual_bm import FactorAccrualBM
from factor_class.factor_pcttotacc import FactorPctTotAcc
from factor_class.factor_rds import FactorRDS
from factor_class.factor_hire import FactorHire
from factor_class.factor_mom_rev import FactorMomRev
from factor_class.factor_grcapx import FactorGrcapx
from factor_class.factor_earning_streak import FactorEarningStreak
from factor_class.factor_sign_ret import FactorSignRet
from factor_class.factor_cond_ret import FactorCondRet
from factor_class.factor_ret_skew import FactorRetSkew
from factor_class.factor_dividend import FactorDividend
from factor_class.factor_vol_comp import FactorVolComp

def exec_factor(start_factor):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------CREATE FACTORS---------------------------------------------------------------------------------------
    print("---------------------------------------------------------------------CREATE FACTORS---------------------------------------------------------------------------------------")
    live = True
    current_date = (date.today()).strftime('%Y-%m-%d')

    stock = read_stock(get_large_dir(live) / 'permno_live.csv')

    ray.init(num_cpus=16, ignore_reinit_error=True)
    start_time = time.time()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------GENERAL----------------------------------------------------------------------------------------
    FactorRet(live=live, file_name='factor_ret', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorRetComp(live=live, file_name='factor_ret_comp', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorVolatility(live=live, file_name='factor_volatility', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorVolume(live=live, file_name='factor_volume', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorTime(live=live, file_name='factor_time', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorTalib(live=live, file_name='factor_talib', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorSignRet(live=live, file_name='factor_sign_ret', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorVolComp(live=live, file_name='factor_vol_comp', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------
    FactorLoadRet(live=live, file_name='factor_load_ret', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
    FactorLoadVolume(live=live, file_name='factor_load_volume', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------CONDITION-------------------------------------------------------------------------------------
    FactorCondRet(live=live, file_name='factor_cond_ret', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------INDUSTRY--------------------------------------------------------------------------------------
    FactorInd(live=live, file_name='factor_ind', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorIndFama(live=live, file_name='factor_ind_fama', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorIndSub(live=live, file_name='factor_ind_sub', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorIndMom(live=live, file_name='factor_ind_mom', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorIndMomFama(live=live, file_name='factor_ind_mom_fama', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorIndMomSub(live=live, file_name='factor_ind_mom_sub', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------OPEN ASSET------------------------------------------------------------------------------------
    FactorAccrual(live=live, file_name='factor_accrual', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorNetDebtFinance(live=live, file_name='factor_net_debt_finance', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorCHTax(live=live, file_name='factor_chtax', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorHire(live=live, file_name='factor_hire', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorAssetGrowth(live=live, file_name='factor_asset_growth', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorNOA(live=live, file_name='factor_noa', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorInvestPPEInv(live=live, file_name='factor_invest_ppe_inv', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorInvGrowth(live=live, file_name='factor_inv_growth', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorCompDebt(live=live, file_name='factor_comp_debt', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorChEQ(live=live, file_name='factor_cheq', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorXFIN(live=live, file_name='factor_xfin', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorEmmult(live=live, file_name='factor_emmult', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorAccrualBM(live=live, file_name='factor_accrual_bm', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorPctTotAcc(live=live, file_name='factor_pcttotacc', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorRDS(live=live, file_name='factor_rds', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorFrontier(live=live, file_name='factor_frontier', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorAgeMom(live=live, file_name='factor_age_mom', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorMomVol(live=live, file_name='factor_mom_vol', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorMomSeasonShort(live=live, file_name='factor_mom_season_short', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorMomSeason(live=live, file_name='factor_mom_season', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorMomSeason6(live=live, file_name='factor_mom_season6', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorMomSeason9(live=live, file_name='factor_mom_season9', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorMomSeason11(live=live, file_name='factor_mom_season11', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorMomSeason16(live=live, file_name='factor_mom_season16', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorIntMom(live=live, file_name='factor_int_mom', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorTrendFactor(live=live, file_name='factor_trend_factor', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorMomOffSeason(live=live, file_name='factor_mom_off_season', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorMomRev(live=live, file_name='factor_mom_rev', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorGrcapx(live=live, file_name='factor_grcapx', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorEarningStreak(live=live, file_name='factor_earning_streak', skip=True, stock=stock, start=start_factor, end=current_date).create_factor()
    FactorRetSkew(live=live, file_name='factor_ret_skew', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorDividend(live=live, file_name='factor_dividend', skip=False, stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------BETAS-----------------------------------------------------------------------------------------
    FactorSBSector(live=live, file_name='factor_sb_sector', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    FactorSBPCA(live=live, file_name='factor_sb_pca', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------CLUSTER---------------------------------------------------------------------------------------
    FactorClustRet(live=live, file_name='factor_clust_ret', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
    FactorClustLoadRet(live=live, file_name='factor_clust_load_ret', stock='all', start=start_factor, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
    FactorClustIndMom(live=live, file_name='factor_clust_ind_mom', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
    FactorClustIndMomSub(live=live, file_name='factor_clust_ind_mom_sub', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
    FactorClustIndMomFama(live=live, file_name='factor_clust_ind_mom_fama', stock=stock, start=start_factor, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total time to create all factors: {int(minutes)}:{int(seconds):02}")
    print("-" * 60)
    ray.shutdown()