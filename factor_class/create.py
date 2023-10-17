from functions.utils.func import *

from factor_class.factor_macro import FactorMacro
from factor_class.factor_ind import FactorInd
from factor_class.factor_ind_mom import FactorIndMom
from factor_class.factor_ret import FactorRet
from factor_class.factor_rank_ret import FactorRankRet
from factor_class.factor_sign_ret import FactorSignRet
from factor_class.factor_volatility import FactorVolatility
from factor_class.factor_volume import FactorVolume
from factor_class.factor_time import FactorTime
from factor_class.factor_rf_ret import FactorRFRet
from factor_class.factor_rf_sign import FactorRFSign
from factor_class.factor_rf_volume import FactorRFVolume
from factor_class.factor_sb_etf import FactorSBETF
from factor_class.factor_sb_fama import FactorSBFama
from factor_class.factor_sb_inverse import FactorSBInverse
from factor_class.factor_talib import FactorTalib
from factor_class.factor_open_asset import FactorOpenAsset
from factor_class.factor_load_ret import FactorLoadRet
from factor_class.factor_clust_ret import FactorClustRet
from factor_class.factor_clust_load_ret import FactorClustLoadRet
from factor_class.factor_sb_macro import FactorSBMacro
from factor_class.factor_clust_volume import FactorClustVolume
from factor_class.factor_clust_ind_mom import FactorClustIndMom
from factor_class.factor_load_volatility import FactorLoadVolatility
from factor_class.factor_load_volume import FactorLoadVolume
from factor_class.factor_clust_volatility import FactorClustVolatility
from factor_class.factor_dividend import FactorDividend
from factor_class.factor_sb_oil import FactorSBOil
from factor_class.factor_sb_sector import FactorSBSector
from factor_class.factor_sb_ind import FactorSBInd
from factor_class.factor_sb_overall import FactorSBOverall
from factor_class.factor_cond_ret import FactorCondRet
from factor_class.factor_sb_fund_ind import FactorSBFundInd
from factor_class.factor_sb_fund_raw import FactorSBFundRaw
from factor_class.factor_fund_raw import FactorFundRaw
from factor_class.factor_fund_q import FactorFundQ
from factor_class.factor_rank_fund_raw import FactorRankFundRaw
from factor_class.factor_clust_fund_raw import FactorClustFundRaw
from factor_class.factor_high import FactorHigh
from factor_class.factor_low import FactorLow
from factor_class.factor_rank_volume import FactorRankVolume
from factor_class.factor_clust_load_volume import FactorClustLoadVolume
from factor_class.factor_total import FactorTotal
from factor_class.factor_rank_volatility import FactorRankVolatility
from factor_class.factor_cond_ind_mom import FactorCondIndMom
from factor_class.factor_ind_vwr import FactorIndVWR
from factor_class.factor_rank_ind_vwr import FactorRankIndVWR
from factor_class.factor_rank_ind_mom import FactorRankIndMom
from factor_class.factor_sb_bond import FactorSBBond
from factor_class.factor_sb_pca import FactorSBPCA
from factor_class.factor_ind_fama import FactorIndFama
from factor_class.factor_ind_mom_fama import FactorIndMomFama
from factor_class.factor_rank_ind_mom_fama import FactorRankIndMomFama
from factor_class.factor_clust_ind_mom_fama import FactorClustIndMomFama
from factor_class.factor_cond_ind_mom_fama import FactorCondIndMomFama
from factor_class.factor_ind_vwr_fama import FactorIndVWRFama
from factor_class.factor_rank_ind_vwr_fama import FactorRankIndVWRFama
from factor_class.factor_age_mom import FactorAgeMom
from factor_class.factor_net_debt_finance import FactorNetDebtFinance
from factor_class.factor_chtax import FactorCHTax
from factor_class.factor_asset_growth import FactorAssetGrowth
from factor_class.factor_mom_season_short import FactorMomSeasonShort
from factor_class.factor_mom_season import FactorMomSeason
from factor_class.factor_noa import FactorNOA
from factor_class.factor_invest_ppe_inv import FactorInvestPPEInv
from factor_class.factor_inv_growth import FactorInvGrowth
from factor_class.factor_earning_streak import FactorEarningStreak
from factor_class.factor_trend_factor import FactorTrendFactor
from factor_class.factor_ind_sub import FactorIndSub
from factor_class.factor_ind_mom_sub import FactorIndMomSub
from factor_class.factor_rank_ind_mom_sub import FactorRankIndMomSub
from factor_class.factor_clust_ind_mom_sub import FactorClustIndMomSub
from factor_class.factor_cond_ind_mom_sub import FactorCondIndMomSub
from factor_class.factor_ind_vwr_sub import FactorIndVWRSub
from factor_class.factor_rank_ind_vwr_sub import FactorRankIndVWRSub
from factor_class.factor_rank_load_ret import FactorRankLoadRet
from factor_class.factor_rank_load_volume import FactorRankLoadVolume
from factor_class.factor_comp_debt import FactorCompDebt
from factor_class.factor_mom_vol import FactorMomVol
from factor_class.factor_ms import FactorMS
from factor_class.factor_mom_season6 import FactorMomSeason6
from factor_class.factor_mom_season11 import FactorMomSeason11
from factor_class.factor_int_mom import FactorIntMom
from factor_class.factor_cheq import FactorChEQ
from factor_class.factor_xfin import FactorXFIN
from factor_class.factor_emmult import FactorEmmult
from factor_class.factor_accrual import FactorAccrual
from factor_class.factor_frontier import FactorFrontier
from factor_class.factor_abnormal_accrual import FactorAbnormalAccrual
from factor_class.factor_rank_sb_fama import FactorRankSBFama
from factor_class.factor_rank_sb_bond import FactorRankSBBond
from factor_class.factor_rank_sb_pca import FactorRankSBPCA
from factor_class.factor_rank_sb_inverse import FactorRankSBInverse
from factor_class.factor_ret_comp import FactorRetComp
from factor_class.factor_rank_ret_comp import FactorRankRetComp
from factor_class.factor_clust_ret_comp import FactorClustRetComp
from factor_class.factor_vol_comp import FactorVolComp
from factor_class.factor_ind_naic import FactorIndNAIC
from factor_class.factor_resid_mom import FactorResidMom
from factor_class.factor_grcapx import FactorGrcapx
from factor_class.factor_ret_skew import FactorRetSkew
from factor_class.factor_earning_disparity import FactorEarningDisparity
from factor_class.factor_gradexp import FactorGradexp
from factor_class.factor_mom_off_season import FactorMomOffSeason
from factor_class.factor_mom_off_season6 import FactorMomOffSeason6
from factor_class.factor_mom_off_season11 import FactorMomOffSeason11
from factor_class.factor_mom_season_shorter import FactorMomSeasonShorter
from factor_class.factor_mom_season16 import FactorMomSeason16
from factor_class.factor_mom_season21 import FactorMomSeason21
from factor_class.factor_mom_season9 import FactorMomSeason9
from factor_class.factor_div_season import FactorDivSeason
from factor_class.factor_accrual_bm import FactorAccrualBM
from factor_class.factor_sign_volume import FactorSignVolume
from factor_class.factor_sign_volatility import FactorSignVolatility
from factor_class.factor_size import FactorSize
from factor_class.factor_ret_max import FactorRetMax
from factor_class.factor_pcttotacc import FactorPctTotAcc
from factor_class.factor_clust_mom_season import FactorClustMomSeason
from factor_class.factor_rds import FactorRDS
from factor_class.factor_hire import FactorHire
from factor_class.factor_ind_mom_comp import FactorIndMomComp
from factor_class.factor_mom_rev import FactorMomRev


start = '2005-01-01'
end = '2023-01-01'
permno = read_stock(get_load_data_large_dir() / 'permno_to_train_fund.csv')

ray.init(num_cpus=16, ignore_reinit_error=True)
start_time = time.time()

# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------GENERAL----------------------------------------------------------------------------------------
# FactorRet(file_name='factor_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRetComp(file_name='factor_ret_comp', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolComp(file_name='factor_vol_comp', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignRet(file_name='factor_sign_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolatility(file_name='factor_volatility', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignVolume(file_name='factor_sign_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignVolatility(file_name='factor_sign_volatility', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolume(file_name='factor_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorTime(file_name='factor_time', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMacro(file_name='factor_macro', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno', general=True).create_factor()
# FactorTalib(file_name='factor_talib', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorFundRaw(file_name='factor_fund_raw', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorFundQ(file_name='factor_fund_q', skip=True, stock=permno, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------
# FactorLoadRet(file_name='factor_load_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# FactorLoadVolume(file_name='factor_load_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# FactorLoadVolatility(file_name='factor_load_volatility', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------CONDITION-------------------------------------------------------------------------------------
# FactorCondRet(file_name='factor_cond_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorCondIndMom(file_name='factor_cond_ind_mom', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorCondIndMomFama(file_name='factor_cond_ind_mom_fama', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorCondIndMomSub(file_name='factor_cond_ind_mom_sub', skip=True, stock=permno, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------INDUSTRY--------------------------------------------------------------------------------------
# FactorInd(file_name='factor_ind', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndFama(file_name='factor_ind_fama', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndSub(file_name='factor_ind_sub', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndNAIC(file_name='factor_ind_naic', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndMom(file_name='factor_ind_mom', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndMomFama(file_name='factor_ind_mom_fama', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndMomSub(file_name='factor_ind_mom_sub', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndVWR(file_name='factor_ind_vwr', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndVWRFama(file_name='factor_ind_vwr_fama', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndVWRSub(file_name='factor_ind_vwr_sub', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorIndMomComp(file_name='factor_ind_mom_comp', skip=True, stock=permno, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------OPEN ASSET------------------------------------------------------------------------------------
# FactorAccrual(file_name='factor_accrual', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorNetDebtFinance(file_name='factor_net_debt_finance', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorCHTax(file_name='factor_chtax', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorHire(file_name='factor_hire', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorAssetGrowth(file_name='factor_asset_growth', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorNOA(file_name='factor_noa', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorInvestPPEInv(file_name='factor_invest_ppe_inv', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorInvGrowth(file_name='factor_inv_growth', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorCompDebt(file_name='factor_comp_debt', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorChEQ(file_name='factor_cheq', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorXFIN(file_name='factor_xfin', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorEmmult(file_name='factor_emmult', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorGrcapx(file_name='factor_grcapx', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorGradexp(file_name='factor_gradexp', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorAccrualBM(file_name='factor_accrual_bm', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorPctTotAcc(file_name='factor_pcttotacc', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRDS(file_name='factor_rds', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorFrontier(file_name='factor_frontier', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorEarningStreak(file_name='factor_earning_streak', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorDividend(file_name='factor_dividend', skip=False, stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorAgeMom(file_name='factor_age_mom', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomVol(file_name='factor_mom_vol', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorMS(file_name='factor_ms', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorMomSeasonShort(file_name='factor_mom_season_short', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason(file_name='factor_mom_season', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason6(file_name='factor_mom_season6', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason9(file_name='factor_mom_season9', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason11(file_name='factor_mom_season11', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason16(file_name='factor_mom_season16', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason21(file_name='factor_mom_season21', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorIntMom(file_name='factor_int_mom', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTrendFactor(file_name='factor_trend_factor', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorResidMom(file_name='factor_resid_mom', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorEarningDisparity(file_name='factor_earning_disparity', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorMomOffSeason(file_name='factor_mom_off_season', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomOffSeason6(file_name='factor_mom_off_season6', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomOffSeason11(file_name='factor_mom_off_season11', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeasonShorter(file_name='factor_mom_season_shorter', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorDivSeason(file_name='factor_div_season', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorSize(file_name='factor_size', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRetSkew(file_name='factor_ret_skew', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRetMax(file_name='factor_ret_max', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorAbnormalAccrual(file_name='factor_abnormal_accrual', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorMomRev(file_name='factor_mom_rev', skip=True, stock=permno, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------BETAS-----------------------------------------------------------------------------------------
# FactorSBSector(file_name='factor_sb_sector', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBFama(file_name='factor_sb_fama', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBPCA(file_name='factor_sb_pca', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBBond(file_name='factor_sb_bond', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBInverse(file_name='factor_sb_inverse', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------RANK-----------------------------------------------------------------------------------------
# FactorRankRet(file_name='factor_rank_ret', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankFundRaw(file_name='factor_rank_fund_raw', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankVolume(file_name='factor_rank_volume', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankVolatility(file_name='factor_rank_volatility', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankIndVWR(file_name='factor_rank_ind_vwr', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankIndVWRFama(file_name='factor_rank_ind_vwr_fama', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankIndVWRSub(file_name='factor_rank_ind_vwr_sub', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankIndMom(file_name='factor_rank_ind_mom', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankIndMomFama(file_name='factor_rank_ind_mom_fama', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankIndMomSub(file_name='factor_rank_ind_mom_sub', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankLoadRet(file_name='factor_rank_load_ret', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankLoadVolume(file_name='factor_rank_load_volume', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankSBFama(file_name='factor_rank_sb_fama', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankSBBond(file_name='factor_rank_sb_bond', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankSBPCA(file_name='factor_rank_sb_pca', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankSBInverse(file_name='factor_rank_sb_inverse', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankRetComp(file_name='factor_rank_ret_comp', skip=True, stock=permno, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------CLUSTER---------------------------------------------------------------------------------------
# FactorClustRet(file_name='factor_clust_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustVolume(file_name='factor_clust_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustVolatility(file_name='factor_clust_volatility', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustRetComp(file_name='factor_clust_ret_comp', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustLoadRet(file_name='factor_clust_load_ret', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustFundRaw(file_name='factor_clust_fund_raw', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=1, cluster=21).create_factor()
# FactorClustLoadVolume(file_name='factor_clust_load_volume', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMom(file_name='factor_clust_ind_mom', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMomSub(file_name='factor_clust_ind_mom_sub', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMomFama(file_name='factor_clust_ind_mom_fama', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustMomSeason(file_name='factor_clust_mom_season', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------FACTORS TO SKIP----------------------------------------------------------------------------------
# FactorHigh(file_name='factor_high', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorLow(file_name='factor_low', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTotal(file_name='factor_total', skip=True, stock=permno, start=start, end=end, group='permno').create_factor()
# FactorSBOverall(file_name='factor_sb_overall', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBInd(file_name='factor_sb_ind', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRFRet(file_name='factor_rf_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorClustRet30(file_name='factor_clust_ret30', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustRet60(file_name='factor_clust_ret60', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorSBETF(file_name='factor_sb_etf', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBMacro(file_name='factor_sb_macro', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRFSign(file_name='factor_rf_sign', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRFVolume(file_name='factor_rf_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBOil(file_name='factor_sb_oil', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorOpenAsset(file_name='factor_open_asset', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorSBFundInd(file_name='factor_sb_fund_ind', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBFundRaw(file_name='factor_sb_fund_raw', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorEPBond(file_name='factor_ep_bond', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorEPETF(file_name='factor_ep_etf', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorEPFama(file_name='factor_ep_fama', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()

elapsed_time = time.time() - start_time
print(f"Total time to create all factors: {round(elapsed_time)} seconds")
print("-" * 60)
ray.shutdown()
