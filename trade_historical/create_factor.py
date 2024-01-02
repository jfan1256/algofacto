from functions.utils.func import *

from class_factor.factor_macro import FactorMacro
from class_factor.factor_ind import FactorInd
from class_factor.factor_ind_mom import FactorIndMom
from class_factor.factor_ret import FactorRet
from class_factor.factor_rank_ret import FactorRankRet
from class_factor.factor_sign_ret import FactorSignRet
from class_factor.factor_volatility import FactorVolatility
from class_factor.factor_volume import FactorVolume
from class_factor.factor_time import FactorTime
from class_factor.factor_rf_ret import FactorRFRet
from class_factor.factor_sb_fama import FactorSBFama
from class_factor.factor_sb_inverse import FactorSBInverse
from class_factor.factor_talib import FactorTalib
from class_factor.factor_load_ret import FactorLoadRet
from class_factor.factor_clust_ret import FactorClustRet
from class_factor.factor_clust_load_ret import FactorClustLoadRet
from class_factor.factor_clust_volume import FactorClustVolume
from class_factor.factor_clust_ind_mom import FactorClustIndMom
from class_factor.factor_load_volatility import FactorLoadVolatility
from class_factor.factor_load_volume import FactorLoadVolume
from class_factor.factor_clust_volatility import FactorClustVolatility
from class_factor.factor_dividend import FactorDividend
from class_factor.factor_sb_sector import FactorSBSector
from class_factor.factor_sb_ind import FactorSBInd
from class_factor.factor_cond_ret import FactorCondRet
from class_factor.factor_sb_fund_raw import FactorSBFundRaw
from class_factor.factor_fund_raw import FactorFundRaw
from class_factor.factor_fund_q import FactorFundQ
from class_factor.factor_rank_fund_raw import FactorRankFundRaw
from class_factor.factor_clust_fund_raw import FactorClustFundRaw
from class_factor.factor_rank_volume import FactorRankVolume
from class_factor.factor_clust_load_volume import FactorClustLoadVolume
from class_factor.factor_rank_volatility import FactorRankVolatility
from class_factor.factor_cond_ind_mom import FactorCondIndMom
from class_factor.factor_ind_vwr import FactorIndVWR
from class_factor.factor_rank_ind_vwr import FactorRankIndVWR
from class_factor.factor_rank_ind_mom import FactorRankIndMom
from class_factor.factor_sb_bond import FactorSBBond
from class_factor.factor_sb_pca import FactorSBPCA
from class_factor.factor_ind_fama import FactorIndFama
from class_factor.factor_ind_mom_fama import FactorIndMomFama
from class_factor.factor_rank_ind_mom_fama import FactorRankIndMomFama
from class_factor.factor_clust_ind_mom_fama import FactorClustIndMomFama
from class_factor.factor_cond_ind_mom_fama import FactorCondIndMomFama
from class_factor.factor_ind_vwr_fama import FactorIndVWRFama
from class_factor.factor_rank_ind_vwr_fama import FactorRankIndVWRFama
from class_factor.factor_age_mom import FactorAgeMom
from class_factor.factor_net_debt_finance import FactorNetDebtFinance
from class_factor.factor_chtax import FactorCHTax
from class_factor.factor_asset_growth import FactorAssetGrowth
from class_factor.factor_mom_season_short import FactorMomSeasonShort
from class_factor.factor_mom_season import FactorMomSeason
from class_factor.factor_noa import FactorNOA
from class_factor.factor_invest_ppe_inv import FactorInvestPPEInv
from class_factor.factor_inv_growth import FactorInvGrowth
from class_factor.factor_earning_streak import FactorEarningStreak
from class_factor.factor_trend_factor import FactorTrendFactor
from class_factor.factor_ind_sub import FactorIndSub
from class_factor.factor_ind_mom_sub import FactorIndMomSub
from class_factor.factor_rank_ind_mom_sub import FactorRankIndMomSub
from class_factor.factor_clust_ind_mom_sub import FactorClustIndMomSub
from class_factor.factor_cond_ind_mom_sub import FactorCondIndMomSub
from class_factor.factor_ind_vwr_sub import FactorIndVWRSub
from class_factor.factor_rank_ind_vwr_sub import FactorRankIndVWRSub
from class_factor.factor_rank_load_ret import FactorRankLoadRet
from class_factor.factor_rank_load_volume import FactorRankLoadVolume
from class_factor.factor_comp_debt import FactorCompDebt
from class_factor.factor_mom_vol import FactorMomVol
from class_factor.factor_ms import FactorMS
from class_factor.factor_mom_season6 import FactorMomSeason6
from class_factor.factor_mom_season11 import FactorMomSeason11
from class_factor.factor_int_mom import FactorIntMom
from class_factor.factor_cheq import FactorChEQ
from class_factor.factor_xfin import FactorXFIN
from class_factor.factor_emmult import FactorEmmult
from class_factor.factor_accrual import FactorAccrual
from class_factor.factor_frontier import FactorFrontier
from class_factor.factor_abnormal_accrual import FactorAbnormalAccrual
from class_factor.factor_rank_sb_fama import FactorRankSBFama
from class_factor.factor_rank_sb_bond import FactorRankSBBond
from class_factor.factor_rank_sb_pca import FactorRankSBPCA
from class_factor.factor_rank_sb_inverse import FactorRankSBInverse
from class_factor.factor_ret_comp import FactorRetComp
from class_factor.factor_rank_ret_comp import FactorRankRetComp
from class_factor.factor_clust_ret_comp import FactorClustRetComp
from class_factor.factor_vol_comp import FactorVolComp
from class_factor.factor_grcapx import FactorGrcapx
from class_factor.factor_ret_skew import FactorRetSkew
from class_factor.factor_earning_disparity import FactorEarningDisparity
from class_factor.factor_gradexp import FactorGradexp
from class_factor.factor_mom_off_season import FactorMomOffSeason
from class_factor.factor_mom_off_season6 import FactorMomOffSeason6
from class_factor.factor_mom_off_season11 import FactorMomOffSeason11
from class_factor.factor_mom_season_shorter import FactorMomSeasonShorter
from class_factor.factor_mom_season16 import FactorMomSeason16
from class_factor.factor_mom_season21 import FactorMomSeason21
from class_factor.factor_mom_season9 import FactorMomSeason9
from class_factor.factor_div_season import FactorDivSeason
from class_factor.factor_accrual_bm import FactorAccrualBM
from class_factor.factor_sign_volume import FactorSignVolume
from class_factor.factor_sign_volatility import FactorSignVolatility
from class_factor.factor_size import FactorSize
from class_factor.factor_ret_max import FactorRetMax
from class_factor.factor_pcttotacc import FactorPctTotAcc
from class_factor.factor_clust_mom_season import FactorClustMomSeason
from class_factor.factor_rds import FactorRDS
from class_factor.factor_hire import FactorHire
from class_factor.factor_ind_mom_comp import FactorIndMomComp
from class_factor.factor_mom_rev import FactorMomRev
from class_factor.factor_sb_pca_copy import FactorSBPCACopy
from class_factor.factor_sb_sector_copy import FactorSBSectorCopy


start = '2005-01-01'
end = date.today().strftime('%Y-%m-%d')
live = False

if live:
    stock = read_stock(get_large(live) / 'permno_live.csv')
else:
    stock = read_stock(get_large(live) / 'permno_to_train_fund.csv')

ray.init(num_cpus=16, ignore_reinit_error=True)
start_time = time.time()

# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------GENERAL----------------------------------------------------------------------------------------
FactorRet(live=live, file_name='factor_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRetComp(trade_live=trade_live, file_name='factor_ret_comp', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolComp(trade_live=trade_live, file_name='factor_vol_comp', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignRet(trade_live=trade_live, file_name='factor_sign_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolatility(trade_live=trade_live, file_name='factor_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignVolume(trade_live=trade_live, file_name='factor_sign_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignVolatility(trade_live=trade_live, file_name='factor_sign_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolume(trade_live=trade_live, file_name='factor_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTime(trade_live=trade_live, file_name='factor_time', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMacro(trade_live=trade_live, file_name='factor_macro', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno', general=True).create_factor()
# FactorTalib(trade_live=trade_live, file_name='factor_talib', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorFundRaw(live=live, file_name='factor_fund_raw', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorFundQ(trade_live=trade_live, file_name='factor_fund_q', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------
# FactorLoadRet(trade_live=trade_live, file_name='factor_load_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# FactorLoadVolume(trade_live=trade_live, file_name='factor_load_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# FactorLoadVolatility(trade_live=trade_live, file_name='factor_load_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------CONDITION-------------------------------------------------------------------------------------
# FactorCondRet(trade_live=trade_live, file_name='factor_cond_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorCondIndMom(trade_live=trade_live, file_name='factor_cond_ind_mom', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCondIndMomFama(trade_live=trade_live, file_name='factor_cond_ind_mom_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCondIndMomSub(trade_live=trade_live, file_name='factor_cond_ind_mom_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------INDUSTRY--------------------------------------------------------------------------------------
# FactorInd(trade_live=trade_live, file_name='factor_ind', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndFama(trade_live=trade_live, file_name='factor_ind_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndSub(trade_live=trade_live, file_name='factor_ind_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndMom(trade_live=trade_live, file_name='factor_ind_mom', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndMomFama(trade_live=trade_live, file_name='factor_ind_mom_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndMomSub(trade_live=trade_live, file_name='factor_ind_mom_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndVWR(trade_live=trade_live, file_name='factor_ind_vwr', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndVWRFama(trade_live=trade_live, file_name='factor_ind_vwr_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndVWRSub(trade_live=trade_live, file_name='factor_ind_vwr_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndMomComp(trade_live=trade_live, file_name='factor_ind_mom_comp', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------OPEN ASSET------------------------------------------------------------------------------------
# FactorAccrual(trade_live=trade_live, file_name='factor_accrual', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorNetDebtFinance(trade_live=trade_live, file_name='factor_net_debt_finance', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCHTax(trade_live=trade_live, file_name='factor_chtax', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorHire(trade_live=trade_live, file_name='factor_hire', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorAssetGrowth(trade_live=trade_live, file_name='factor_asset_growth', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorNOA(trade_live=trade_live, file_name='factor_noa', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorInvestPPEInv(trade_live=trade_live, file_name='factor_invest_ppe_inv', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorInvGrowth(trade_live=trade_live, file_name='factor_inv_growth', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCompDebt(trade_live=trade_live, file_name='factor_comp_debt', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorChEQ(trade_live=trade_live, file_name='factor_cheq', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorXFIN(trade_live=trade_live, file_name='factor_xfin', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorEmmult(trade_live=trade_live, file_name='factor_emmult', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorGrcapx(trade_live=trade_live, file_name='factor_grcapx', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorGradexp(trade_live=trade_live, file_name='factor_gradexp', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorAccrualBM(trade_live=trade_live, file_name='factor_accrual_bm', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorPctTotAcc(trade_live=trade_live, file_name='factor_pcttotacc', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRDS(trade_live=trade_live, file_name='factor_rds', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorFrontier(trade_live=trade_live, file_name='factor_frontier', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorEarningStreak(trade_live=trade_live, file_name='factor_earning_streak', skip=True, stock=stock, start=start, end=end).create_factor()
FactorDividend(live=live, file_name='factor_dividend', skip=False, stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorAgeMom(trade_live=trade_live, file_name='factor_age_mom', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomVol(trade_live=trade_live, file_name='factor_mom_vol', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMS(trade_live=trade_live, file_name='factor_ms', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMomSeasonShort(trade_live=trade_live, file_name='factor_mom_season_short', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason(trade_live=trade_live, file_name='factor_mom_season', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason6(trade_live=trade_live, file_name='factor_mom_season6', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason9(trade_live=trade_live, file_name='factor_mom_season9', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason11(trade_live=trade_live, file_name='factor_mom_season11', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason16(trade_live=trade_live, file_name='factor_mom_season16', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason21(trade_live=trade_live, file_name='factor_mom_season21', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorIntMom(trade_live=trade_live, file_name='factor_int_mom', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTrendFactor(trade_live=trade_live, file_name='factor_trend_factor', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorEarningDisparity(trade_live=trade_live, file_name='factor_earning_disparity', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMomOffSeason(trade_live=trade_live, file_name='factor_mom_off_season', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomOffSeason6(trade_live=trade_live, file_name='factor_mom_off_season6', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomOffSeason11(trade_live=trade_live, file_name='factor_mom_off_season11', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeasonShorter(trade_live=trade_live, file_name='factor_mom_season_shorter', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorDivSeason(trade_live=trade_live, file_name='factor_div_season', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorSize(trade_live=trade_live, file_name='factor_size', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRetSkew(trade_live=trade_live, file_name='factor_ret_skew', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRetMax(trade_live=trade_live, file_name='factor_ret_max', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorAbnormalAccrual(trade_live=trade_live, file_name='factor_abnormal_accrual', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMomRev(trade_live=trade_live, file_name='factor_mom_rev', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------BETAS-----------------------------------------------------------------------------------------
FactorSBSector(live=live, file_name='factor_sb_sector', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorSBFama(live=live, file_name='factor_sb_fama', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorSBPCA(live=live, file_name='factor_sb_pca', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorSBBond(live=live, file_name='factor_sb_bond', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorSBInverse(live=live, file_name='factor_sb_inverse', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBPCACopy(trade_live=trade_live, file_name='factor_sb_pca_copy', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBSectorCopy(trade_live=trade_live, file_name='factor_sb_sector_copy', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------RANK-----------------------------------------------------------------------------------------
# FactorRankRet(trade_live=trade_live, file_name='factor_rank_ret', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankFundRaw(trade_live=trade_live, file_name='factor_rank_fund_raw', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankVolume(trade_live=trade_live, file_name='factor_rank_volume', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankVolatility(trade_live=trade_live, file_name='factor_rank_volatility', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndVWR(trade_live=trade_live, file_name='factor_rank_ind_vwr', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndVWRFama(trade_live=trade_live, file_name='factor_rank_ind_vwr_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndVWRSub(trade_live=trade_live, file_name='factor_rank_ind_vwr_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndMom(trade_live=trade_live, file_name='factor_rank_ind_mom', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndMomFama(trade_live=trade_live, file_name='factor_rank_ind_mom_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndMomSub(trade_live=trade_live, file_name='factor_rank_ind_mom_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankLoadRet(trade_live=trade_live, file_name='factor_rank_load_ret', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankLoadVolume(trade_live=trade_live, file_name='factor_rank_load_volume', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBFama(trade_live=trade_live, file_name='factor_rank_sb_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBBond(trade_live=trade_live, file_name='factor_rank_sb_bond', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBPCA(trade_live=trade_live, file_name='factor_rank_sb_pca', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBInverse(trade_live=trade_live, file_name='factor_rank_sb_inverse', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankRetComp(trade_live=trade_live, file_name='factor_rank_ret_comp', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------CLUSTER---------------------------------------------------------------------------------------
# FactorClustRet(trade_live=trade_live, file_name='factor_clust_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustVolume(trade_live=trade_live, file_name='factor_clust_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustVolatility(trade_live=trade_live, file_name='factor_clust_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustRetComp(trade_live=trade_live, file_name='factor_clust_ret_comp', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustLoadRet(trade_live=trade_live, file_name='factor_clust_load_ret', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustFundRaw(trade_live=trade_live, file_name='factor_clust_fund_raw', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=1, cluster=21).create_factor()
# FactorClustLoadVolume(trade_live=trade_live, file_name='factor_clust_load_volume', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMom(trade_live=trade_live, file_name='factor_clust_ind_mom', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMomSub(trade_live=trade_live, file_name='factor_clust_ind_mom_sub', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMomFama(trade_live=trade_live, file_name='factor_clust_ind_mom_fama', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustMomSeason(trade_live=trade_live, file_name='factor_clust_mom_season', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------FACTORS TO SKIP----------------------------------------------------------------------------------
# FactorSBInd(trade_live=trade_live, file_name='factor_sb_ind', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRFRet(trade_live=trade_live, file_name='factor_rf_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBFundRaw(trade_live=trade_live, file_name='factor_sb_fund_raw', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()

elapsed_time = time.time() - start_time
print(f"Total time to create all factors: {round(elapsed_time)} seconds")
print("-" * 60)
ray.shutdown()
