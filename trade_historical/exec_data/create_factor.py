from class_factor.factor_ind_mom import FactorIndMom
from class_factor.factor_ind_mom_excess import FactorIndMomExcess
from class_factor.factor_ret import FactorRet
from class_factor.factor_sb_fama import FactorSBFama
from class_factor.factor_sb_inverse import FactorSBInverse
from class_factor.factor_dividend import FactorDividend
from class_factor.factor_sb_sector import FactorSBSector
from class_factor.factor_fund_raw import FactorFundRaw
from class_factor.factor_sb_bond import FactorSBBond
from class_factor.factor_sb_pca import FactorSBPCA
from class_factor.factor_load_ret import FactorLoadRet

from core.operation import *

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
# FactorRet(live=live, file_name='factor_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRetComp(live=live, file_name='factor_ret_comp', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolComp(live=live, file_name='factor_vol_comp', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignRet(live=live, file_name='factor_sign_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolatility(live=live, file_name='factor_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignVolume(live=live, file_name='factor_sign_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSignVolatility(live=live, file_name='factor_sign_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolume(live=live, file_name='factor_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTime(live=live, file_name='factor_time', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMacro(live=live, file_name='factor_macro', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno', general=True).create_factor()
# FactorTalib(live=live, file_name='factor_talib', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorFundRaw(live=live, file_name='factor_fund_raw', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorFundQ(live=live, file_name='factor_fund_q', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------
# FactorLoadRet(live=live, file_name='factor_load_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# FactorLoadVolume(live=live, file_name='factor_load_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# FactorLoadVolatility(live=live, file_name='factor_load_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------CONDITION-------------------------------------------------------------------------------------
# FactorCondRet(live=live, file_name='factor_cond_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorCondIndMom(live=live, file_name='factor_cond_ind_mom', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCondIndMomFama(live=live, file_name='factor_cond_ind_mom_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCondIndMomSub(live=live, file_name='factor_cond_ind_mom_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------INDUSTRY--------------------------------------------------------------------------------------
# FactorInd(live=live, file_name='factor_ind', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndFama(live=live, file_name='factor_ind_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndSub(live=live, file_name='factor_ind_sub', skip=True, stock=stock, start=start, end=end).create_factor()
FactorIndMom(live=live, file_name='factor_ind_mom', skip=True, stock=stock, start=start, end=end).create_factor()
FactorIndMomExcess(live=live, file_name='factor_ind_mom_excess', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndMomFama(live=live, file_name='factor_ind_mom_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndMomSub(live=live, file_name='factor_ind_mom_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndVWR(live=live, file_name='factor_ind_vwr', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndVWRFama(live=live, file_name='factor_ind_vwr_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndVWRSub(live=live, file_name='factor_ind_vwr_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorIndMomComp(live=live, file_name='factor_ind_mom_comp', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------OPEN ASSET------------------------------------------------------------------------------------
# FactorAccrual(live=live, file_name='factor_accrual', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorNetDebtFinance(live=live, file_name='factor_net_debt_finance', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCHTax(live=live, file_name='factor_chtax', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorHire(live=live, file_name='factor_hire', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorAssetGrowth(live=live, file_name='factor_asset_growth', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorNOA(live=live, file_name='factor_noa', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorInvestPPEInv(live=live, file_name='factor_invest_ppe_inv', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorInvGrowth(live=live, file_name='factor_inv_growth', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorCompDebt(live=live, file_name='factor_comp_debt', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorChEQ(live=live, file_name='factor_cheq', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorXFIN(live=live, file_name='factor_xfin', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorEmmult(live=live, file_name='factor_emmult', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorGrcapx(live=live, file_name='factor_grcapx', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorGradexp(live=live, file_name='factor_gradexp', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorAccrualBM(live=live, file_name='factor_accrual_bm', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorPctTotAcc(live=live, file_name='factor_pcttotacc', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRDS(live=live, file_name='factor_rds', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorFrontier(live=live, file_name='factor_frontier', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorEarningStreak(live=live, file_name='factor_earning_streak', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorDividend(live=live, file_name='factor_dividend', skip=False, stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorAgeMom(live=live, file_name='factor_age_mom', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomVol(live=live, file_name='factor_mom_vol', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMS(live=live, file_name='factor_ms', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMomSeasonShort(live=live, file_name='factor_mom_season_short', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason(live=live, file_name='factor_mom_season', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason6(live=live, file_name='factor_mom_season6', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason9(live=live, file_name='factor_mom_season9', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason11(live=live, file_name='factor_mom_season11', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason16(live=live, file_name='factor_mom_season16', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeason21(live=live, file_name='factor_mom_season21', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorIntMom(live=live, file_name='factor_int_mom', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTrendFactor(live=live, file_name='factor_trend_factor', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorEarningDisparity(live=live, file_name='factor_earning_disparity', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMomOffSeason(live=live, file_name='factor_mom_off_season', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomOffSeason6(live=live, file_name='factor_mom_off_season6', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomOffSeason11(live=live, file_name='factor_mom_off_season11', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorMomSeasonShorter(live=live, file_name='factor_mom_season_shorter', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorDivSeason(live=live, file_name='factor_div_season', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorSize(live=live, file_name='factor_size', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRetSkew(live=live, file_name='factor_ret_skew', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRetMax(live=live, file_name='factor_ret_max', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorAbnormalAccrual(live=live, file_name='factor_abnormal_accrual', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorMomRev(live=live, file_name='factor_mom_rev', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------BETAS-----------------------------------------------------------------------------------------
# FactorSBSector(live=live, file_name='factor_sb_sector', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBFama(live=live, file_name='factor_sb_fama', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBPCA(live=live, file_name='factor_sb_pca', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBBond(live=live, file_name='factor_sb_bond', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBInverse(live=live, file_name='factor_sb_inverse', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBPCACopy(live=live, file_name='factor_sb_pca_copy', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBSectorCopy(live=live, file_name='factor_sb_sector_copy', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------RANK-----------------------------------------------------------------------------------------
# FactorRankRet(live=live, file_name='factor_rank_ret', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankFundRaw(live=live, file_name='factor_rank_fund_raw', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankVolume(live=live, file_name='factor_rank_volume', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankVolatility(live=live, file_name='factor_rank_volatility', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndVWR(live=live, file_name='factor_rank_ind_vwr', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndVWRFama(live=live, file_name='factor_rank_ind_vwr_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndVWRSub(live=live, file_name='factor_rank_ind_vwr_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndMom(live=live, file_name='factor_rank_ind_mom', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndMomFama(live=live, file_name='factor_rank_ind_mom_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankIndMomSub(live=live, file_name='factor_rank_ind_mom_sub', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankLoadRet(live=live, file_name='factor_rank_load_ret', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankLoadVolume(live=live, file_name='factor_rank_load_volume', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBFama(live=live, file_name='factor_rank_sb_fama', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBBond(live=live, file_name='factor_rank_sb_bond', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBPCA(live=live, file_name='factor_rank_sb_pca', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankSBInverse(live=live, file_name='factor_rank_sb_inverse', skip=True, stock=stock, start=start, end=end).create_factor()
# FactorRankRetComp(live=live, file_name='factor_rank_ret_comp', skip=True, stock=stock, start=start, end=end).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------CLUSTER---------------------------------------------------------------------------------------
# FactorClustRet(live=live, file_name='factor_clust_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustVolume(live=live, file_name='factor_clust_volume', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustVolatility(live=live, file_name='factor_clust_volatility', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustRetComp(live=live, file_name='factor_clust_ret_comp', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustLoadRet(live=live, file_name='factor_clust_load_ret', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustFundRaw(live=live, file_name='factor_clust_fund_raw', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=1, cluster=21).create_factor()
# FactorClustLoadVolume(live=live, file_name='factor_clust_load_volume', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMom(live=live, file_name='factor_clust_ind_mom', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMomSub(live=live, file_name='factor_clust_ind_mom_sub', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustIndMomFama(live=live, file_name='factor_clust_ind_mom_fama', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# FactorClustMomSeason(live=live, file_name='factor_clust_mom_season', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------FACTORS TO SKIP----------------------------------------------------------------------------------
# FactorSBInd(live=live, file_name='factor_sb_ind', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRFRet(live=live, file_name='factor_rf_ret', stock=stock, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBFundRaw(live=live, file_name='factor_sb_fund_raw', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()

elapsed_time = time.time() - start_time
print(f"Total time to create all factors: {round(elapsed_time)} seconds")
print("-" * 60)
ray.shutdown()
