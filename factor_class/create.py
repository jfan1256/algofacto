from functions.utils.func import *

from factor_class.factor_macro import FactorMacro
from factor_class.factor_ind import FactorInd
from factor_class.factor_ind_mom import FactorIndMom
from factor_class.factor_ret import FactorRet
from factor_class.factor_rank_ret import FactorRankRet
from factor_class.factor_sign import FactorSign
from factor_class.factor_volatility import FactorVolatility
from factor_class.factor_volume import FactorVolume
from factor_class.factor_time import FactorTime
from factor_class.factor_streversal import FactorStreversal
from factor_class.factor_rf_ret import FactorRFRet
from factor_class.factor_rf_sign import FactorRFSign
from factor_class.factor_rf_volume import FactorRFVolume
from factor_class.factor_sb_etf import FactorSBETF
from factor_class.factor_sb_fama import FactorSBFama
from factor_class.factor_sb_spy_inv import FactorSBSPYInv
from factor_class.factor_talib import FactorTalib
from factor_class.factor_open_asset import FactorOpenAsset
from factor_class.factor_load_ret import FactorLoadRet
from factor_class.factor_clust_ret import FactorClustRet
from factor_class.factor_clust_load_ret import FactorClustLoadRet
from factor_class.factor_sb_bond import FactorSBBond
from factor_class.factor_sb_macro import FactorSBMacro
from factor_class.factor_clust_volume import FactorClustVolume
from factor_class.factor_clust_ind_mom import FactorClustIndMom
from factor_class.factor_load_volatility import FactorLoadVolatility
from factor_class.factor_load_volume import FactorLoadVolume
from factor_class.factor_clust_volatility import FactorClustVolatility
from factor_class.factor_dividend import FactorDividend
from factor_class.factor_clust_ret30 import FactorClustRet30
from factor_class.factor_clust_ret60 import FactorClustRet60
from factor_class.factor_sb_oil import FactorSBOil
from factor_class.factor_sb_sector import FactorSBSector
from factor_class.factor_sb_ind import FactorSBInd
from factor_class.factor_sb_overall import FactorSBOverall
from factor_class.factor_sb_macro_test import FactorSBMacroTest
from factor_class.factor_sb_bond_test import FactorSBBondTest
from factor_class.factor_sb_fama_test import FactorSBFamaTest
from factor_class.factor_sb_etf_test import FactorSBETFTest
from factor_class.factor_rf_ret_test import FactorRFRetTest
from factor_class.factor_ret_condition import FactorRetCondition
from factor_class.factor_sb_fund_ind import FactorSBFundInd
from factor_class.factor_sb_fund_raw import FactorSBFundRaw
from factor_class.factor_mf_ret import FactorMFRet
from factor_class.factor_fund_raw import FactorFundRaw
from factor_class.factor_fund_q import FactorFundQ
from factor_class.factor_rank_fund_ratio import FactorRankFundRaw
from factor_class.factor_clust_fund_rank import FactorClustFundRaw
from factor_class.factor_fund_ratio import FactorFundRatio
from factor_class.factor_high import FactorHigh
from factor_class.factor_low import FactorLow
from factor_class.factor_rank_volume import FactorRankVolume
from factor_class.factor_clust_ret_test import FactorClustRetTest
from factor_class.factor_clust_load_volume import FactorClustLoadVolume
from factor_class.factor_total import FactorTotal
from factor_class.factor_rank_volatility import FactorRankVolatility
from factor_class.factor_ep_bond import FactorEPBond
from factor_class.factor_ep_etf import FactorEPETF
from factor_class.factor_ep_fama import FactorEPFama


start = '2006-01-01'
end = '2022-01-01'
permno = read_stock(get_load_data_large_dir() / 'permno_to_train.csv')
# ticker = read_stock(get_load_data_large_dir() / 'tickers_to_train_fundamental.csv')

ray.init(num_cpus=16, ignore_reinit_error=True)
start_time = time.time()

# FactorInd(file_name='factor_ind', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRet(file_name='factor_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRankRet(file_name='factor_rank_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSign(file_name='factor_sign', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolatility(file_name='factor_volatility', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorVolume(file_name='factor_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTime(file_name='factor_time', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorStreversal(file_name='factor_streversal', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRFRet(file_name='factor_rf_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBETF(file_name='factor_sb_etf', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorSBFama(file_name='factor_sb_fama', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorLoadRet(file_name='factor_load_ret', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, component=5).create_factor()
# FactorClustRet(file_name='factor_clust_ret', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# FactorClustLoadRet(file_name='factor_clust_load_ret', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# FactorSBBond(file_name='factor_sb_bond', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorSBMacro(file_name='factor_sb_macro', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorLoadVolume(file_name='factor_load_volume', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, component=5).create_factor()
# FactorMacro(file_name='factor_macro', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno', general=True).create_factor()
# # FactorRFSign(file_name='factor_rf_sign', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorRFVolume(file_name='factor_rf_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorIndMom(file_name='factor_ind_mom', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorSBSPYInv(file_name='factor_sb_spy_inv', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorTalib(file_name='factor_talib', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorOpenAsset(file_name='factor_open_asset', skip=True, stock='all', start=start, end=end).create_factor()
# FactorClustVolume(file_name='factor_clust_volume', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# FactorClustIndMom(file_name='factor_clust_ind_mom', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# FactorLoadVolatility(file_name='factor_load_volatility', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, component=5).create_factor()
# FactorClustVolatility(file_name='factor_clust_volatility', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# # FactorDividend(file_name='factor_dividend', skip=False, stock='all', start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorClustRet30(file_name='factor_clust_ret30', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# # FactorClustRet60(file_name='factor_clust_ret60', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# # FactorSBOil(file_name='factor_sb_oil', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorSBSector(file_name='factor_sb_sector', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorSBOverall(file_name='factor_sb_overall', ticker=tickers, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorSBInd(file_name='factor_sb_ind', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRetCondition(file_name='factor_ret_condition', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorSBFundInd(file_name='factor_sb_fund_ind', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorSBFundRaw(file_name='factor_sb_fund_raw', stock='all', start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# # FactorMFRet(file_name='factor_mf_ret', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorFundRaw(file_name='factor_fund_raw', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorFundQ(file_name='factor_fund_q', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorRankFundRaw(file_name='factor_rank_fund_raw', skip=True, stock=permno, start=start, end=end).create_factor()
# FactorClustFundRaw(file_name='factor_clust_fund_raw', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=1, cluster=15).create_factor()
# # FactorFundRatio(file_name='factor_fund_ratio', skip=True, stock='all', start=start, end=end).create_factor()
# FactorHigh(file_name='factor_high', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorLow(file_name='factor_low', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorRankVolume(file_name='factor_rank_volume', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
# FactorClustLoadVolume(file_name='factor_clust_load_volume', start=start, end=end, batch_size=10, splice_size=20, group='date', join='permno', window=60, cluster=15).create_factor()
# FactorTotal(file_name='factor_total', skip=True, stock=permno, start=start, end=end, group='permno').create_factor()
# FactorRankVolatility(file_name='factor_rank_volatility', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorEPBond(file_name='factor_ep_bond', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorEPETF(file_name='factor_ep_etf', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()
FactorEPFama(file_name='factor_ep_fama', stock=permno, start=start, end=end, batch_size=10, splice_size=20, group='permno').create_factor()


elapsed_time = time.time() - start_time
print(f"Total time to create all factors: {round(elapsed_time)} seconds")
print("-" * 60)
ray.shutdown()
