from functions.utils.func import *

from factor_class.factor_macro import FactorMacro
from factor_class.factor_fund_ratio import FactorFundRatio
from factor_class.factor_ind import FactorInd
from factor_class.factor_ind_mom import FactorIndMom
from factor_class.factor_ret import FactorRet
from factor_class.factor_rank import FactorRank
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

start = '2006-01-01'
end = '2023-01-01'
tickers = read_ticker(get_load_data_large_dir() / 'tickers_to_train_fundamental.csv')

ray.init(num_cpus=16, ignore_reinit_error=True)
start_time = time.time()

# FactorMacro(file_name='factor_macro', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker', general=True).create_factor()
# FactorFundRatio(file_name='factor_fund_ratio', skip=True, ticker=tickers, start=start, end=end).create_factor()
# FactorInd(file_name='factor_ind', skip=True, ticker=tickers, start=start, end=end).create_factor()
# FactorIndMom(file_name='factor_ind_mom', skip=True, ticker=tickers, start=start, end=end).create_factor()
# FactorRet(file_name='factor_ret', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorRank(file_name='factor_rank', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorSign(file_name='factor_sign', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorVolatility(file_name='factor_volatility', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorVolume(file_name='factor_volume', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorTime(file_name='factor_time', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorStreversal(file_name='factor_streversal', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorRFRet(file_name='factor_rf_ret', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorRFSign(file_name='factor_rf_sign', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorRFVolume(file_name='factor_rf_volume', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorSBETF(file_name='factor_sb_etf', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorSBFama(file_name='factor_sb_fama', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorSBSPYInv(file_name='factor_sb_spy_inv', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorTalib(file_name='factor_talib', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorOpenAsset(file_name='factor_open_asset', skip=True, ticker=['all'], start=start, end=end).create_factor()
# FactorLoadRet(file_name='factor_load_ret', start=start, end=end, batch_size=8, splice_size=22, group='date', window=60, component=5).create_factor()
# FactorClustRet(file_name='factor_clust_ret', start=start, end=end, batch_size=8, splice_size=22, group='date', window=60, cluster=15).create_factor()
# FactorClustLoadRet(file_name='factor_clust_load_ret', start=start, end=end, batch_size=8, splice_size=22, group='date', window=60, cluster=15).create_factor()
# FactorSBBond(file_name='factor_sb_bond', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorSBMacro(file_name='factor_sb_macro', ticker=tickers, start=start, end=end, batch_size=8, splice_size=22, group='ticker').create_factor()
# FactorClustVolume(file_name='factor_clust_volume', start=start, end=end, batch_size=8, splice_size=22, group='date', window=60, cluster=15).create_factor()
FactorClustIndMom(file_name='factor_clust_ind_mom', start=start, end=end, batch_size=8, splice_size=22, group='date', window=60, cluster=15).create_factor()

elapsed_time = time.time() - start_time
print(f"Total time to create all factors: {round(elapsed_time)} seconds")
print("-" * 60)
ray.shutdown()
