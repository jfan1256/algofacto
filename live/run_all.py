from prep_factor import PrepFactor
from alpha_model import AlphaModel
from live_data import LiveData

from functions.utils.func import *

from factor_class.factor_ind import FactorInd
from factor_class.factor_ind_mom import FactorIndMom
from factor_class.factor_ret import FactorRet
from factor_class.factor_sign_ret import FactorSignRet
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
from factor_class.factor_cond_ret import FactorCondRet
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
from factor_class.factor_earning_streak import FactorEarningStreak
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

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------GET LIVE DATA----------------------------------------------------------------------------------
print("---------------------------------------------------------------------------GET LIVE DATA----------------------------------------------------------------------------------")
live = True
start = '2004-01-01'
update_price = True
current_date = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
total_time = time.time()

start_time = time.time()
live_data = LiveData(live=live, start_date=start, current_date=current_date)

if update_price:
    live_data.create_crsp_price()
live_data.create_compustat_quarterly()
live_data.create_compustat_annual()
live_data.create_stock_list()
live_data.create_live_price()
live_data.create_compustat_pension()
live_data.create_misc()
live_data.create_industry()
live_data.create_ibes()
live_data.create_macro()
live_data.create_risk_rate()

elapsed_time = time.time() - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"Total time to get live data: {int(minutes)}:{int(seconds):02}")
print("-" * 60)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------CREATE FACTORS---------------------------------------------------------------------------------------
print("---------------------------------------------------------------------CREATE FACTORS---------------------------------------------------------------------------------------")
stock = read_stock(get_large_dir(live) / 'permno_live.csv')

ray.init(num_cpus=16, ignore_reinit_error=True)
start_time = time.time()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------GENERAL----------------------------------------------------------------------------------------
FactorRet(live=live, file_name='factor_ret', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorRetComp(live=live, file_name='factor_ret_comp', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorSignRet(live=live, file_name='factor_sign_ret', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorVolatility(live=live, file_name='factor_volatility', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorVolume(live=live, file_name='factor_volume', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorTime(live=live, file_name='factor_time', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorTalib(live=live, file_name='factor_talib', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------
FactorLoadRet(live=live, file_name='factor_load_ret', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
FactorLoadVolume(live=live, file_name='factor_load_volume', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, component=5).create_factor()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------CONDITION-------------------------------------------------------------------------------------
FactorCondRet(live=live, file_name='factor_cond_ret', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------INDUSTRY--------------------------------------------------------------------------------------
FactorInd(live=live, file_name='factor_ind', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorIndFama(live=live, file_name='factor_ind_fama', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorIndSub(live=live, file_name='factor_ind_sub', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorIndMom(live=live, file_name='factor_ind_mom', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorIndMomFama(live=live, file_name='factor_ind_mom_fama', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorIndMomSub(live=live, file_name='factor_ind_mom_sub', skip=True, stock=stock, start=start, end=current_date).create_factor()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------OPEN ASSET------------------------------------------------------------------------------------
FactorAccrual(live=live, file_name='factor_accrual', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorNetDebtFinance(live=live, file_name='factor_net_debt_finance', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorCHTax(live=live, file_name='factor_chtax', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorHire(live=live, file_name='factor_hire', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorAssetGrowth(live=live, file_name='factor_asset_growth', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorNOA(live=live, file_name='factor_noa', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorInvestPPEInv(live=live, file_name='factor_invest_ppe_inv', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorInvGrowth(live=live, file_name='factor_inv_growth', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorCompDebt(live=live, file_name='factor_comp_debt', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorChEQ(live=live, file_name='factor_cheq', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorXFIN(live=live, file_name='factor_xfin', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorEmmult(live=live, file_name='factor_emmult', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorAccrualBM(live=live, file_name='factor_accrual_bm', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorPctTotAcc(live=live, file_name='factor_pcttotacc', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorRDS(live=live, file_name='factor_rds', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorFrontier(live=live, file_name='factor_frontier', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorEarningStreak(live=live, file_name='factor_earning_streak', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorAgeMom(live=live, file_name='factor_age_mom', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorMomVol(live=live, file_name='factor_mom_vol', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorMomSeasonShort(live=live, file_name='factor_mom_season_short', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorMomSeason(live=live, file_name='factor_mom_season', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorMomSeason6(live=live, file_name='factor_mom_season6', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorMomSeason9(live=live, file_name='factor_mom_season9', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorMomSeason11(live=live, file_name='factor_mom_season11', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorMomSeason16(live=live, file_name='factor_mom_season16', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorIntMom(live=live, file_name='factor_int_mom', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorTrendFactor(live=live, file_name='factor_trend_factor', skip=True, stock=stock, start=start, end=current_date).create_factor()
FactorMomOffSeason(live=live, file_name='factor_mom_off_season', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorMomRev(live=live, file_name='factor_mom_rev', skip=True, stock=stock, start=start, end=current_date).create_factor()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------BETAS-----------------------------------------------------------------------------------------
FactorSBSector(live=live, file_name='factor_sb_sector', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
FactorSBPCA(live=live, file_name='factor_sb_pca', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='permno').create_factor()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------CLUSTER---------------------------------------------------------------------------------------
FactorClustRet(live=live, file_name='factor_clust_ret', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
FactorClustLoadRet(live=live, file_name='factor_clust_load_ret', stock='all', start=start, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
FactorClustIndMom(live=live, file_name='factor_clust_ind_mom', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
FactorClustIndMomSub(live=live, file_name='factor_clust_ind_mom_sub', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()
FactorClustIndMomFama(live=live, file_name='factor_clust_ind_mom_fama', stock=stock, start=start, end=current_date, batch_size=10, splice_size=20, group='date', join='permno', window=21, cluster=21).create_factor()

elapsed_time = time.time() - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"Total time to create all factors: {int(minutes)}:{int(seconds):02}")
print("-" * 60)
ray.shutdown()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
print("---------------------------------------------------------------------TRAIN MODEL------------------------------------------------------------------------------------------")
stock = read_stock(get_large_dir(live) / 'permno_live.csv')

start = '2013-01-01'
save = True
start_time = time.time()

lightgbm_params = {
    'max_depth':          {'optuna': ('suggest_categorical', [6]),              'gridsearch': [4, 6, 8],                     'default': 6},
    'learning_rate':      {'optuna': ('suggest_float', 0.10, 0.50, False),      'gridsearch': [0.005, 0.01, 0.1, 0.15],      'default': 0.15},
    'num_leaves':         {'optuna': ('suggest_int', 5, 150),                   'gridsearch': [20, 40, 60],                  'default': 15},
    'feature_fraction':   {'optuna': ('suggest_categorical',[1.0]),             'gridsearch': [0.7, 0.8, 0.9],               'default': 1.0},
    'min_gain_to_split':  {'optuna': ('suggest_float', 0.02, 0.02, False),      'gridsearch': [0.0001, 0.001, 0.01],         'default': 0.02},
    'min_data_in_leaf':   {'optuna': ('suggest_int', 50, 200),                  'gridsearch': [40, 60, 80],                  'default': 60},
    'lambda_l1':          {'optuna': ('suggest_float', 0, 0, False),            'gridsearch': [0.001, 0.01],                 'default': 0},
    'lambda_l2':          {'optuna': ('suggest_float', 1e-5, 10, True),         'gridsearch': [0.001, 0.01],                 'default': 0.01},
    'bagging_fraction':   {'optuna': ('suggest_float', 1.0, 1.0, True),         'gridsearch': [0.9, 1],                      'default': 1},
    'bagging_freq':       {'optuna': ('suggest_int', 0, 0),                     'gridsearch': [0, 20],                       'default': 0},
}

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
format_end = date.today().strftime('%Y%m%d')
model_name = f'lightgbm_{format_end}'
alpha = AlphaModel(live=live, model_name=model_name, end=current_date, tuning='default', plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, incr=True, opt='wfo',
                   weight=False, outlier=False, early=True, pretrain_len=1260, train_len=504, valid_len=126, test_len=21, **lightgbm_params)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
ret = PrepFactor(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(ret)
del ret

ret_comp = PrepFactor(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(ret_comp)
del ret_comp

cycle = PrepFactor(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(cycle, categorical=True)
del cycle

talib = PrepFactor(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(talib)
del talib

volume = PrepFactor(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=current_date, save=save).prep()
alpha.add_factor(volume)
del volume

volatility = PrepFactor(live=live, factor_name='factor_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(volatility)
del volatility

sign_ret = PrepFactor(live=live, factor_name='factor_sign_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(sign_ret, categorical=True)
del sign_ret

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
load_ret = PrepFactor(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(load_ret)
del load_ret

load_volume = PrepFactor(live=live, factor_name='factor_load_volume', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(load_volume)
del load_volume

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------CONDITION-----------------------------------------------------------------------------------------
cond_ret = PrepFactor(live=live, factor_name='factor_cond_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(cond_ret, categorical=True)
del cond_ret

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
ind = PrepFactor(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(ind, categorical=True)
del ind

# ind_fama = PrepFactor(live=live, factor_name='factor_ind_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
# alpha.add_factor(ind_fama, categorical=True)
# del ind_fama

ind_sub = PrepFactor(live=live, factor_name='factor_ind_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(ind_sub, categorical=True)
del ind_sub

ind_mom = PrepFactor(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(ind_mom)
del ind_mom

# ind_mom_fama = PrepFactor(live=live, factor_name='factor_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
# alpha.add_factor(ind_mom_fama)
# del ind_mom_fama

ind_mom_sub = PrepFactor(live=live, factor_name='factor_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(ind_mom_sub)
del ind_mom_sub

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------OPEN ASSET----------------------------------------------------------------------------------------
age_mom = PrepFactor(live=live, factor_name='factor_age_mom', group='permno', interval='D', kind='age', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(age_mom)
del age_mom

net_debt_finance = PrepFactor(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(net_debt_finance)
del net_debt_finance

chtax = PrepFactor(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(chtax)
del chtax

asset_growth = PrepFactor(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(asset_growth)
del asset_growth

mom_season_short = PrepFactor(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_season_short)
del mom_season_short

mom_season = PrepFactor(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_season)
del mom_season

noa = PrepFactor(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(noa)
del noa

invest_ppe = PrepFactor(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(invest_ppe)
del invest_ppe

inv_growth = PrepFactor(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(inv_growth)
del inv_growth

trend_factor = PrepFactor(live=live, factor_name='factor_trend_factor', group='permno', interval='D', kind='trend', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(trend_factor)
del trend_factor

mom_season6 = PrepFactor(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_season6)
del mom_season6

mom_season11 = PrepFactor(live=live, factor_name='factor_mom_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_season11)
del mom_season11

mom_season16 = PrepFactor(live=live, factor_name='factor_mom_season16', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_season16)
del mom_season16

comp_debt = PrepFactor(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(comp_debt)
del comp_debt

mom_vol = PrepFactor(live=live, factor_name='factor_mom_vol', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_vol, categorical=True)
del mom_vol

int_mom = PrepFactor(live=live, factor_name='factor_int_mom', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(int_mom)
del int_mom

cheq = PrepFactor(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(cheq)
del cheq

xfin = PrepFactor(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(xfin)
del xfin

emmult = PrepFactor(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(emmult)
del emmult

accrual = PrepFactor(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(accrual)
del accrual

frontier = PrepFactor(live=live, factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(frontier)
del frontier

mom_rev = PrepFactor(live=live, factor_name='factor_mom_rev', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_rev, categorical=True)
del mom_rev

hire = PrepFactor(live=live, factor_name='factor_hire', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(hire)
del hire

rds = PrepFactor(live=live, factor_name='factor_rds', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(rds)
del rds

pcttoacc = PrepFactor(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(pcttoacc)
del pcttoacc

accrual_bm = PrepFactor(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(accrual_bm)
del accrual_bm

mom_off_season = PrepFactor(live=live, factor_name='factor_mom_off_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(mom_off_season)
del mom_off_season

earning_streak = PrepFactor(live=live, factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(earning_streak)
del earning_streak

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------BETAS---------------------------------------------------------------------------------------------
sb_pca = PrepFactor(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(sb_pca)
del sb_pca

sb_sector = PrepFactor(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(sb_sector)
del sb_sector

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
clust_ret = PrepFactor(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(clust_ret, categorical=True)
del clust_ret

clust_load_ret = PrepFactor(live=live, factor_name='factor_clust_load_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(clust_load_ret, categorical=True)
del clust_load_ret

clust_ind_mom = PrepFactor(live=live, factor_name='factor_clust_ind_mom', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(clust_ind_mom, categorical=True)
del clust_ind_mom

clust_ind_mom_fama = PrepFactor(live=live, factor_name='factor_clust_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(clust_ind_mom_fama, categorical=True)
del clust_ind_mom_fama

clust_ind_mom_sub = PrepFactor(live=live, factor_name='factor_clust_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=current_date, save=save).prep()
alpha.add_factor(clust_ind_mom_sub, categorical=True)
del clust_ind_mom_sub

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------TRAINING------------------------------------------------------------------------------------------
elapsed_time = time.time() - start_time
print("-" * 60)
print(f"Total time to prep and add all factors: {round(elapsed_time)} seconds")
print(f"AlphaModel Dataframe Shape: {alpha.data.shape}")
print("-" * 60)
print("Run Model")

alpha.lightgbm()

elapsed_time = time.time() - total_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"Total time to execute everything: {int(minutes)}:{int(seconds):02}")
print("-" * 60)
