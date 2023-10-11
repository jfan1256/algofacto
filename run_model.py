from prep_factor import PrepFactor
from alpha_model import AlphaModel

from functions.utils.func import *

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
stock = read_stock(get_load_data_large_dir() / 'permno_to_train_fund.csv')

start = '2010-01-01'
end = '2023-01-01'
save = False
lightgbm_params = {
    'max_depth':         {'optuna': ('suggest_categorical', [4, 6, 8]),         'gridsearch': [4, 6, 8],                     'default': 6},
    'learning_rate':     {'optuna': ('suggest_float', 0.10, 0.50, False),       'gridsearch': [0.005, 0.01, 0.1, 0.15],      'default': 0.15},
    'num_leaves':        {'optuna': ('suggest_int', 5, 150),                    'gridsearch': [20, 40, 60],                  'default': 15},
    'feature_fraction':  {'optuna': ('suggest_float', 0.5, 1.0),                'gridsearch': [0.7, 0.8, 0.9],               'default': 0.85},
    'min_gain_to_split': {'optuna': ('suggest_float', 0.02, 0.02, False),       'gridsearch': [0.0001, 0.001, 0.01],         'default': 0.02},
    'min_data_in_leaf':  {'optuna': ('suggest_int', 60, 60),                    'gridsearch': [40, 60, 80],                  'default': 60},
    'lambda_l1':         {'optuna': ('suggest_float', 0, 0, False),             'gridsearch': [0.001, 0.01],                 'default': 0},
    'lambda_l2':         {'optuna': ('suggest_float', 1e-5, 10, True),          'gridsearch': [0.001, 0.01],                 'default': 0.01},
    'bagging_fraction':  {'optuna': ('suggest_float', 1.0, 1.0, True),          'gridsearch': [0.9, 1],                      'default': 1},
    'bagging_freq':      {'optuna': ('suggest_int', 0, 0),                      'gridsearch': [0, 20],                       'default': 0},
}

catboost_params = {
    'max_depth':          {'optuna': ('suggest_categorical', [4, 6, 8]),         'gridsearch': [4, 6, 8],                     'default': 6},
    'learning_rate':      {'optuna': ('suggest_float', 0.10, 0.50, False),       'gridsearch': [0.005, 0.01, 0.1, 0.15],      'default': 0.03},
    'num_leaves':         {'optuna': ('suggest_int', 5, 150),                    'gridsearch': [20, 40, 60],                  'default': 31},
    'min_child_samples':  {'optuna': ('suggest_float', 0.5, 1.0),                'gridsearch': [0.7, 0.8, 0.9],               'default': 1},
    'l2_leaf_reg':        {'optuna': ('suggest_float', 1e-5, 10, True),          'gridsearch': [0.0001, 0.001, 0.01],         'default': 3.0}
}

start_time = time.time()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
alpha = AlphaModel(model_name='lightgbm_trial_47', tuning=['optuna', 30], plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, incr=True, opt='wfo',
                   weight=False, outlier=False, early=True, pretrain_len=504, train_len=1260, valid_len=252, test_len=21, **lightgbm_params)

# alpha = AlphaModel(model_name='catboost_trial_1', tuning='default', plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, incr=False, opt='ewo',
#                    weight=False, outlier=False, early=True, pretrain_len=0, train_len=1260, valid_len=252, test_len=21, **catboost_params)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------

ret = PrepFactor(factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ret)
del ret

ret_comp = PrepFactor(factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ret_comp)
del ret_comp

cycle = PrepFactor(factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cycle, categorical=True)
del cycle

talib = PrepFactor(factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(talib)
del talib

macro = PrepFactor(factor_name='factor_macro', group='permno', interval='D', kind='macro', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(macro)
del macro

volume = PrepFactor(factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
alpha.add_factor(volume)
del volume

volatility = PrepFactor(factor_name='factor_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(volatility)
del volatility

sign = PrepFactor(factor_name='factor_sign', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sign, categorical=True)
del sign

fund_raw = PrepFactor(factor_name='factor_fund_raw', group='permno', interval='D', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(fund_raw)
del fund_raw

# fund_q = PrepFactor(factor_name='factor_fund_q', group='permno', interval='D', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(fund_q, categorical=True)
# del fund_q
#
# high = PrepFactor(factor_name='factor_high'3, group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(high)
# del high
#
# low = PrepFactor(factor_name='factor_low', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(low)
# del low
#
# total = PrepFactor(factor_name='factor_total', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(total)
# del total

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------BETAS---------------------------------------------------------------------------------------------
sb_fama = PrepFactor(factor_name='factor_sb_fama', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_fama)
del sb_fama

sb_lag_bond = PrepFactor(factor_name='factor_sb_lag_bond', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_lag_bond)
del sb_lag_bond

sb_pca = PrepFactor(factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_pca)
del sb_pca

sb_sector = PrepFactor(factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_sector)
del sb_sector

# sb_inverse = PrepFactor(factor_name='factor_sb_inverse', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=False).prep()
# alpha.add_factor(sb_inverse)
# del sb_inverse

# sb_oil = PrepFactor(factor_name='factor_sb_oil', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_oil)
# del sb_oil

# sb_bond = PrepFactor(factor_name='factor_sb_bond', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_bond)
# del sb_bond

# sb_ind = PrepFactor(factor_name='factor_sb_ind', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_ind)
# del sb_ind

# sb_overall = PrepFactor(factor_name='factor_sb_overall', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_overall)
# del sb_overall

# sb_etf = PrepFactor(factor_name='factor_sb_etf', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_etf)
# del sb_etf

# rf_ret = PrepFactor(factor_name='factor_rf_ret', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_ret)
# del rf_ret

# rf_volume = PrepFactor(factor_name='factor_rf_volume', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_volume)
# del rf_volume

# rf_sign = PrepFactor(factor_name='factor_rf_sign', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_sign)
# del rf_sign

# sb_macro = PrepFactor(factor_name='factor_sb_macro', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_macro)
# del sb_macro

# sb_fund_ind = PrepFactor(factor_name='factor_sb_fund_ind', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_fund_ind)
# del sb_fund_ind

# sb_fund_raw = PrepFactor(factor_name='factor_sb_fund_raw', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_fund_raw)
# del sb_fund_raw

# mf_ret = PrepFactor(factor_name='factor_mf_ret', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(mf_ret)
# del mf_ret

# ep_bond = PrepFactor(factor_name='factor_ep_bond', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(ep_bond)
# del ep_bond

# ep_etf = PrepFactor(factor_name='factor_ep_etf', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(ep_etf)
# del ep_etf

# ep_fama = PrepFactor(factor_name='factor_ep_fama', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(ep_fama)
# del ep_fama

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
clust_ret = PrepFactor(factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ret, categorical=True)
del clust_ret

clust_load_ret = PrepFactor(factor_name='factor_clust_load_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_load_ret, categorical=True)
del clust_load_ret

clust_fund_raw = PrepFactor(factor_name='factor_clust_fund_raw', group='permno', interval='M', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_fund_raw, categorical=True)
del clust_fund_raw

clust_ind_mom = PrepFactor(factor_name='factor_clust_ind_mom', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ind_mom, categorical=True)
del clust_ind_mom

clust_ind_mom_fama = PrepFactor(factor_name='factor_clust_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ind_mom_fama, categorical=True)
del clust_ind_mom_fama

clust_ind_mom_sub = PrepFactor(factor_name='factor_clust_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ind_mom_sub, categorical=True)
del clust_ind_mom_sub

# clust_load_volume = PrepFactor(factor_name='factor_clust_load_volume', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_load_volume, categorical=True)
# del clust_load_volume

# clust_volatility = PrepFactor(factor_name='factor_clust_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volatility, categorical=True)
# del clust_volatility

# clust_volume = PrepFactor(factor_name='factor_clust_volume', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volume, categorical=True)
# del clust_volume

# clust_ret30 = PrepFactor(factor_name='factor_clust_ret30', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_ret30, categorical=True)
# del clust_ret30

# clust_ret60 = PrepFactor(factor_name='factor_clust_ret60', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_ret60, categorical=True)
# del clust_ret60

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
load_ret = PrepFactor(factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_ret)
del load_ret

load_volume = PrepFactor(factor_name='factor_load_volume', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_volume)
del load_volume

# load_volatility = PrepFactor(factor_name='factor_load_volatility', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(load_volatility)
# del load_volatility

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------CONDITION-----------------------------------------------------------------------------------------
cond_ret = PrepFactor(factor_name='factor_cond_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cond_ret, categorical=True)
del cond_ret

cond_ind_mom = PrepFactor(factor_name='factor_cond_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cond_ind_mom, categorical=True)
del cond_ind_mom

cond_ind_mom_fama = PrepFactor(factor_name='factor_cond_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cond_ind_mom_fama, categorical=True)
del cond_ind_mom_fama

cond_ind_mom_sub = PrepFactor(factor_name='factor_cond_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cond_ind_mom_sub, categorical=True)
del cond_ind_mom_sub

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
ind = PrepFactor(factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind, categorical=True)
del ind

ind_mom = PrepFactor(factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_mom)
del ind_mom

ind_fama = PrepFactor(factor_name='factor_ind_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_fama, categorical=True)
del ind_fama

ind_sub = PrepFactor(factor_name='factor_ind_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_sub, categorical=True)
del ind_sub

ind_mom_fama = PrepFactor(factor_name='factor_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_mom_fama)
del ind_mom_fama

ind_mom_sub = PrepFactor(factor_name='factor_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_mom_sub)
del ind_mom_sub

ind_vwr = PrepFactor(factor_name='factor_ind_vwr', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_vwr)
del ind_vwr

ind_vwr_fama = PrepFactor(factor_name='factor_ind_vwr_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_vwr_fama)
del ind_vwr_fama

ind_vwr_sub = PrepFactor(factor_name='factor_ind_vwr_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_vwr_sub)
del ind_vwr_sub

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------RANK----------------------------------------------------------------------------------------------
rank_load_ret = PrepFactor(factor_name='factor_rank_load_ret', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_load_ret, categorical=True, date=True)
del rank_load_ret

rank_ret = PrepFactor(factor_name='factor_rank_ret', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ret, categorical=True, date=True)
del rank_ret

rank_fund_raw = PrepFactor(factor_name='factor_rank_fund_raw', group='permno', interval='M', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_fund_raw, categorical=True, date=True)
del rank_fund_raw

rank_volume = PrepFactor(factor_name='factor_rank_volume', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_volume, categorical=True, date=True)
del rank_volume

rank_volatility = PrepFactor(factor_name='factor_rank_volatility', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_volatility, categorical=True, date=True)
del rank_volatility

rank_ind_vwr = PrepFactor(factor_name='factor_rank_ind_vwr', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ind_vwr, categorical=True)
del rank_ind_vwr

rank_ind_vwr_fama = PrepFactor(factor_name='factor_rank_ind_vwr_fama', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ind_vwr_fama, categorical=True)
del rank_ind_vwr_fama

rank_ind_vwr_sub = PrepFactor(factor_name='factor_rank_ind_vwr_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ind_vwr_sub, categorical=True)
del rank_ind_vwr_sub

rank_ind_mom = PrepFactor(factor_name='factor_rank_ind_mom', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ind_mom, categorical=True, date=True)
del rank_ind_mom

rank_ind_mom_fama = PrepFactor(factor_name='factor_rank_ind_mom_fama', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ind_mom_fama, categorical=True)
del rank_ind_mom_fama

rank_ind_mom_sub = PrepFactor(factor_name='factor_rank_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ind_mom_sub, categorical=True, date=True)
del rank_ind_mom_sub

rank_ret_comp = PrepFactor(factor_name='factor_rank_ret_comp', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ret_comp, categorical=True, date=True)
del rank_ret_comp

# rank_load_volume = PrepFactor(factor_name='factor_rank_load_volume', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_load_volume, categorical=True, date=True)
# del rank_load_volume

# rank_sb_fama = PrepFactor(factor_name='factor_rank_sb_fama', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_fama, categorical=True, date=True)
# del rank_sb_fama

# rank_sb_lag_bond = PrepFactor(factor_name='factor_rank_sb_lag_bond', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_lag_bond, categorical=True, date=True)
# del rank_sb_lag_bond
#
# rank_sb_pca = PrepFactor(factor_name='factor_rank_sb_pca', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_pca, categorical=True, date=True)
# del rank_sb_pca

# rank_sb_inverse = PrepFactor(factor_name='factor_rank_sb_inverse', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_inverse, categorical=True, date=True)
# del rank_sb_inverse

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------OPEN ASSET----------------------------------------------------------------------------------------

age_mom = PrepFactor(factor_name='factor_age_mom', group='permno', interval='D', kind='age', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(age_mom)
del age_mom

net_debt_finance = PrepFactor(factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(net_debt_finance)
del net_debt_finance

chtax = PrepFactor(factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(chtax)
del chtax

asset_growth = PrepFactor(factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(asset_growth)
del asset_growth

mom_season_short = PrepFactor(factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season_short)
del mom_season_short

mom_season = PrepFactor(factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season)
del mom_season

noa = PrepFactor(factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(noa)
del noa

invest_ppe = PrepFactor(factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(invest_ppe)
del invest_ppe

inv_growth = PrepFactor(factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(inv_growth)
del inv_growth

dividend = PrepFactor(factor_name='factor_dividend', group='permno', interval='D', kind='dividend', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(dividend, categorical=True)
del dividend

earning_streak = PrepFactor(factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(earning_streak)
del earning_streak

trend_factor = PrepFactor(factor_name='factor_trend_factor', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(trend_factor)
del trend_factor

mom_season6 = PrepFactor(factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season6)
del mom_season6

mom_season11 = PrepFactor(factor_name='factor_mom_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season11)
del mom_season11

comp_debt = PrepFactor(factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(comp_debt)
del comp_debt

mom_vol = PrepFactor(factor_name='factor_mom_vol', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_vol, categorical=True)
del mom_vol

ms = PrepFactor(factor_name='factor_ms', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ms, categorical=True)
del ms

int_mom = PrepFactor(factor_name='factor_int_mom', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(int_mom)
del int_mom

cheq = PrepFactor(factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cheq)
del cheq

xfin = PrepFactor(factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(xfin)
del xfin

emmult = PrepFactor(factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(emmult)
del emmult

accrual = PrepFactor(factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(accrual)
del accrual

frontier = PrepFactor(factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(frontier)
del frontier

# abnormal_accrual = PrepFactor(factor_name='factor_abnormal_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(abnormal_accrual)
# del abnormal_accrual

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------TRAINING------------------------------------------------------------------------------------------
elapsed_time = time.time() - start_time
print("-" * 60)
print(f"AlphaModel Data Shape: {alpha.data.shape}")
print(f"Prep and Add took: {round(elapsed_time)} seconds")
print("-" * 60)
print("Run Model")

if 'lightgbm' in alpha.model_name:
    alpha.lightgbm()
elif 'catboost' in alpha.model_name:
    alpha.catboost()



