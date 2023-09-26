from prep_factor import PrepFactor
from alpha_model import AlphaModel

from functions.utils.func import *

stock = read_stock(get_load_data_large_dir() / 'permno_to_train_fund.csv')

start = '2010-01-01'
end = '2022-01-01'
save = False
params = {
    'max_depth':         {'optuna': ('suggest_int', 6, 6),                      'gridsearch': [-1, 6, 10],                   'default': 6},
    'learning_rate':     {'optuna': ('suggest_float', 0.10, 0.50, False),       'gridsearch': [0.005, 0.01, 0.1, 0.15],      'default': 0.15},
    'num_leaves':        {'optuna': ('suggest_int', 5, 150),                    'gridsearch': [20, 40, 60],                  'default': 15},
    'feature_fraction':  {'optuna': ('suggest_float', 0.5, 1.0),                'gridsearch': [0.7, 0.8, 0.9],               'default': 0.85},
    'min_gain_to_split': {'optuna': ('suggest_float', 0.001, 1, False),         'gridsearch': [0.0001, 0.001, 0.01],         'default': 0.02},
    'min_data_in_leaf':  {'optuna': ('suggest_int', 60, 60),                    'gridsearch': [40, 60, 80],                  'default': 60},
    'lambda_l1':         {'optuna': ('suggest_float', 0, 0, False),             'gridsearch': [0.001, 0.01],                 'default': 0},
    'lambda_l2':         {'optuna': ('suggest_float', 1e-5, 10, True),          'gridsearch': [0.001, 0.01],                 'default': 0.01},
    'bagging_fraction':  {'optuna': ('suggest_float', 1.0, 1.0, True),          'gridsearch': [0.9, 1],                      'default': 1},
    'bagging_freq':      {'optuna': ('suggest_int', 0, 0),                      'gridsearch': [0, 20],                       'default': 0},
}

start_time = time.time()
alpha = AlphaModel(model_name='lightgbm_trial_35', tuning=['optuna', 30], plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, incr=True, opt='wfo',
                   weight=False, outlier=False, early=True, pretrain_len=1260, train_len=504, valid_len=21, test_len=21, **params)

ret = PrepFactor(factor_name='factor_ret', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ret)
del ret

cycle = PrepFactor(factor_name='factor_time', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cycle, categorical=True)
del cycle

talib = PrepFactor(factor_name='factor_talib', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(talib)
del talib

ind = PrepFactor(factor_name='factor_ind', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind, categorical=True)
del ind

macro = PrepFactor(factor_name='factor_macro', interval='D', kind='macro', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(macro)
del macro

volume = PrepFactor(factor_name='factor_volume', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
alpha.add_factor(volume)
del volume

volatility = PrepFactor(factor_name='factor_volatility', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(volatility)
del volatility

ind_mom = PrepFactor(factor_name='factor_ind_mom', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_mom)
del ind_mom

sb_fama = PrepFactor(factor_name='factor_sb_fama', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_fama)
del sb_fama

# sb_etf = PrepFactor(factor_name='factor_sb_etf', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_etf)
# del sb_etf

# rf_ret = PrepFactor(factor_name='factor_rf_ret', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_ret)
# del rf_ret

load_ret = PrepFactor(factor_name='factor_load_ret', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_ret)
del load_ret

clust_ret = PrepFactor(factor_name='factor_clust_ret', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ret, categorical=True)
del clust_ret

clust_load_ret = PrepFactor(factor_name='factor_clust_load_ret', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_load_ret, categorical=True)
del clust_load_ret

# open_asset = PrepFactor(factor_name='factor_open_asset', interval='M', kind='open', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(open_asset)
# del open_asset

rank_ret = PrepFactor(factor_name='factor_rank_ret', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ret, categorical=True)
del rank_ret

sign = PrepFactor(factor_name='factor_sign', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sign, categorical=True)
del sign

# rf_volume = PrepFactor(factor_name='factor_rf_volume', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_volume)
# del rf_volume

# rf_sign = PrepFactor(factor_name='factor_rf_sign', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_sign)
# del rf_sign

# sb_spy_inv = PrepFactor(factor_name='factor_sb_spy_inv', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_spy_inv)
# del sb_spy_inv

# sb_macro = PrepFactor(factor_name='factor_sb_macro', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_macro)
# del sb_macro

# clust_volume = PrepFactor(factor_name='factor_clust_volume', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volume, categorical=True)
# del clust_volume

# clust_ind_mom = PrepFactor(factor_name='factor_clust_ind_mom', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=True).prep()
# alpha.add_factor(clust_ind_mom, categorical=True)
# del clust_ind_mom

# load_volatility = PrepFactor(factor_name='factor_load_volatility', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(load_volatility)
# del load_volatility

load_volume = PrepFactor(factor_name='factor_load_volume', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_volume)
del load_volume

# clust_volatility = PrepFactor(factor_name='factor_clust_volatility', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volatility, categorical=True)
# del clust_volatility

# dividend = PrepFactor(factor_name='factor_dividend', interval='D', kind='dividend', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(dividend, categorical=True)
# del dividend

# clust_ret30 = PrepFactor(factor_name='factor_clust_ret30', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_ret30, categorical=True)
# del clust_ret30
#
# clust_ret60 = PrepFactor(factor_name='factor_clust_ret60', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_ret60, categorical=True)
# del clust_ret60

# sb_oil = PrepFactor(factor_name='factor_sb_oil', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_oil)
# del sb_oil

# sb_bond = PrepFactor(factor_name='factor_sb_bond', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_bond)
# del sb_bond

sb_sector = PrepFactor(factor_name='factor_sb_sector', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_sector)
del sb_sector

sb_ind = PrepFactor(factor_name='factor_sb_ind', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_ind)
del sb_ind

sb_overall = PrepFactor(factor_name='factor_sb_overall', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_overall)
del sb_overall

# cond_ret = PrepFactor(factor_name='factor_cond_ret', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(cond_ret, categorical=True)
# del cond_ret

# sb_fund_ind = PrepFactor(factor_name='factor_sb_fund_ind', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_fund_ind)
# del sb_fund_ind

# sb_fund_raw = PrepFactor(factor_name='factor_sb_fund_raw', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_fund_raw)
# del sb_fund_raw

# mf_ret = PrepFactor(factor_name='factor_mf_ret', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(mf_ret)
# del mf_ret

fund_raw = PrepFactor(factor_name='factor_fund_raw', interval='D', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(fund_raw)
del fund_raw

# fund_q = PrepFactor(factor_name='factor_fund_q', interval='D', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(fund_q, categorical=True)
# del fund_q

rank_fund_raw = PrepFactor(factor_name='factor_rank_fund_raw', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_fund_raw, categorical=True)
del rank_fund_raw

clust_fund_raw = PrepFactor(factor_name='factor_clust_fund_raw', interval='M', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_fund_raw, categorical=True)
del clust_fund_raw

# fund_ratio = PrepFactor(factor_name='factor_fund_ratio', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(fund_ratio)
# del fund_ratio

# high = PrepFactor(factor_name='factor_high', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=True).prep()
# alpha.add_factor(high)
# del high

# low = PrepFactor(factor_name='factor_low', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=True).prep()
# alpha.add_factor(low)
# del low

rank_volume = PrepFactor(factor_name='factor_rank_volume', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_volume, categorical=True)
del rank_volume

# clust_load_volume = PrepFactor(factor_name='factor_clust_load_volume', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_load_volume, categorical=True)
# del clust_load_volume

# total = PrepFactor(factor_name='factor_total', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(total)
# del total

rank_volatility = PrepFactor(factor_name='factor_rank_volatility', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_volatility)
del rank_volatility

# ep_bond = PrepFactor(factor_name='factor_ep_bond', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(ep_bond)
# del ep_bond
#
# ep_etf = PrepFactor(factor_name='factor_ep_etf', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(ep_etf)
# del ep_etf
#
# ep_fama = PrepFactor(factor_name='factor_ep_fama', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(ep_fama)
# del ep_fama

cond_ind_mom = PrepFactor(factor_name='factor_cond_ind_mom', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cond_ind_mom, categorical=True)
del cond_ind_mom

# ind_vwr = PrepFactor(factor_name='factor_ind_vwr', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=True).prep()
# alpha.add_factor(ind_vwr)
# del ind_vwr

# rank_ind_vwr = PrepFactor(factor_name='factor_rank_ind_vwr', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=True).prep()
# alpha.add_factor(rank_ind_vwr, categorical=True)
# del rank_ind_vwr

rank_ind_mom = PrepFactor(factor_name='factor_rank_ind_mom', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank_ind_mom, categorical=True)
del rank_ind_mom

# sb_lag_bond = PrepFactor(factor_name='factor_sb_lag_bond', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_lag_bond)
# del sb_lag_bond


elapsed_time = time.time() - start_time
print(f"AlphaModel data shape: {alpha.data.shape}")
print(f"Prep and Add took: {round(elapsed_time)} seconds")
print("-" * 60)
print("Run Model")
alpha.lightgbm()


