from prep_factor import PrepFactor
from alpha_model import AlphaModel

from functions.utils.func import *

tickers = read_ticker(get_load_data_large_dir() / 'tickers_to_train_fundamental.csv')

start = '2010-01-01'
end = '2022-01-01'
save = False
params = {
    'max_depth':         {'optuna': ('suggest_int', 6, 16),                    'gridsearch': [-1, 6, 10],                  'default': -1},
    'learning_rate':     {'optuna': ('suggest_float', 0.12, 0.15, False),      'gridsearch': [0.005, 0.01, 0.1, 0.15],     'default': 0.15},
    'num_leaves':        {'optuna': ('suggest_int', 10, 32),                   'gridsearch': [20, 40, 60],                 'default': 15},
    'feature_fraction':  {'optuna': ('suggest_float', 0.70, 0.90),             'gridsearch': [0.7, 0.8, 0.9],              'default': 0.85},
    'min_gain_to_split': {'optuna': ('suggest_float', 0.02, 0.02,  False),    'gridsearch': [0.0001, 0.001, 0.01],        'default': 0.02},
    'min_data_in_leaf':  {'optuna': ('suggest_int', 60, 60),                  'gridsearch': [40, 60, 80],                 'default': 60},
    'lambda_l1':         {'optuna': ('suggest_float', 0, 0, False),            'gridsearch': [0.001, 0.01],                'default': 0},
    'lambda_l2':         {'optuna': ('suggest_float', 1e-5, 1, True),          'gridsearch': [0.001, 0.01],                'default': 0.01},
    'bagging_fraction':  {'optuna': ('suggest_float', 1.0, 1.0),               'gridsearch': [0.9, 1],                     'default': 1},
    'bagging_freq':      {'optuna': ('suggest_int', 0, 0),                     'gridsearch': [0, 20],                      'default': 0},
}

start_time = time.time()
alpha = AlphaModel(model_name='lightgbm_trial_25', tuning='default', plot_loss=False, pred='price',
                   lookahead=1, incr=True, pretrain_len=1260, train_len=504, test_len=21, **params)

ret = PrepFactor(factor_name='factor_ret', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ret)
del ret

cycle = PrepFactor(factor_name='factor_time', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cycle, categorical=True)
del cycle

# talib = PrepFactor(factor_name='factor_talib', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(talib)
# del talib

ind = PrepFactor(factor_name='factor_ind', interval='D', kind='ind', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind, categorical=True)
del ind

# fund_ratio = PrepFactor(factor_name='factor_fund_ratio', interval='D', kind='fundamental', tickers=tickers, div=True, start=start, end=end, save=save).prep()
# alphaModel.add_factor(fund_ratio)
# del fund_ratio

# macro = PrepFactor(factor_name='factor_macro', interval='D', kind='macro', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alphaModel.add_factor(macro)
# del macro

volume = PrepFactor(factor_name='factor_volume', interval='D', kind='price', div=False, tickers=tickers, start=start, end=end, save=save).prep()
alpha.add_factor(volume)
del volume

volatility = PrepFactor(factor_name='factor_volatility', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(volatility)
del volatility

# ind_mom = PrepFactor(factor_name='factor_ind_mom', interval='D', kind='ind', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ind_mom)
# del ind_mom

sb_fama = PrepFactor(factor_name='factor_sb_fama', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_fama)
del sb_fama

sb_etf = PrepFactor(factor_name='factor_sb_etf', interval='D', kind='price', div=False, tickers=tickers, start=start, end=end, save=save).prep()
alpha.add_factor(sb_etf)
del sb_etf

rf_ret = PrepFactor(factor_name='factor_rf_ret', interval='D', kind='price', div=False, tickers=tickers, start=start, end=end, save=save).prep()
alpha.add_factor(rf_ret)
del rf_ret

load_ret = PrepFactor(factor_name='factor_load_ret', interval='D', kind='loading', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_ret)
del load_ret

clust_ret = PrepFactor(factor_name='factor_clust_ret', interval='D', kind='cluster', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ret, categorical=True)
del clust_ret

clust_load_ret = PrepFactor(factor_name='factor_clust_load_ret', interval='D', kind='cluster', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_load_ret, categorical=True)
del clust_load_ret

# open_asset = PrepFactor(factor_name='factor_open_asset', interval='M', kind='open', tickers='all', div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(open_asset)
# del open_asset

# streversal = PrepFactor(factor_name='factor_streversal', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(streversal)
# del streversal

rank = PrepFactor(factor_name='factor_rank', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rank)
del rank

sb_bond = PrepFactor(factor_name='factor_sb_bond', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_bond)
del sb_bond

sign = PrepFactor(factor_name='factor_sign', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sign, categorical=True)
del sign

# rf_volume = PrepFactor(factor_name='factor_rf_volume', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_volume)
# del rf_volume
#
# rf_sign = PrepFactor(factor_name='factor_rf_sign', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_sign)
# del rf_sign

# sb_spy_inv = PrepFactor(factor_name='factor_sb_spy_inv', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_spy_inv)
# del sb_spy_inv

sb_macro = PrepFactor(factor_name='factor_sb_macro', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_macro)
del sb_macro

# clust_volume = PrepFactor(factor_name='factor_clust_volume', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volume, categorical=True)
# del clust_volume

# clust_ind_mom = PrepFactor(factor_name='factor_clust_ind_mom', interval='D', kind='cluster', tickers='all', div=False, start=start, end=end, save=True).prep()
# alpha.add_factor(clust_ind_mom, categorical=True)
# del clust_ind_mom

# load_volatility = PrepFactor(factor_name='factor_load_volatility', interval='D', kind='loading', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(load_volatility)
# del load_volatility

load_volume = PrepFactor(factor_name='factor_load_volume', interval='D', kind='loading', tickers=tickers, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_volume)
del load_volume

# clust_volatility = PrepFactor(factor_name='factor_clust_volatility', interval='D', kind='price', tickers=tickers, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volatility, categorical=True)
# del clust_volatility

# factor_dividend = PrepFactor(factor_name='factor_dividend', interval='D', kind='dividend', tickers='all', div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(factor_dividend, categorical=True)
# del factor_dividend

elapsed_time = time.time() - start_time
print(f"AlphaModel data shape: {alpha.data.shape}")
print(f"Prep and Add took: {round(elapsed_time)} seconds")
print("-" * 60)
print("Run Model")
alpha.lightgbm()
