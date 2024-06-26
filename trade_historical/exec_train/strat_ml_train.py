from class_model.model_cboost import ModelCboost
from class_model.model_lgbm import ModelLgbm

from class_model.model_catboost import ModelCatboost
from class_model.model_lightgbm import ModelLightgbm
from class_model.model_prep import ModelPrep
from class_model.model_rf import ModelRf

from class_model.model_randomforest import ModelRandomforest
from core.operation import *

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
live = False

if live:
    stock = read_stock(get_large(live) / 'permno_live.csv')
else:
    stock = read_stock(get_large(live) / 'permno_to_train_fund.csv')

start = '2008-01-01'
end = '2023-01-01'
save = False
normalize = 'rank_normalize'
impute = 'cross_median'

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

catboost_params = {
    'max_depth':          {'optuna': ('suggest_categorical', [4, 6, 8]),        'gridsearch': [4, 6, 8],                     'default': 6},
    'learning_rate':      {'optuna': ('suggest_float', 0.10, 0.50, False),      'gridsearch': [0.005, 0.01, 0.1, 0.15],      'default': 0.03},
    'num_leaves':         {'optuna': ('suggest_int', 5, 150),                   'gridsearch': [20, 40, 60],                  'default': 31},
    'min_child_samples':  {'optuna': ('suggest_float', 0.5, 1.0),               'gridsearch': [0.7, 0.8, 0.9],               'default': 1},
    'l2_leaf_reg':        {'optuna': ('suggest_float', 1e-5, 10, True),         'gridsearch': [0.0001, 0.001, 0.01],         'default': 3.0}
}

randomforest_params = {
    'n_estimators':       {'optuna': ('suggest_int', 100, 1000),                            'gridsearch': [100, 300, 500, 800],               'default': 50},
    'max_depth':          {'optuna': ('suggest_int', 4, 30),                                'gridsearch': [4, 6, 8, 12, 16, 20],              'default': 6},
    'min_samples_split':  {'optuna': ('suggest_int', 2, 10),                                'gridsearch': [2, 4, 6, 8],                       'default': 2},
    'min_samples_leaf':   {'optuna': ('suggest_int', 1, 4),                                 'gridsearch': [1, 2, 3],                          'default': 1}
}

start_time = time.time()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
alpha = ModelLightgbm(live=live, model_name='lightgbm_trial_106', end=end, tuning='default', shap=False, plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, trend=0,
                           incr=True, opt='wfo', weight=False, outlier=False, early=True, pretrain_len=1260, train_len=504, valid_len=63, test_len=21, **lightgbm_params)

alpha = ModelRandomforest(live=live, model_name='randomforest_trial_10', end=end, tuning='default', plot_loss=False, plot_hist=False, pred='sign', stock='permno', lookahead=1,
                   trend=1, opt='ewo', outlier=False, train_len=504, valid_len=21, test_len=21, **randomforest_params)

alpha = ModelCatboost(live=live, model_name='catboost_trial_1', tuning='default', plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, opt='ewo',
                    outlier=False, early=True, train_len=1260, valid_len=252, test_len=21, **catboost_params)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
ret = ModelPrep(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ret)
del ret

ret_comp = ModelPrep(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ret_comp, normalize=normalize, impute=impute)
del ret_comp

cycle = ModelPrep(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cycle, categorical=True)
del cycle

talib = ModelPrep(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(talib, normalize=normalize, impute=impute)
del talib

volume = ModelPrep(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
alpha.add_factor(volume, normalize=normalize, impute=impute)
del volume

volatility = ModelPrep(live=live, factor_name='factor_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(volatility, normalize=normalize, impute=impute)
del volatility

sign_ret = ModelPrep(live=live, factor_name='factor_sign_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sign_ret, categorical=True)
del sign_ret

# macro = PrepFactor(trade_live=trade_live, factor_name='factor_macro', group='permno', interval='D', kind='macro', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(macro)
# del macro

vol_comp = ModelPrep(live=live, factor_name='factor_vol_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(vol_comp, normalize=normalize, impute=impute)
del vol_comp

# sign_volume = ModelPrep(live=live, factor_name='factor_sign_volume', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sign_volume, categorical=True)
# del sign_volume
#
# sign_volatility = ModelPrep(live=live, factor_name='factor_sign_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sign_volatility, categorical=True)
# del sign_volatility

# fund_raw = PrepFactor(trade_live=trade_live, factor_name='factor_fund_raw', group='permno', interval='D', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(fund_raw)
# del fund_raw

# fund_q = PrepFactor(trade_live=trade_live, factor_name='factor_fund_q', group='permno', interval='D', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(fund_q, categorical=True)
# del fund_q

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_ret, normalize=normalize, impute=impute)
del load_ret

load_volume = ModelPrep(live=live, factor_name='factor_load_volume', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_volume, normalize=normalize, impute=impute)
del load_volume

load_volatility = ModelPrep(live=live, factor_name='factor_load_volatility', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(load_volatility, normalize=normalize, impute=impute)
del load_volatility

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------CONDITION-----------------------------------------------------------------------------------------
cond_ret = ModelPrep(live=live, factor_name='factor_cond_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cond_ret, categorical=True)
del cond_ret

# cond_ind_mom = PrepFactor(live=live, factor_name='factor_cond_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(cond_ind_mom, categorical=True)
# del cond_ind_mom
#
# cond_ind_mom_fama = PrepFactor(live=live, factor_name='factor_cond_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(cond_ind_mom_fama, categorical=True)
# del cond_ind_mom_fama
#
# cond_ind_mom_sub = PrepFactor(live=live, factor_name='factor_cond_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(cond_ind_mom_sub, categorical=True)
# del cond_ind_mom_sub

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
ind = ModelPrep(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind, categorical=True)
del ind

ind_fama = ModelPrep(live=live, factor_name='factor_ind_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_fama, categorical=True)
del ind_fama

ind_sub = ModelPrep(live=live, factor_name='factor_ind_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_sub, categorical=True)
del ind_sub

# ind_naic = PrepFactor(trade_live=trade_live, factor_name='factor_ind_naic', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ind_naic, categorical=True, compress=True)
# del ind_naic

ind_mom = ModelPrep(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_mom, normalize=normalize, impute=impute)
del ind_mom

ind_mom_fama = ModelPrep(live=live, factor_name='factor_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_mom_fama, normalize=normalize, impute=impute)
del ind_mom_fama

ind_mom_sub = ModelPrep(live=live, factor_name='factor_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ind_mom_sub, normalize=normalize, impute=impute)
del ind_mom_sub

# ind_vwr = ModelPrep(live=live, factor_name='factor_ind_vwr', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ind_vwr, normalize=normalize, impute=impute)
# del ind_vwr
#
# ind_vwr_fama = ModelPrep(live=live, factor_name='factor_ind_vwr_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ind_vwr_fama, normalize=normalize, impute=impute)
# del ind_vwr_fama
#
# ind_vwr_sub = ModelPrep(live=live, factor_name='factor_ind_vwr_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ind_vwr_sub, normalize=normalize, impute=impute)
# del ind_vwr_sub

# ind_mom_comp = PrepFactor(trade_live=trade_live, factor_name='factor_ind_mom_comp', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ind_mom_comp)
# del ind_mom_comp

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------OPEN ASSET----------------------------------------------------------------------------------------
age_mom = ModelPrep(live=live, factor_name='factor_age_mom', group='permno', interval='D', kind='age', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(age_mom, normalize=normalize, impute=impute)
del age_mom

net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(net_debt_finance, normalize=normalize, impute=impute)
del net_debt_finance

chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(chtax, normalize=normalize, impute=impute)
del chtax

asset_growth = ModelPrep(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(asset_growth, normalize=normalize, impute=impute)
del asset_growth

mom_season_short = ModelPrep(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season_short, normalize=normalize, impute=impute)
del mom_season_short

mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season, normalize=normalize, impute=impute)
del mom_season

noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(noa, normalize=normalize, impute=impute)
del noa

invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(invest_ppe, normalize=normalize, impute=impute)
del invest_ppe

inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(inv_growth, normalize=normalize, impute=impute)
del inv_growth

trend_factor = ModelPrep(live=live, factor_name='factor_trend_factor', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(trend_factor, normalize=normalize, impute=impute)
del trend_factor

mom_season6 = ModelPrep(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season6, normalize=normalize, impute=impute)
del mom_season6

mom_season11 = ModelPrep(live=live, factor_name='factor_mom_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season11, normalize=normalize, impute=impute)
del mom_season11

mom_season16 = ModelPrep(live=live, factor_name='factor_mom_season16', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_season16, normalize=normalize, impute=impute)
del mom_season16

comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(comp_debt, normalize=normalize, impute=impute)
del comp_debt

mom_vol = ModelPrep(live=live, factor_name='factor_mom_vol', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_vol, categorical=True)
del mom_vol

int_mom = ModelPrep(live=live, factor_name='factor_int_mom', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(int_mom, normalize=normalize, impute=impute)
del int_mom

cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(cheq, normalize=normalize, impute=impute)
del cheq

xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(xfin, normalize=normalize, impute=impute)
del xfin

emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(emmult, normalize=normalize, impute=impute)
del emmult

accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(accrual, normalize=normalize, impute=impute)
del accrual

frontier = ModelPrep(live=live, factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(frontier, normalize=normalize, impute=impute)
del frontier

mom_rev = ModelPrep(live=live, factor_name='factor_mom_rev', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_rev, categorical=True)
del mom_rev

hire = ModelPrep(live=live, factor_name='factor_hire', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(hire, normalize=normalize, impute=impute)
del hire

rds = ModelPrep(live=live, factor_name='factor_rds', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(rds, normalize=normalize, impute=impute)
del rds

pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(pcttoacc, normalize=normalize, impute=impute)
del pcttoacc

accrual_bm = ModelPrep(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(accrual_bm, normalize=normalize, impute=impute)
del accrual_bm

mom_off_season = ModelPrep(live=live, factor_name='factor_mom_off_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(mom_off_season, normalize=normalize, impute=impute)
del mom_off_season

earning_streak = ModelPrep(live=live, factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(earning_streak, normalize=normalize, impute=impute)
del earning_streak

ms = ModelPrep(live=live, factor_name='factor_ms', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(ms, categorical=True)
del ms

dividend = ModelPrep(live=live, factor_name='factor_dividend', group='permno', interval='D', kind='dividend', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(dividend, categorical=True)
del dividend

# div_season = PrepFactor(trade_live=trade_live, factor_name='factor_div_season', group='permno', interval='D', kind='dividend', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(div_season, categorical=True)
# del div_season

# grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(grcapx, normalize=normalize, impute=impute)
# del grcapx
#
# gradexp = ModelPrep(live=live, factor_name='factor_gradexp', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(gradexp, normalize=normalize, impute=impute)
# del gradexp
#
# ret_skew = ModelPrep(live=live, factor_name='factor_ret_skew', group='permno', interval='D', kind='skew', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ret_skew, normalize=normalize, impute=impute)
# del ret_skew
#
# size = ModelPrep(live=live, factor_name='factor_size', group='permno', interval='D', kind='size', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(size, normalize=normalize, impute=impute)
# del size
#
# ret_max = ModelPrep(live=live, factor_name='factor_ret_max', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(ret_max, normalize=normalize, impute=impute)
# del ret_max
#
# mom_off_season6 = ModelPrep(live=live, factor_name='factor_mom_off_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(mom_off_season6, normalize=normalize, impute=impute)
# del mom_off_season6
#
# mom_off_season11 = ModelPrep(live=live, factor_name='factor_mom_off_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(mom_off_season11, normalize=normalize, impute=impute)
# del mom_off_season11

# mom_season9 = PrepFactor(live=live, factor_name='factor_mom_season9', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(mom_season9)
# del mom_season9
#
# mom_season21 = PrepFactor(live=live, factor_name='factor_mom_season21', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(mom_season21)
# del mom_season21

# mom_season_shorter = PrepFactor(live=live, factor_name='factor_mom_season_shorter', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(mom_season_shorter)
# del mom_season_shorter

# earning_disparity = PrepFactor(trade_live=trade_live, factor_name='factor_earning_disparity', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(earning_disparity)
# del earning_disparity

# resid_mom = PrepFactor(trade_live=trade_live, factor_name='factor_resid_mom', group='permno', interval='D', kind='trend', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(resid_mom)
# del resid_mom

# abnormal_accrual = PrepFactor(trade_live=trade_live, factor_name='factor_abnormal_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(abnormal_accrual)
# del abnormal_accrual

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------BETAS---------------------------------------------------------------------------------------------
sb_pca = ModelPrep(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_pca, normalize=normalize, impute=impute)
del sb_pca

sb_sector = ModelPrep(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(sb_sector, normalize=normalize, impute=impute)
del sb_sector

# sb_fama = PrepFactor(trade_live=trade_live, factor_name='factor_sb_fama', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_fama)
# del sb_fama
#
# sb_bond = PrepFactor(trade_live=trade_live, factor_name='factor_sb_bond', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_bond)
# del sb_bond

# sb_pca_copy = PrepFactor(trade_live=trade_live, factor_name='factor_sb_pca_copy', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_pca_copy)
# del sb_pca_copy
#
# sb_sector_copy = PrepFactor(trade_live=trade_live, factor_name='factor_sb_sector_copy', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_sector_copy)
# del sb_sector_copy

# sb_inverse = ModelPrep(live=live, factor_name='factor_sb_inverse', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_inverse, normalize=normalize, impute=impute)
# del sb_inverse

# sb_oil = PrepFactor(trade_live=trade_live, factor_name='factor_sb_oil', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_oil)
# del sb_oil

# sb_ind = PrepFactor(trade_live=trade_live, factor_name='factor_sb_ind', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_ind)
# del sb_ind

# sb_overall = PrepFactor(trade_live=trade_live, factor_name='factor_sb_overall', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_overall)
# del sb_overall

# sb_etf = PrepFactor(trade_live=trade_live, factor_name='factor_sb_etf', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_etf)
# del sb_etf

# rf_ret = PrepFactor(trade_live=trade_live, factor_name='factor_rf_ret', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_ret)
# del rf_ret

# rf_volume = PrepFactor(trade_live=trade_live, factor_name='factor_rf_volume', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_volume)
# del rf_volume

# rf_sign = PrepFactor(trade_live=trade_live, factor_name='factor_rf_sign', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rf_sign)
# del rf_sign

# sb_macro = PrepFactor(trade_live=trade_live, factor_name='factor_sb_macro', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_macro)
# del sb_macro

# sb_fund_ind = PrepFactor(trade_live=trade_live, factor_name='factor_sb_fund_ind', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_fund_ind)
# del sb_fund_ind

# sb_fund_raw = PrepFactor(trade_live=trade_live, factor_name='factor_sb_fund_raw', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(sb_fund_raw)
# del sb_fund_raw

# mf_ret = PrepFactor(trade_live=trade_live, factor_name='factor_mf_ret', group='permno', interval='D', kind='price', div=False, stock=stock, start=start, end=end, save=save).prep()
# alpha.add_factor(mf_ret)
# del mf_ret

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------RANK----------------------------------------------------------------------------------------------
# rank_load_ret = PrepFactor(trade_live=trade_live, factor_name='factor_rank_load_ret', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_load_ret, categorical=True, compress=True)
# del rank_load_ret
#
# rank_ret = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ret', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ret, categorical=True, compress=True)
# del rank_ret
#
# rank_fund_raw = PrepFactor(trade_live=trade_live, factor_name='factor_rank_fund_raw', group='permno', interval='M', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_fund_raw, categorical=True, compress=True)
# del rank_fund_raw
#
# rank_volume = PrepFactor(trade_live=trade_live, factor_name='factor_rank_volume', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_volume, categorical=True, compress=True)
# del rank_volume
#
# rank_volatility = PrepFactor(trade_live=trade_live, factor_name='factor_rank_volatility', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_volatility, categorical=True, compress=True)
# del rank_volatility
#
# rank_ind_vwr = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ind_vwr', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ind_vwr, categorical=True)
# del rank_ind_vwr
#
# rank_ind_vwr_fama = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ind_vwr_fama', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ind_vwr_fama, categorical=True)
# del rank_ind_vwr_fama
#
# rank_ind_vwr_sub = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ind_vwr_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ind_vwr_sub, categorical=True)
# del rank_ind_vwr_sub
#
# rank_ind_mom = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ind_mom', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ind_mom, categorical=True, compress=True)
# del rank_ind_mom
#
# rank_ind_mom_fama = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ind_mom_fama', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ind_mom_fama, categorical=True)
# del rank_ind_mom_fama
#
# rank_ind_mom_sub = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ind_mom_sub, categorical=True, compress=True)
# del rank_ind_mom_sub
#
# rank_ret_comp = PrepFactor(trade_live=trade_live, factor_name='factor_rank_ret_comp', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_ret_comp, categorical=True, compress=True)
# del rank_ret_comp

# rank_load_volume = PrepFactor(trade_live=trade_live, factor_name='factor_rank_load_volume', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_load_volume, categorical=True, compress=True)
# del rank_load_volume

# rank_sb_fama = PrepFactor(trade_live=trade_live, factor_name='factor_rank_sb_fama', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_fama, categorical=True, compress=True)
# del rank_sb_fama

# rank_sb_bond = PrepFactor(trade_live=trade_live, factor_name='factor_rank_sb_bond', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_bond, categorical=True, compress=True)
# del rank_sb_bond
#
# rank_sb_pca = PrepFactor(trade_live=trade_live, factor_name='factor_rank_sb_pca', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_pca, categorical=True, compress=True)
# del rank_sb_pca

# rank_sb_inverse = PrepFactor(trade_live=trade_live, factor_name='factor_rank_sb_inverse', group='permno', interval='D', kind='rank', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(rank_sb_inverse, categorical=True, compress=True)
# del rank_sb_inverse

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
clust_ret = ModelPrep(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ret, categorical=True)
del clust_ret

clust_load_ret = ModelPrep(live=live, factor_name='factor_clust_load_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_load_ret, categorical=True)
del clust_load_ret

# clust_fund_raw = PrepFactor(trade_live=trade_live, factor_name='factor_clust_fund_raw', group='permno', interval='M', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_fund_raw, categorical=True)
# del clust_fund_raw

clust_ind_mom = ModelPrep(live=live, factor_name='factor_clust_ind_mom', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ind_mom, categorical=True)
del clust_ind_mom

clust_ind_mom_fama = ModelPrep(live=live, factor_name='factor_clust_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ind_mom_fama, categorical=True)
del clust_ind_mom_fama

clust_ind_mom_sub = ModelPrep(live=live, factor_name='factor_clust_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start, end=end, save=save).prep()
alpha.add_factor(clust_ind_mom_sub, categorical=True)
del clust_ind_mom_sub

# clust_mom_season = PrepFactor(trade_live=trade_live, factor_name='factor_clust_mom_season', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_mom_season, categorical=True)
# del clust_mom_season

# clust_ret_comp = PrepFactor(trade_live=trade_live, factor_name='factor_clust_ret_comp', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_ret_comp, categorical=True)
# del clust_ret_comp

# clust_load_volume = ModelPrep(live=live, factor_name='factor_clust_load_volume', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_load_volume, categorical=True)
# del clust_load_volume
#
# clust_volatility = ModelPrep(live=live, factor_name='factor_clust_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volatility, categorical=True)
# del clust_volatility
#
# clust_volume = ModelPrep(live=live, factor_name='factor_clust_volume', group='permno', interval='D', kind='price', stock=stock, div=False, start=start, end=end, save=save).prep()
# alpha.add_factor(clust_volume, categorical=True)
# del clust_volume

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------TRAINING------------------------------------------------------------------------------------------
elapsed_time = time.time() - start_time
print("-" * 60)
print(f"AlphaModel Data Shape: {alpha.data.shape}")
print(f"Prep and Add took: {round(elapsed_time)} seconds")
print("-" * 60)
print("Run Model")

alpha.exec_train()



