from class_model.model_lightgbm import ModelLightgbm
from class_model.model_prep import ModelPrep
from core.operation import *

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
live = False
save = False
start_model = '2008-01-01'
current_date = '2023-01-01'
normalize = 'rank_normalize'
model_name = 'lightgbm_trial_226'
tune = 'default'

if live:
    stock = read_stock(get_large(live) / 'permno_live.csv')
else:
    stock = read_stock(get_large(live) / 'permno_to_train_fund.csv')

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

start_time = time.time()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
alpha = ModelLightgbm(live=live, model_name=model_name, tuning=tune, shap=False, plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, trend=0,
                      incr=True, opt='wfo', outlier=False, early=True, pretrain_len=1260, train_len=504, valid_len=63, test_len=21, **lightgbm_params)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
ret = ModelPrep(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(ret)
del ret

ret_comp = ModelPrep(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(ret_comp, normalize=normalize)
del ret_comp

cycle = ModelPrep(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(cycle, categorical=True)
del cycle

talib = ModelPrep(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(talib, normalize=normalize)
del talib

volume = ModelPrep(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(volume, normalize=normalize)
del volume

# volatility = ModelPrep(live=live, factor_name='factor_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(volatility, normalize=normalize)
# del volatility

# sign_ret = ModelPrep(live=live, factor_name='factor_sign_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sign_ret, categorical=True)
# del sign_ret

# vol_comp = ModelPrep(live=live, factor_name='factor_vol_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(vol_comp, normalize=normalize)
# del vol_comp

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(load_ret, normalize=normalize)
del load_ret

# load_volume = ModelPrep(live=live, factor_name='factor_load_volume', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(load_volume, normalize=normalize)
# del load_volume

# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------CONDITION-----------------------------------------------------------------------------------------
# cond_ret = ModelPrep(live=live, factor_name='factor_cond_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(cond_ret, categorical=True)
# del cond_ret

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
ind = ModelPrep(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(ind, categorical=True)
del ind

# ind_fama = ModelPrep(live=live, factor_name='factor_ind_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_fama, categorical=True)
# del ind_fama

# ind_sub = ModelPrep(live=live, factor_name='factor_ind_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_sub, categorical=True)
# del ind_sub

ind_mom = ModelPrep(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(ind_mom, normalize=normalize)
del ind_mom

# ind_mom_excess = ModelPrep(live=live, factor_name='factor_ind_mom_excess', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# alpha.add_factor(ind_mom_excess, normalize=normalize)
# del ind_mom_excess

# ind_vwr = ModelPrep(live=live, factor_name='factor_ind_vwr', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_vwr, normalize=normalize)
# del ind_vwr

# ind_mom_fama = ModelPrep(live=live, factor_name='factor_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_mom_fama, normalize=normalize)
# del ind_mom_fama
#
# ind_mom_sub = ModelPrep(live=live, factor_name='factor_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_mom_sub, normalize=normalize)
# del ind_mom_sub
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------OPEN ASSET----------------------------------------------------------------------------------------
# age_mom = ModelPrep(live=live, factor_name='factor_age_mom', group='permno', interval='D', kind='age', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(age_mom, normalize=normalize)
# del age_mom

net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(net_debt_finance, normalize=normalize)
del net_debt_finance

chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(chtax, normalize=normalize)
del chtax

asset_growth = ModelPrep(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(asset_growth, normalize=normalize)
del asset_growth
#
# mom_season_short = ModelPrep(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_season_short, normalize=normalize)
# del mom_season_short
#
# mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_season, normalize=normalize)
# del mom_season
#
noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(noa, normalize=normalize)
del noa

invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(invest_ppe, normalize=normalize)
del invest_ppe

inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(inv_growth, normalize=normalize)
del inv_growth
#
# trend_factor = ModelPrep(live=live, factor_name='factor_trend_factor', group='permno', interval='D', kind='trend', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(trend_factor, normalize=normalize)
# del trend_factor
#
# mom_season6 = ModelPrep(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_season6, normalize=normalize)
# del mom_season6
#
# mom_season11 = ModelPrep(live=live, factor_name='factor_mom_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_season11, normalize=normalize)
# del mom_season11
#
# mom_season16 = ModelPrep(live=live, factor_name='factor_mom_season16', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_season16, normalize=normalize)
# del mom_season16

comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(comp_debt, normalize=normalize)
del comp_debt
#
# mom_vol = ModelPrep(live=live, factor_name='factor_mom_vol', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_vol, categorical=True)
# del mom_vol
#
# int_mom = ModelPrep(live=live, factor_name='factor_int_mom', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(int_mom, normalize=normalize)
# del int_mom

cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(cheq, normalize=normalize)
del cheq

xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(xfin, normalize=normalize)
del xfin

emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(emmult, normalize=normalize)
del emmult

accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(accrual, normalize=normalize)
del accrual
#
# frontier = ModelPrep(live=live, factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(frontier, normalize=normalize)
# del frontier
#
# mom_rev = ModelPrep(live=live, factor_name='factor_mom_rev', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_rev, categorical=True)
# del mom_rev
#
# hire = ModelPrep(live=live, factor_name='factor_hire', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(hire, normalize=normalize)
# del hire
#
# rds = ModelPrep(live=live, factor_name='factor_rds', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(rds, normalize=normalize)
# del rds
#
# pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(pcttoacc, normalize=normalize)
# del pcttoacc
#
# accrual_bm = ModelPrep(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(accrual_bm, normalize=normalize)
# del accrual_bm
#
# mom_off_season = ModelPrep(live=live, factor_name='factor_mom_off_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(mom_off_season, normalize=normalize)
# del mom_off_season
#
# grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(grcapx, normalize=normalize)
# del grcapx
#
# earning_streak = ModelPrep(live=live, factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(earning_streak, normalize=normalize)
# del earning_streak
#
# ret_skew = ModelPrep(live=live, factor_name='factor_ret_skew', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ret_skew, normalize=normalize)
# del ret_skew
#
# dividend = ModelPrep(live=live, factor_name='factor_dividend', group='permno', interval='D', kind='dividend', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(dividend, categorical=True)
# del dividend
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------BETAS---------------------------------------------------------------------------------------------
# sb_pca = ModelPrep(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sb_pca, normalize=normalize)
# del sb_pca
#
# sb_sector = ModelPrep(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sb_sector, normalize=normalize)
# del sb_sector
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
clust_ret = ModelPrep(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(clust_ret, categorical=True)
del clust_ret

# clust_load_ret = ModelPrep(live=live, factor_name='factor_clust_load_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_load_ret, categorical=True)
# del clust_load_ret

# clust_ind_mom = ModelPrep(live=live, factor_name='factor_clust_ind_mom', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_ind_mom, categorical=True)
# del clust_ind_mom

# clust_ind_mom_fama = ModelPrep(live=live, factor_name='factor_clust_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_ind_mom_fama, categorical=True)
# del clust_ind_mom_fama
#
# clust_ind_mom_sub = ModelPrep(live=live, factor_name='factor_clust_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_ind_mom_sub, categorical=True)
# del clust_ind_mom_sub

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------TRAINING------------------------------------------------------------------------------------------
elapsed_time = time.time() - start_time
print("-" * 60)
print(f"Total time to prep and add all factors: {round(elapsed_time)} seconds")
print(f"AlphaModel Dataframe Shape: {alpha.data.shape}")
print("-" * 60)
print("Run Model")

alpha.exec_train()