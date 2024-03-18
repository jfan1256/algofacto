from class_model.model_fnn import ModelFNN
from class_model.model_lregression import ModelLRegression
from class_model.model_prep import ModelPrep
from core.operation import *

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
live = True
save = True
start_model = '2008-01-01'
current_date = '2025-01-01'
normalize = 'rank_normalize'
impute = 'cross_median'
format_end = date.today().strftime('%Y%m%d')
model_name = f'lregression_{format_end}'
tune = 'default'

if live:
    stock = read_stock(get_large(live) / 'permno_live.csv')
else:
    stock = read_stock(get_large(live) / 'permno_to_train_fund.csv')

lr_params = {
    'alpha':     {'optuna': ('suggest_float', 1e-5, 1),     'gridsearch': [1e-3, 1e-4, 1e-5],      'default': 0.01},
    'l1_ratio':  {'optuna': ('suggest_float', 1e-5, 1),     'gridsearch': [1e-3, 1e-4, 1e-5],      'default': 0.005},
}

start_time = time.time()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
alpha = ModelLRegression(live=live, model_name=model_name, tuning=tune, plot_hist=False, pred='price', model='elastic', stock='permno',
                         lookahead=1, trend=0, opt='ewo', outlier=False, train_len=504, valid_len=21, test_len=21, **lr_params)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
ret = ModelPrep(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(ret, impute=impute)
del ret

# ret_comp = ModelPrep(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ret_comp, normalize=normalize, impute=impute)
# del ret_comp

# cycle = ModelPrep(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(cycle, categorical=True)
# del cycle

# talib = ModelPrep(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(talib, normalize=normalize, impute=impute)
# del talib

# volume = ModelPrep(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(volume, normalize=normalize, impute=impute)
# del volume

# volatility = ModelPrep(live=live, factor_name='factor_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(volatility, normalize=normalize, impute=impute)
# del volatility

# sign_ret = ModelPrep(live=live, factor_name='factor_sign_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sign_ret, categorical=True)
# del sign_ret
#
# vol_comp = ModelPrep(live=live, factor_name='factor_vol_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(vol_comp, normalize=normalize, impute=impute)
# del vol_comp
#
# sign_volume = ModelPrep(live=live, factor_name='factor_sign_volume', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sign_volume, categorical=True)
# del sign_volume
#
# sign_volatility = ModelPrep(live=live, factor_name='factor_sign_volatility', group='permno', interval='D', kind='sign', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sign_volatility, categorical=True)
# del sign_volatility
#
# fund_raw = ModelPrep(live=live, factor_name='factor_fund_q', group='permno', interval='D', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(fund_raw, categorical=True)
# del fund_raw

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
alpha.add_factor(load_ret, normalize=normalize, impute=impute)
del load_ret

# load_volume = ModelPrep(live=live, factor_name='factor_load_volume', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(load_volume, normalize=normalize, impute=impute)
# del load_volume

# load_volatility = ModelPrep(live=live, factor_name='factor_load_volatility', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(load_volatility, normalize=normalize, impute=impute)
# del load_volatility

# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------CONDITION-----------------------------------------------------------------------------------------
# cond_ret = ModelPrep(live=live, factor_name='factor_cond_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(cond_ret, categorical=True)
# del cond_ret
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
# ind = ModelPrep(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind, categorical=True, impute=impute)
# del ind
#
# ind_fama = ModelPrep(live=live, factor_name='factor_ind_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_fama, categorical=True)
# del ind_fama
#
# ind_sub = ModelPrep(live=live, factor_name='factor_ind_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_sub, categorical=True)
# del ind_sub

# ind_mom = ModelPrep(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_mom, normalize=normalize, impute=impute)
# del ind_mom

# ind_mom_fama = ModelPrep(live=live, factor_name='factor_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_mom_fama, normalize=normalize, impute=impute)
# del ind_mom_fama
#
# ind_mom_sub = ModelPrep(live=live, factor_name='factor_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_mom_sub, normalize=normalize, impute=impute)
# del ind_mom_sub
#
# ind_vwr = ModelPrep(live=live, factor_name='factor_ind_vwr', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_vwr, normalize=normalize, impute=impute)
# del ind_vwr
#
# ind_vwr_fama = ModelPrep(live=live, factor_name='factor_ind_vwr_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_vwr_fama, normalize=normalize, impute=impute)
# del ind_vwr_fama
#
# ind_vwr_sub = ModelPrep(live=live, factor_name='factor_ind_vwr_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(ind_vwr_sub, normalize=normalize, impute=impute)
# del ind_vwr_sub

# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------OPEN ASSET MONTHLY------------------------------------------------------------------------------------
# net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# net_debt_finance = net_debt_finance.groupby('permno').shift(84)
# alpha.add_factor(net_debt_finance, normalize=normalize)
# del net_debt_finance
#
# chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# chtax = chtax.groupby('permno').shift(84)
# alpha.add_factor(chtax, normalize=normalize)
# del chtax
#
# asset_growth = ModelPrep(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# asset_growth = asset_growth.groupby('permno').shift(84)
# alpha.add_factor(asset_growth, normalize=normalize)
# del asset_growth
#
# noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# noa = noa.groupby('permno').shift(84)
# alpha.add_factor(noa, normalize=normalize)
# del noa
#
# invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# invest_ppe = invest_ppe.groupby('permno').shift(84)
# alpha.add_factor(invest_ppe, normalize=normalize)
# del invest_ppe
#
# inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# inv_growth = inv_growth.groupby('permno').shift(84)
# alpha.add_factor(inv_growth, normalize=normalize)
# del inv_growth
#
# comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# comp_debt = comp_debt.groupby('permno').shift(84)
# alpha.add_factor(comp_debt, normalize=normalize)
# del comp_debt
#
# cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# cheq = cheq.groupby('permno').shift(84)
# alpha.add_factor(cheq, normalize=normalize)
# del cheq
#
# xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# xfin = xfin.groupby('permno').shift(84)
# alpha.add_factor(xfin, normalize=normalize)
# del xfin
#
# emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# emmult = emmult.groupby('permno').shift(84)
# alpha.add_factor(emmult, normalize=normalize)
# del emmult
#
# accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# accrual = accrual.groupby('permno').shift(84)
# alpha.add_factor(accrual, normalize=normalize)
# del accrual
#
# frontier = ModelPrep(live=live, factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# frontier = frontier.groupby('permno').shift(84)
# alpha.add_factor(frontier, normalize=normalize)
# del frontier
#
# hire = ModelPrep(live=live, factor_name='factor_hire', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# hire = hire.groupby('permno').shift(84)
# alpha.add_factor(hire, normalize=normalize)
# del hire
#
# rds = ModelPrep(live=live, factor_name='factor_rds', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# rds = rds.groupby('permno').shift(84)
# alpha.add_factor(rds, normalize=normalize)
# del rds
#
# pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# pcttoacc = pcttoacc.groupby('permno').shift(84)
# alpha.add_factor(pcttoacc, normalize=normalize)
# del pcttoacc
#
# accrual_bm = ModelPrep(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# accrual_bm = accrual_bm.groupby('permno').shift(84)
# alpha.add_factor(accrual_bm, normalize=normalize)
# del accrual_bm
#
# earning_streak = ModelPrep(live=live, factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# earning_streak = earning_streak.groupby('permno').shift(84)
# alpha.add_factor(earning_streak, normalize=normalize)
# del earning_streak
#
# ms = ModelPrep(live=live, factor_name='factor_ms', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# ms = ms.groupby('permno').shift(84)
# alpha.add_factor(ms, categorical=True)
# del ms
#
# grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# grcapx = grcapx.groupby('permno').shift(84)
# alpha.add_factor(grcapx, normalize=normalize)
# del grcapx
#
# gradexp = ModelPrep(live=live, factor_name='factor_gradexp', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=True).prep()
# gradexp = gradexp.groupby('permno').shift(84)
# alpha.add_factor(gradexp, normalize=normalize)
# del gradexp

# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------BETAS---------------------------------------------------------------------------------------------
# sb_pca = ModelPrep(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sb_pca, normalize=normalize, impute=impute)
# del sb_pca

# sb_sector = ModelPrep(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(sb_sector, normalize=normalize, impute=impute)
# del sb_sector

# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
# clust_ret = ModelPrep(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_ret, categorical=True)
# del clust_ret
#
# clust_load_ret = ModelPrep(live=live, factor_name='factor_clust_load_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_load_ret, categorical=True)
# del clust_load_ret
#
# clust_ind_mom = ModelPrep(live=live, factor_name='factor_clust_ind_mom', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_ind_mom, categorical=True)
# del clust_ind_mom
#
# clust_ind_mom_fama = ModelPrep(live=live, factor_name='factor_clust_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_ind_mom_fama, categorical=True)
# del clust_ind_mom_fama
#
# clust_ind_mom_sub = ModelPrep(live=live, factor_name='factor_clust_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_ind_mom_sub, categorical=True)
# del clust_ind_mom_sub
#
# clust_load_volume = ModelPrep(live=live, factor_name='factor_clust_load_volume', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_load_volume, categorical=True)
# del clust_load_volume
#
# clust_volatility = ModelPrep(live=live, factor_name='factor_clust_volatility', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_volatility, categorical=True)
# del clust_volatility
#
# clust_volume = ModelPrep(live=live, factor_name='factor_clust_volume', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save).prep()
# alpha.add_factor(clust_volume, categorical=True)
# del clust_volume

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------TRAINING------------------------------------------------------------------------------------------
elapsed_time = time.time() - start_time
print("-" * 60)
print(f"Total time to prep and add all factors: {round(elapsed_time)} seconds")
print(f"AlphaModel Dataframe Shape: {alpha.data.shape}")
print("-" * 60)
print("Run Model")

alpha.exec_train()

