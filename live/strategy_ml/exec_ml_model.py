from prep_factor import PrepFactor
from alpha_model import AlphaModel

from functions.utils.func import *

def exec_ml_model(start_model, tune, save_prep):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
    print("---------------------------------------------------------------------TRAIN MODEL------------------------------------------------------------------------------------------")
    live = True
    current_date = (date.today()).strftime('%Y-%m-%d')
    total_time = time.time()

    stock = read_stock(get_large_dir(live) / 'permno_live.csv')

    start_time = time.time()

    best_params = {
        'max_depth': [6, 6, 6, 6, 6, 6],
        'learning_rate': [0.1005147, 0.1088923, 0.1160839, 0.1304572, 0.1365527, 0.1866055],
        'num_leaves': [57, 120, 42, 116, 119, 136],
        'feature_fraction': [1, 1, 1, 1, 1, 1],
        'min_gain_to_split': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        'min_data_in_leaf': [200, 173, 189, 185, 135, 157],
        'lambda_l1': [0, 0, 0, 0, 0, 0],
        'lambda_l2': [0.2773845686298494, 2.8e-05, 0.0079915, 8.35e-05, 1e-05, 0.0153158],
        'bagging_fraction': [1, 1, 1, 1, 1, 1],
        'bagging_freq': [0, 0, 0, 0, 0, 0]
    }

    lightgbm_params = {
        'max_depth':          {'optuna': ('suggest_categorical', [6]),            'gridsearch': [4, 6, 8],                     'default': 6,       'best': best_params['max_depth']},
        'learning_rate':      {'optuna': ('suggest_float', 0.10, 0.50, False),    'gridsearch': [0.005, 0.01, 0.1, 0.15],      'default': 0.15,    'best': best_params['learning_rate']},
        'num_leaves':         {'optuna': ('suggest_int', 5, 150),                 'gridsearch': [20, 40, 60],                  'default': 15,      'best': best_params['num_leaves']},
        'feature_fraction':   {'optuna': ('suggest_categorical', [1.0]),          'gridsearch': [0.7, 0.8, 0.9],               'default': 1.0,     'best': best_params['feature_fraction']},
        'min_gain_to_split':  {'optuna': ('suggest_float', 0.02, 0.02, False),    'gridsearch': [0.0001, 0.001, 0.01],         'default': 0.02,    'best': best_params['min_gain_to_split']},
        'min_data_in_leaf':   {'optuna': ('suggest_int', 50, 200),                'gridsearch': [40, 60, 80],                  'default': 60,      'best': best_params['min_data_in_leaf']},
        'lambda_l1':          {'optuna': ('suggest_float', 0, 0, False),          'gridsearch': [0.001, 0.01],                 'default': 0,       'best': best_params['lambda_l1']},
        'lambda_l2':          {'optuna': ('suggest_float', 1e-5, 10, True),       'gridsearch': [0.001, 0.01],                 'default': 0.01,    'best': best_params['lambda_l2']},
        'bagging_fraction':   {'optuna': ('suggest_float', 1.0, 1.0, True),       'gridsearch': [0.9, 1],                      'default': 1,       'best': best_params['bagging_fraction']},
        'bagging_freq':       {'optuna': ('suggest_int', 0, 0),                   'gridsearch': [0, 20],                       'default': 0,       'best': best_params['bagging_freq']},
    }

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
    format_end = date.today().strftime('%Y%m%d')
    model_name = f'lightgbm_{format_end}'

    alpha = AlphaModel(live=live, model_name=model_name, end=current_date, tuning=tune, shap=False, plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, trend=0, incr=True, opt='wfo',
                       weight=False, outlier=False, early=True, pretrain_len=1260, train_len=504, valid_len=63, test_len=21, **lightgbm_params)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
    ret = PrepFactor(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ret)
    del ret

    ret_comp = PrepFactor(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ret_comp)
    del ret_comp

    cycle = PrepFactor(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(cycle, categorical=True)
    del cycle

    talib = PrepFactor(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(talib)
    del talib

    volume = PrepFactor(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(volume)
    del volume

    volatility = PrepFactor(live=live, factor_name='factor_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(volatility)
    del volatility

    sign_ret = PrepFactor(live=live, factor_name='factor_sign_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(sign_ret, categorical=True)
    del sign_ret

    vol_comp = PrepFactor(live=live, factor_name='factor_vol_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(vol_comp)
    del vol_comp

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
    load_ret = PrepFactor(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(load_ret)
    del load_ret

    load_volume = PrepFactor(live=live, factor_name='factor_load_volume', group='permno', interval='D', kind='loading', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(load_volume)
    del load_volume

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------CONDITION-----------------------------------------------------------------------------------------
    cond_ret = PrepFactor(live=live, factor_name='factor_cond_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(cond_ret, categorical=True)
    del cond_ret

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
    ind = PrepFactor(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ind, categorical=True)
    del ind

    ind_fama = PrepFactor(live=live, factor_name='factor_ind_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ind_fama, categorical=True)
    del ind_fama

    ind_sub = PrepFactor(live=live, factor_name='factor_ind_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ind_sub, categorical=True)
    del ind_sub

    ind_mom = PrepFactor(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ind_mom)
    del ind_mom

    ind_mom_fama = PrepFactor(live=live, factor_name='factor_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ind_mom_fama)
    del ind_mom_fama

    ind_mom_sub = PrepFactor(live=live, factor_name='factor_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ind_mom_sub)
    del ind_mom_sub

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------OPEN ASSET----------------------------------------------------------------------------------------
    age_mom = PrepFactor(live=live, factor_name='factor_age_mom', group='permno', interval='D', kind='age', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(age_mom)
    del age_mom

    net_debt_finance = PrepFactor(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(net_debt_finance)
    del net_debt_finance

    chtax = PrepFactor(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(chtax)
    del chtax

    asset_growth = PrepFactor(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(asset_growth)
    del asset_growth

    mom_season_short = PrepFactor(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_season_short)
    del mom_season_short

    mom_season = PrepFactor(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_season)
    del mom_season

    noa = PrepFactor(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(noa)
    del noa

    invest_ppe = PrepFactor(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(invest_ppe)
    del invest_ppe

    inv_growth = PrepFactor(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(inv_growth)
    del inv_growth

    trend_factor = PrepFactor(live=live, factor_name='factor_trend_factor', group='permno', interval='D', kind='trend', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(trend_factor)
    del trend_factor

    mom_season6 = PrepFactor(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_season6)
    del mom_season6

    mom_season11 = PrepFactor(live=live, factor_name='factor_mom_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_season11)
    del mom_season11

    mom_season16 = PrepFactor(live=live, factor_name='factor_mom_season16', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_season16)
    del mom_season16

    comp_debt = PrepFactor(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(comp_debt)
    del comp_debt

    mom_vol = PrepFactor(live=live, factor_name='factor_mom_vol', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_vol, categorical=True)
    del mom_vol

    int_mom = PrepFactor(live=live, factor_name='factor_int_mom', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(int_mom)
    del int_mom

    cheq = PrepFactor(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(cheq)
    del cheq

    xfin = PrepFactor(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(xfin)
    del xfin

    emmult = PrepFactor(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(emmult)
    del emmult

    accrual = PrepFactor(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(accrual)
    del accrual

    frontier = PrepFactor(live=live, factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(frontier)
    del frontier

    mom_rev = PrepFactor(live=live, factor_name='factor_mom_rev', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_rev, categorical=True)
    del mom_rev

    hire = PrepFactor(live=live, factor_name='factor_hire', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(hire)
    del hire

    rds = PrepFactor(live=live, factor_name='factor_rds', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(rds)
    del rds

    pcttoacc = PrepFactor(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(pcttoacc)
    del pcttoacc

    accrual_bm = PrepFactor(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(accrual_bm)
    del accrual_bm

    mom_off_season = PrepFactor(live=live, factor_name='factor_mom_off_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(mom_off_season)
    del mom_off_season

    grcapx = PrepFactor(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(grcapx)
    del grcapx

    earning_streak = PrepFactor(live=live, factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(earning_streak)
    del earning_streak

    ret_skew = PrepFactor(live=live, factor_name='factor_ret_skew', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(ret_skew)
    del ret_skew

    dividend = PrepFactor(live=live, factor_name='factor_dividend', group='permno', interval='D', kind='dividend', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(dividend, categorical=True)
    del dividend

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------BETAS---------------------------------------------------------------------------------------------
    sb_pca = PrepFactor(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(sb_pca)
    del sb_pca

    sb_sector = PrepFactor(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(sb_sector)
    del sb_sector

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
    clust_ret = PrepFactor(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(clust_ret, categorical=True)
    del clust_ret

    clust_load_ret = PrepFactor(live=live, factor_name='factor_clust_load_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(clust_load_ret, categorical=True)
    del clust_load_ret

    clust_ind_mom = PrepFactor(live=live, factor_name='factor_clust_ind_mom', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(clust_ind_mom, categorical=True)
    del clust_ind_mom

    clust_ind_mom_fama = PrepFactor(live=live, factor_name='factor_clust_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
    alpha.add_factor(clust_ind_mom_fama, categorical=True)
    del clust_ind_mom_fama

    clust_ind_mom_sub = PrepFactor(live=live, factor_name='factor_clust_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=start_model, end=current_date, save=save_prep).prep()
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


# exec_model(threshold=6_000_000_000, update_price=False, start_data='2004-01-01', start_factor='2004-01-01', start_model='2008-01-01', tune='best', save_prep=True)
