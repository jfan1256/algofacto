import shutil
import quantstats as qs

from scipy.stats import spearmanr

from class_model.model_lightgbm import ModelLightgbm
from class_model.model_test import ModelTest
from class_model.model_prep import ModelPrep
from class_strat.strat import Strategy
from core.operation import *

class StratMLRetGBM(Strategy):
    def __init__(self,
                 allocate=None,
                 current_date=None,
                 start_model=None,
                 threshold=None,
                 num_stocks=None,
                 leverage=None,
                 port_opt=None,
                 use_top=None):

        '''
        allocate (float): Percentage of capital to allocate for this strategy
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for model training)
        start_model (str: YYYY-MM-DD): Start date for model training
        threshold (int): Market cap threshold to determine whether a stock is buyable/shortable or not
        num_stocks (int): Number of stocks to long/short
        leverage (int): Leverage value for long/short (i.e., 0.5 means 0.5 * long + 0.5 short)
        port_opt (str): Type of portfolio optimization to use
        use_top (int): Number of models to use for ensemble prediction
        '''

        super().__init__(allocate, current_date, threshold)
        self.allocate = allocate
        self.current_date =current_date
        self.start_model = start_model
        self.threshold = threshold
        self.num_stocks = num_stocks
        self.leverage = leverage
        self.port_opt = port_opt
        self.use_top = use_top

        with open(get_config() / 'api_key.json') as f:
            config = json.load(f)
            fred_key = config['fred_key']
        self.fred_key = fred_key
    
    def exec_backtest(self):
        print("-------------------------------------------------------------------EXEC ML RET MODEL--------------------------------------------------------------------------------------")
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
        live = True
        normalize = 'rank_normalize'
        total_time = time.time()

        stock = read_stock(get_large(live) / 'permno_live.csv')

        start_time = time.time()

        best_params = {
            'max_depth':          [4,             4,                4            ],
            'learning_rate':      [0.1989804,     0.2017221,        0.1000341    ],
            'num_leaves':         [61,            83,               29           ],
            'feature_fraction':   [1,             1,                1            ],
            'min_gain_to_split':  [0.02,          0.02,             0.02         ],
            'min_data_in_leaf':   [90,            77,               102          ],
            'lambda_l1':          [0,             0,                0            ],
            'lambda_l2':          [0.0013003,     0.0017645,        0.0001043    ],
            'bagging_fraction':   [1,             1,                1            ],
            'bagging_freq':       [0,             0,                0            ]
        }

        best_params = {
            'max_depth': [4],
            'learning_rate': [0.15],
            'num_leaves': [124],
            'feature_fraction': [1],
            'min_gain_to_split': [0.02],
            'min_data_in_leaf': [167],
            'lambda_l1': [0],
            'lambda_l2': [0.011],
            'bagging_fraction': [1],
            'bagging_freq': [0]
        }

        lightgbm_params = {
            'max_depth':           {'optuna': ('suggest_categorical', [4]),               'gridsearch': [4],                                                                                                      'default': 6,        'best': best_params['max_depth']},
            'learning_rate':       {'optuna': ('suggest_float', 0.10, 0.50, False),       'gridsearch': [round(i / 100, 2) for i in range(10, 31)],                                                               'default': 0.15,     'best': best_params['learning_rate']},
            'num_leaves':          {'optuna': ('suggest_int', 5, 150),                    'gridsearch': list(range(5, 151)),                                                                                      'default': 15,       'best': best_params['num_leaves']},
            'feature_fraction':    {'optuna': ('suggest_categorical', [1.0]),             'gridsearch': [1],                                                                                                      'default': 1.0,      'best': best_params['feature_fraction']},
            'min_gain_to_split':   {'optuna': ('suggest_float', 0.02, 0.02, False),       'gridsearch': [0.02],                                                                                                   'default': 0.02,     'best': best_params['min_gain_to_split']},
            'min_data_in_leaf':    {'optuna': ('suggest_int', 50, 200),                   'gridsearch': list(range(50, 201)),                                                                                     'default': 60,       'best': best_params['min_data_in_leaf']},
            'lambda_l1':           {'optuna': ('suggest_float', 0, 0, False),             'gridsearch': [0],                                                                                                      'default': 0,        'best': best_params['lambda_l1']},
            'lambda_l2':           {'optuna': ('suggest_float', 1e-5, 10, True),          'gridsearch': sorted(set([custom_round(10**x, 2) for x in [i * 0.001 for i in range(-5000, 1001)]] + [10])),     'default': 0.01,     'best': best_params['lambda_l2']},
            'bagging_fraction':    {'optuna': ('suggest_float', 1.0, 1.0, True),          'gridsearch': [1],                                                                                                      'default': 1,        'best': best_params['bagging_fraction']},
            'bagging_freq':        {'optuna': ('suggest_int', 0, 0),                      'gridsearch': [0],                                                                                                      'default': 0,        'best': best_params['bagging_freq']},
        }

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
        format_end = date.today().strftime('%Y%m%d')
        model_name = f'lightgbm_{format_end}'
        tune = ['gridsearch', 50]

        alpha = ModelLightgbm(live=live, model_name=model_name, tuning=tune, shap=False, plot_loss=False, plot_hist=False, pred='price', stock='permno', lookahead=1, trend=0,
                              incr=True, opt='wfo', outlier=False, early=True, pretrain_len=1260, train_len=504, valid_len=63, test_len=21, **lightgbm_params)

        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
        # ret = ModelPrep(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(ret)
        # del ret
        #
        # ret_comp = ModelPrep(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(ret_comp, normalize=normalize)
        # del ret_comp
        #
        # cycle = ModelPrep(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(cycle, categorical=True)
        # del cycle
        #
        # talib = ModelPrep(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(talib, normalize=normalize)
        # del talib
        #
        # volume = ModelPrep(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(volume, normalize=normalize)
        # del volume
        #
        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
        # load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(load_ret, normalize=normalize)
        # del load_ret
        #
        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
        # ind = ModelPrep(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(ind, categorical=True)
        # del ind
        #
        # ind_mom = ModelPrep(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(ind_mom, normalize=normalize)
        # del ind_mom
        #
        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # -------------------------------------------------------------------------OPEN ASSET MONTHLY------------------------------------------------------------------------------------
        # net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # net_debt_finance = net_debt_finance.groupby('permno').shift(84)
        # alpha.add_factor(net_debt_finance, normalize=normalize)
        # del net_debt_finance
        #
        # chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # chtax = chtax.groupby('permno').shift(84)
        # alpha.add_factor(chtax, normalize=normalize)
        # del chtax
        #
        # asset_growth = ModelPrep(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # asset_growth = asset_growth.groupby('permno').shift(84)
        # alpha.add_factor(asset_growth, normalize=normalize)
        # del asset_growth
        #
        # noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # noa = noa.groupby('permno').shift(84)
        # alpha.add_factor(noa, normalize=normalize)
        # del noa
        #
        # invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # invest_ppe = invest_ppe.groupby('permno').shift(84)
        # alpha.add_factor(invest_ppe, normalize=normalize)
        # del invest_ppe
        #
        # inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # inv_growth = inv_growth.groupby('permno').shift(84)
        # alpha.add_factor(inv_growth, normalize=normalize)
        # del inv_growth
        #
        # comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # comp_debt = comp_debt.groupby('permno').shift(84)
        # alpha.add_factor(comp_debt, normalize=normalize)
        # del comp_debt
        #
        # cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # cheq = cheq.groupby('permno').shift(84)
        # alpha.add_factor(cheq, normalize=normalize)
        # del cheq
        #
        # xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # xfin = xfin.groupby('permno').shift(84)
        # alpha.add_factor(xfin, normalize=normalize)
        # del xfin
        #
        # emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # emmult = emmult.groupby('permno').shift(84)
        # alpha.add_factor(emmult, normalize=normalize)
        # del emmult
        #
        # accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # accrual = accrual.groupby('permno').shift(84)
        # alpha.add_factor(accrual, normalize=normalize)
        # del accrual
        #
        # frontier = ModelPrep(live=live, factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # frontier = frontier.groupby('permno').shift(84)
        # alpha.add_factor(frontier, normalize=normalize)
        # del frontier
        #
        # hire = ModelPrep(live=live, factor_name='factor_hire', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # hire = hire.groupby('permno').shift(84)
        # alpha.add_factor(hire, normalize=normalize)
        # del hire
        #
        # rds = ModelPrep(live=live, factor_name='factor_rds', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # rds = rds.groupby('permno').shift(84)
        # alpha.add_factor(rds, normalize=normalize)
        # del rds
        #
        # pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # pcttoacc = pcttoacc.groupby('permno').shift(84)
        # alpha.add_factor(pcttoacc, normalize=normalize)
        # del pcttoacc
        #
        # accrual_bm = ModelPrep(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # accrual_bm = accrual_bm.groupby('permno').shift(84)
        # alpha.add_factor(accrual_bm, normalize=normalize)
        # del accrual_bm
        #
        # earning_streak = ModelPrep(live=live, factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # earning_streak = earning_streak.groupby('permno').shift(84)
        # alpha.add_factor(earning_streak, normalize=normalize)
        # del earning_streak
        #
        # ms = ModelPrep(live=live, factor_name='factor_ms', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # ms = ms.groupby('permno').shift(84)
        # alpha.add_factor(ms, categorical=True)
        # del ms
        #
        # grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # grcapx = grcapx.groupby('permno').shift(84)
        # alpha.add_factor(grcapx, normalize=normalize)
        # del grcapx
        #
        # gradexp = ModelPrep(live=live, factor_name='factor_gradexp', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # gradexp = gradexp.groupby('permno').shift(84)
        # alpha.add_factor(gradexp, normalize=normalize)
        # del gradexp
        #
        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # -------------------------------------------------------------------------OPEN ASSET DAILY--------------------------------------------------------------------------------------
        # mom_rev = ModelPrep(live=live, factor_name='factor_mom_rev', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_rev, categorical=True)
        # del mom_rev
        #
        # mom_season16 = ModelPrep(live=live, factor_name='factor_mom_season16', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_season16, normalize=normalize)
        # del mom_season16
        #
        # mom_off_season = ModelPrep(live=live, factor_name='factor_mom_off_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_off_season, normalize=normalize)
        # del mom_off_season
        #
        # dividend = ModelPrep(live=live, factor_name='factor_dividend', group='permno', interval='D', kind='dividend', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(dividend, categorical=True)
        # del dividend
        #
        # ret_skew = ModelPrep(live=live, factor_name='factor_ret_skew', group='permno', interval='D', kind='skew', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(ret_skew, normalize=normalize)
        # del ret_skew
        #
        # size = ModelPrep(live=live, factor_name='factor_size', group='permno', interval='D', kind='size', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(size, normalize=normalize)
        # del size
        #
        # mom_off_season6 = ModelPrep(live=live, factor_name='factor_mom_off_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_off_season6, normalize=normalize)
        # del mom_off_season6
        #
        # mom_off_season11 = ModelPrep(live=live, factor_name='factor_mom_off_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_off_season11, normalize=normalize)
        # del mom_off_season11
        #
        # age_mom = ModelPrep(live=live, factor_name='factor_age_mom', group='permno', interval='D', kind='age', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(age_mom, normalize=normalize)
        # del age_mom
        #
        # mom_season_short = ModelPrep(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_season_short, normalize=normalize)
        # del mom_season_short
        #
        # mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_season, normalize=normalize)
        # del mom_season
        #
        # trend_factor = ModelPrep(live=live, factor_name='factor_trend_factor', group='permno', interval='D', kind='trend', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(trend_factor, normalize=normalize)
        # del trend_factor
        #
        # mom_season6 = ModelPrep(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_season6, normalize=normalize)
        # del mom_season6
        #
        # mom_season11 = ModelPrep(live=live, factor_name='factor_mom_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_season11, normalize=normalize)
        # del mom_season11
        #
        # mom_vol = ModelPrep(live=live, factor_name='factor_mom_vol', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_vol, categorical=True)
        # del mom_vol
        #
        # int_mom = ModelPrep(live=live, factor_name='factor_int_mom', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(int_mom, normalize=normalize)
        # del int_mom
        #
        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
        # clust_ret = ModelPrep(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(clust_ret, categorical=True)
        # del clust_ret

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
        ret = ModelPrep(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret)
        del ret

        ret_comp = ModelPrep(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret_comp, normalize=normalize)
        del ret_comp

        cycle = ModelPrep(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(cycle, categorical=True)
        del cycle

        talib = ModelPrep(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(talib, normalize=normalize)
        del talib

        volume = ModelPrep(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(volume, normalize=normalize)
        del volume

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
        load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(load_ret, normalize=normalize)
        del load_ret

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
        ind = ModelPrep(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind, categorical=True)
        del ind

        ind_mom = ModelPrep(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_mom, normalize=normalize)
        del ind_mom

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
        clust_ret = ModelPrep(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_ret, categorical=True)
        del clust_ret

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------TRAINING------------------------------------------------------------------------------------------
        elapsed_time = time.time() - start_time
        print("-" * 60)
        print(f"Total time to prep and add all factors: {round(elapsed_time)} seconds")
        print(f"AlphaModel Dataframe Shape: {alpha.data.shape}")
        print("-" * 60)
        print("Run Model")

        alpha.exec_train()

        elapsed_time = time.time() - total_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Total time to execute everything: {int(minutes)}:{int(seconds):02}")
        print("-" * 60)

    def exec_live(self):
        print("--------------------------------------------------------------------------EXEC ML RET PRED--------------------------------------------------------------------------------")
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------PARAMS------------------------------------------------------------------------------------
        live = True
        model_name = f"lightgbm_{date.today().strftime('%Y%m%d')}"
        dir_path = Path(get_ml_report(live, model_name) / model_name)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------INITIATE LIVE TEST------------------------------------------------------------------------------
        print("--------------------------------------------------------------------------INITIATE LIVE TEST------------------------------------------------------------------------------")
        live_test = ModelTest(live=live, num_stocks=self.num_stocks, leverage=self.leverage, port_opt=self.port_opt, model_name=model_name, current_date=self.current_date, dir_path=dir_path)
        files = live_test.read_result('metrics')

        # Create directory for backtest report
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------CALCULATE SHARPE PER TRIAL--------------------------------------------------------------------------
        print("----------------------------------------------------------------------CALCULATE SHARPE PER TRIAL--------------------------------------------------------------------------")
        # Dictionary to keep track of SHARPE
        keep = {}
        ticker = pd.read_parquet(get_parquet(live) / 'data_ticker.parquet.brotli')
        misc = pd.read_parquet(get_parquet(live) / 'data_misc.parquet.brotli', columns=['market_cap'])

        # Iterate through each trial
        for i, row in files.iterrows():
            # Read file in
            read_file = live_test.get_max_metric_file(row)
            # Execute ranking of stocks
            returns = live_test.sharpe_ret(read_file, iteration=False)
            # Convert Permno to Ticker
            tic = returns.merge(ticker, left_index=True, right_index=True, how='left')
            tic = tic.merge(misc, left_index=True, right_index=True, how='left')
            tic['market_cap'] = tic.groupby('permno')['market_cap'].ffill()
            tic = tic.reset_index().set_index(['window', 'ticker', 'date'])
            tic = tic.drop('permno', axis=1)
            # Calculate SHARPE with EWP
            pred = live_test.sharpe_backtest(tic, self.threshold)
            equal_weight, long_weight, short_weight = live_test.exec_port_opt(data=pred)
            strat_ret = equal_weight['totalRet']
            sharpe = qs.stats.sharpe(strat_ret)
            calmar = qs.stats.calmar(strat_ret)
            maxdd = qs.stats.max_drawdown(strat_ret)
            # Display metrics
            print('-' * 60)
            print(f'Row: {i}')
            metrics = read_file['metrics']
            column_widths = [max(len(str(val)) for val in metrics[col]) for col in metrics.columns]
            header_widths = [len(header) for header in metrics.columns]
            max_widths = [max(col_width, header_width) for col_width, header_width in zip(column_widths, header_widths)]
            headers = " | ".join([header.ljust(width) for header, width in zip(metrics.columns, max_widths)])
            values = " | ".join([str(val).ljust(width) for val, width in zip(metrics.iloc[0], max_widths)])
            formatted_metrics = headers + "\n" + values
            print(formatted_metrics)
            print(f'SHARPE Ratio: {sharpe}')
            print(f'CALMAR Ratio: {calmar}')
            print(f'MAX DD: {maxdd*100}%')
            # Save SHARPE to dictionary
            keep[i] = np.mean(sharpe + calmar)
            # keep[i] = sharpe
            # keep[i] = calmar

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------PERFORM ENSEMBLE PREDICTION AND SAVE METRIC/IC---------------------------------------------------------------
        print("-------------------------------------------------------------PERFORM ENSEMBLE PREDICTION AND SAVE METRIC/IC---------------------------------------------------------------")
        # Retrieves the indices from the top 5 best performing SHARPE
        max_sharpe_idxs = sorted(keep, key=keep.get, reverse=True)[:self.use_top]
        collect = []
        # Append the individual trial predictions to a dataframe
        for idx in max_sharpe_idxs:
            print(f'Best Sharpe Idx: {idx}')
            best_model_params = live_test.get_max_metric_file(files.iloc[idx])
            merged = live_test.price(best_model_params, dir_path, iteration=False, plot=False)
            collect.append(merged['predictions'])

        # Concat and calculate the mean of the predictions
        total = pd.concat(collect, axis=1)
        total['mean_predictions'] = total.mean(axis=1)
        merged['predictions'] = total['mean_predictions']
        # Calculate IC
        by_day = merged.groupby(level='date')
        ic_by_day = by_day.apply(lambda x: spearmanr(x.predictions, x.returns)[0]).to_frame('combined')
        # Save Plot
        live_test.plot_ensemble(merged, ic_by_day)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------CONVERT FROM PERMNO TO TICKER/EXCHANGE-------------------------------------------------------------------
        print("-----------------------------------------------------------------CONVERT FROM PERMNO TO TICKER/EXCHANGE-------------------------------------------------------------------")
        tic = merged.merge(ticker, left_index=True, right_index=True, how='left')
        tic = tic.merge(misc, left_index=True, right_index=True, how='left')
        tic['market_cap'] = tic.groupby('permno')['market_cap'].ffill()
        tic = tic.reset_index().set_index(['window', 'ticker', 'date'])
        exchange = pd.read_parquet(get_parquet(live) / 'data_exchange.parquet.brotli')
        tic_reset = tic.reset_index()
        exchange_df_reset = exchange.reset_index()
        combined = pd.merge(tic_reset, exchange_df_reset, on=['ticker', 'permno'], how='left')
        combined = combined.set_index(['window', 'ticker', 'date'])

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------EXECUTE LIVETEST------------------------------------------------------------------------------
        print("----------------------------------------------------------------------------EXECUTE LIVETEST------------------------------------------------------------------------------")
        # Create the desired dataframe with structure longRet, longStocks, shortRet, shortStocks
        pred_return = live_test.backtest(combined, self.threshold)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------RETRIEVE LONG/SHORT----------------------------------------------------------------------------
        print("---------------------------------------------------------------------------RETRIEVE LONG/SHORT----------------------------------------------------------------------------")
        data = pred_return.copy(deep=True)
        pred_return_opt, long_weights, short_weights = live_test.exec_port_opt(data=data)
        strat_ret = pred_return_opt['totalRet']

        # Save plot to "report" directory
        spy = get_spy(start_date='2005-01-01', end_date=self.current_date)
        qs.reports.html(strat_ret, spy, output=dir_path / 'report.html')

        # Retrieve stocks to long/short tomorrow (only get 'ticker')
        long = [stock_pair[0] for stock_pair in pred_return.iloc[-1]['longStocks']]
        short = [stock_pair[0] for stock_pair in pred_return.iloc[-1]['shortStocks']]

        # Retrieve weights for long/short and multiply by self.allocate
        long_weight = (long_weights[-1] * self.allocate).tolist()
        short_weight = (short_weight[-1] * self.allocate).tolist()

        # Long Stock Dataframe
        long_df = pd.DataFrame({
            'date': [self.current_date] * len(long),
            'ticker': long,
            'weight': long_weight,
            'type': 'long'
        })

        # Short Stock Dataframe
        short_df = pd.DataFrame({
            'date': [self.current_date] * len(short),
            'ticker': short,
            'weight': short_weight,
            'type': 'short'
        })

        # Combine long and short dataframes
        combined_df = pd.concat([long_df, short_df], axis=0)
        combined_df = combined_df.set_index(['date', 'ticker', 'type']).sort_index(level=['date', 'ticker', 'type'])
        filename = get_live_stock() / 'trade_stock_ml_ret_gbm.parquet.brotli'
        combined_df.to_parquet(filename, compression='brotli')