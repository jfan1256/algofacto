import shutil
import json
import quantstats as qs

from fredapi import Fred
from scipy.stats import spearmanr
from class_model.model_test import ModelTest
from class_model.model_prep import ModelPrep
from class_model.model_train import ModelTrain
from class_strat.strat import Strategy
from class_trend.trend_helper import TrendHelper
from core.operation import *

class StratMLTrend(Strategy):
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
        print("-----------------------------------------------------------------EXEC ML TREND MODEL--------------------------------------------------------------------------------------")
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------PARAMS--------------------------------------------------------------------------------------------
        live = True
        total_time = time.time()

        stock = read_stock(get_large(live) / 'permno_live.csv')

        start_time = time.time()

        randomforest_params = {
            'n_estimators': {'optuna': ('suggest_int', 50, 1000),           'gridsearch': [100, 300, 500, 800],           'default': 50},
            'max_depth': {'optuna': ('suggest_int', 4, 6),                  'gridsearch': [4, 6, 8, 12, 16, 20],          'default': 6},
            'min_samples_split': {'optuna': ('suggest_int', 2, 10),          'gridsearch': [2, 4, 6, 8],                   'default': 2},
            'min_samples_leaf': {'optuna': ('suggest_int', 1, 4),            'gridsearch': [1, 2, 3],                      'default': 1}
        }

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
        format_end = date.today().strftime('%Y%m%d')
        model_name = f'randomforest_{format_end}'
        tune = 'default'

        alpha = ModelTrain(live=live, model_name=model_name, end=self.current_date, tuning=tune, shap=False, plot_loss=False, plot_hist=False, pred='sign',
                           stock='permno', lookahead=1, trend=1, incr=False, opt='ewo', weight=False, outlier=False, early=False,
                           pretrain_len=0, train_len=504, valid_len=21, test_len=21, **randomforest_params)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
        ret = ModelPrep(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret)
        del ret

        ret_comp = ModelPrep(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret_comp)
        del ret_comp

        cycle = ModelPrep(live=live, factor_name='factor_time', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(cycle, categorical=True)
        del cycle

        talib = ModelPrep(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(talib)
        del talib

        volume = ModelPrep(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(volume)
        del volume

        volatility = ModelPrep(live=live, factor_name='factor_volatility', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(volatility)
        del volatility

        sign_ret = ModelPrep(live=live, factor_name='factor_sign_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(sign_ret, categorical=True)
        del sign_ret

        vol_comp = ModelPrep(live=live, factor_name='factor_vol_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(vol_comp)
        del vol_comp

        sign_volume = ModelPrep(live=live, factor_name='factor_sign_volume', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(sign_volume, categorical=True)
        del sign_volume

        sign_volatility = ModelPrep(live=live, factor_name='factor_sign_volatility', group='permno', interval='D', kind='sign', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(sign_volatility, categorical=True)
        del sign_volatility

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
        load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(load_ret)
        del load_ret

        load_volume = ModelPrep(live=live, factor_name='factor_load_volume', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(load_volume)
        del load_volume

        load_volatility = ModelPrep(live=live, factor_name='factor_load_volatility', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(load_volatility)
        del load_volatility

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------CONDITION-----------------------------------------------------------------------------------------
        cond_ret = ModelPrep(live=live, factor_name='factor_cond_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(cond_ret, categorical=True)
        del cond_ret

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
        ind = ModelPrep(live=live, factor_name='factor_ind', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind, categorical=True)
        del ind

        ind_fama = ModelPrep(live=live, factor_name='factor_ind_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_fama, categorical=True)
        del ind_fama

        ind_sub = ModelPrep(live=live, factor_name='factor_ind_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_sub, categorical=True)
        del ind_sub

        ind_mom = ModelPrep(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_mom)
        del ind_mom

        ind_mom_fama = ModelPrep(live=live, factor_name='factor_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_mom_fama)
        del ind_mom_fama

        ind_mom_sub = ModelPrep(live=live, factor_name='factor_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_mom_sub)
        del ind_mom_sub

        ind_vwr = ModelPrep(live=live, factor_name='factor_ind_vwr', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_vwr)
        del ind_vwr

        ind_vwr_fama = ModelPrep(live=live, factor_name='factor_ind_vwr_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_vwr_fama)
        del ind_vwr_fama

        ind_vwr_sub = ModelPrep(live=live, factor_name='factor_ind_vwr_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_vwr_sub)
        del ind_vwr_sub

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------OPEN ASSET----------------------------------------------------------------------------------------
        age_mom = ModelPrep(live=live, factor_name='factor_age_mom', group='permno', interval='D', kind='age', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(age_mom)
        del age_mom

        net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date,
                                     save=True).prep()
        alpha.add_factor(net_debt_finance)
        del net_debt_finance

        chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(chtax)
        del chtax

        asset_growth = ModelPrep(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(asset_growth)
        del asset_growth

        mom_season_short = ModelPrep(live=live, factor_name='factor_mom_season_short', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_season_short)
        del mom_season_short

        mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_season)
        del mom_season

        noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(noa)
        del noa

        invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(invest_ppe)
        del invest_ppe

        inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(inv_growth)
        del inv_growth

        trend_factor = ModelPrep(live=live, factor_name='factor_trend_factor', group='permno', interval='D', kind='trend', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(trend_factor)
        del trend_factor

        mom_season6 = ModelPrep(live=live, factor_name='factor_mom_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_season6)
        del mom_season6

        mom_season11 = ModelPrep(live=live, factor_name='factor_mom_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_season11)
        del mom_season11

        mom_season16 = ModelPrep(live=live, factor_name='factor_mom_season16', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_season16)
        del mom_season16

        comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(comp_debt)
        del comp_debt

        mom_vol = ModelPrep(live=live, factor_name='factor_mom_vol', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_vol, categorical=True)
        del mom_vol

        int_mom = ModelPrep(live=live, factor_name='factor_int_mom', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(int_mom)
        del int_mom

        cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(cheq)
        del cheq

        xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(xfin)
        del xfin

        emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(emmult)
        del emmult

        accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(accrual)
        del accrual

        frontier = ModelPrep(live=live, factor_name='factor_frontier', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(frontier)
        del frontier

        mom_rev = ModelPrep(live=live, factor_name='factor_mom_rev', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_rev, categorical=True)
        del mom_rev

        hire = ModelPrep(live=live, factor_name='factor_hire', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(hire)
        del hire

        rds = ModelPrep(live=live, factor_name='factor_rds', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(rds)
        del rds

        pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(pcttoacc)
        del pcttoacc

        accrual_bm = ModelPrep(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(accrual_bm)
        del accrual_bm

        mom_off_season = ModelPrep(live=live, factor_name='factor_mom_off_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_off_season)
        del mom_off_season

        earning_streak = ModelPrep(live=live, factor_name='factor_earning_streak', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(earning_streak)
        del earning_streak

        ms = ModelPrep(live=live, factor_name='factor_ms', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ms, categorical=True)
        del ms

        dividend = ModelPrep(live=live, factor_name='factor_dividend', group='permno', interval='D', kind='dividend', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(dividend, categorical=True)
        del dividend

        grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(grcapx)
        del grcapx

        gradexp = ModelPrep(live=live, factor_name='factor_gradexp', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(gradexp)
        del gradexp

        ret_skew = ModelPrep(live=live, factor_name='factor_ret_skew', group='permno', interval='D', kind='skew', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret_skew)
        del ret_skew

        size = ModelPrep(live=live, factor_name='factor_size', group='permno', interval='D', kind='size', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(size)
        del size

        ret_max = ModelPrep(live=live, factor_name='factor_ret_max', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret_max)
        del ret_max

        mom_off_season6 = ModelPrep(live=live, factor_name='factor_mom_off_season6', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_off_season6)
        del mom_off_season6

        mom_off_season11 = ModelPrep(live=live, factor_name='factor_mom_off_season11', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(mom_off_season11)
        del mom_off_season11

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------BETAS---------------------------------------------------------------------------------------------
        sb_pca = ModelPrep(live=live, factor_name='factor_sb_pca', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(sb_pca)
        del sb_pca

        sb_sector = ModelPrep(live=live, factor_name='factor_sb_sector', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(sb_sector)
        del sb_sector

        sb_inverse = ModelPrep(live=live, factor_name='factor_sb_inverse', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(sb_inverse)
        del sb_inverse

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------CLUSTER-------------------------------------------------------------------------------------------
        clust_ret = ModelPrep(live=live, factor_name='factor_clust_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_ret, categorical=True)
        del clust_ret

        clust_load_ret = ModelPrep(live=live, factor_name='factor_clust_load_ret', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_load_ret, categorical=True)
        del clust_load_ret

        clust_ind_mom = ModelPrep(live=live, factor_name='factor_clust_ind_mom', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_ind_mom, categorical=True)
        del clust_ind_mom

        clust_ind_mom_fama = ModelPrep(live=live, factor_name='factor_clust_ind_mom_fama', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_ind_mom_fama, categorical=True)
        del clust_ind_mom_fama

        clust_ind_mom_sub = ModelPrep(live=live, factor_name='factor_clust_ind_mom_sub', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_ind_mom_sub, categorical=True)
        del clust_ind_mom_sub

        clust_load_volume = ModelPrep(live=live, factor_name='factor_clust_load_volume', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_load_volume, categorical=True)
        del clust_load_volume

        clust_volatility = ModelPrep(live=live, factor_name='factor_clust_volatility', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_volatility, categorical=True)
        del clust_volatility

        clust_volume = ModelPrep(live=live, factor_name='factor_clust_volume', group='permno', interval='D', kind='cluster', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(clust_volume, categorical=True)
        del clust_volume

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------TRAINING------------------------------------------------------------------------------------------
        elapsed_time = time.time() - start_time
        print("-" * 60)
        print(f"Total time to prep and add all factors: {round(elapsed_time)} seconds")
        print(f"AlphaModel Dataframe Shape: {alpha.data.shape}")
        print("-" * 60)
        print("Run Model")

        alpha.randomforest()

        elapsed_time = time.time() - total_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Total time to execute everything: {int(minutes)}:{int(seconds):02}")
        print("-" * 60)

    def exec_live(self):
        print("--------------------------------------------------------------------------EXEC ML TREND PRED------------------------------------------------------------------------------")
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------PARAMS------------------------------------------------------------------------------------
        live = True
        model_name = f"randomforest_{date.today().strftime('%Y%m%d')}"
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
            # Save SHARPE to dictionary
            keep[i] = sharpe

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
        # -----------------------------------------------------------------------------------CREATE HEDGE--------------------------------------------------------------------------------
        print("------------------------------------------------------------------------------CREATE HEDGE-------------------------------------------------------------------------------")
        # Commodities
        trend_helper = TrendHelper(current_date=self.current_date, start_date=self.start_model, num_stocks=10)
        re_ticker = ['VNQ', 'IYR', 'SCHH', 'RWR', 'USRT', 'REZ']
        re = trend_helper._get_ret(re_ticker)
        # Bonds
        bond_ticker = ['HYG', 'JNK', 'LQD', 'EMB', 'SHY', 'TLT', 'SPTL', 'IGSB', 'SPAB']
        bond = trend_helper._get_ret(bond_ticker)

        # Create portfolio
        bond_re_port = pd.concat([bond, re], axis=0)
        bond_re_port['vol'] = bond_re_port.groupby('ticker')['RET_01'].transform(lambda x: x.rolling(5).std().shift(1))
        bond_re_port['inv_vol'] = 1 / bond_re_port['vol']
        bond_re_port['norm_inv_vol'] = bond_re_port.groupby('date')['inv_vol'].apply(lambda x: x / x.sum()).reset_index(level=0, drop=True)
        bond_re_port['RET_01'] = bond_re_port['RET_01'].groupby('ticker').shift(-1)
        bond_re_port['weighted_ret'] = bond_re_port['RET_01'] * bond_re_port['norm_inv_vol']
        hedge_ret = bond_re_port.groupby('date')['weighted_ret'].sum()
        hedge_ret = hedge_ret.to_frame()
        hedge_ret.columns = ['bond_comm_ret']

        # Date Index
        date_index = hedge_ret.index
        date_index = date_index.to_frame().drop('date', axis=1).reset_index()

        # 5-Year Inflation Rate
        fred = Fred(api_key=self.fred_key)
        inflation = fred.get_series("T5YIE").to_frame()
        inflation.columns = ['5YIF']
        inflation = inflation.shift(1)
        inflation = inflation.reset_index()
        inflation = pd.merge_asof(date_index, inflation, left_on='date', right_on='index', direction='backward')
        inflation = inflation.set_index('date').drop('index', axis=1)
        inflation = inflation.ffill()

        # 5-Year Market Yield
        fred = Fred(api_key=self.fred_key)
        unemploy = fred.get_series("DFII5").to_frame()
        unemploy.columns = ['DF']
        unemploy = unemploy.shift(1)
        unemploy = unemploy.reset_index()
        unemploy = pd.merge_asof(date_index, unemploy, left_on='date', right_on='index', direction='backward')
        unemploy = unemploy.set_index('date').drop('index', axis=1)
        unemploy = unemploy.ffill()

        # 10-year vs. 2-year Yield Curve
        fred = Fred(api_key=self.fred_key)
        yield_curve = fred.get_series("T10Y2Y").to_frame()
        yield_curve.columns = ['YIELD']
        yield_curve = yield_curve.shift(1)
        yield_curve = yield_curve.reset_index()
        yield_curve = pd.merge_asof(date_index, yield_curve, left_on='date', right_on='index', direction='backward')
        yield_curve = yield_curve.set_index('date').drop('index', axis=1)
        yield_curve = yield_curve.ffill()

        # Macro Trend
        macro = pd.concat([inflation, unemploy, yield_curve], axis=1)
        macro['5YIF_z'] = (macro['5YIF'] - macro['5YIF'].mean()) / macro['5YIF'].std()
        macro['DF_z'] = (macro['DF'] - macro['DF'].mean()) / macro['DF'].std()
        macro['YIELD_z'] = (macro['YIELD'] - macro['YIELD'].mean()) / macro['YIELD'].std()
        macro['mt'] = macro[['5YIF_z', 'DF_z', 'YIELD_z']].mean(axis=1)

        for t in [21, 60]:
            macro[f'mt_{t}'] = macro['mt'].rolling(t).mean()

        macro_buy = (macro['mt_21'] > macro['mt_60'])
        macro_buy_df = macro_buy.to_frame()
        macro_buy_df.columns = ['macro_buy']

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------RETRIEVE LONG/SHORT----------------------------------------------------------------------------
        print("---------------------------------------------------------------------------RETRIEVE LONG/SHORT----------------------------------------------------------------------------")
        data = pred_return.copy(deep=True)
        pred_return_opt, long_weights, short_weights = live_test.exec_port_opt(data=data)
        strat_ret = pred_return_opt['totalRet']

        # Create total portfolio
        total_ret = pd.merge(strat_ret.to_frame('ml_trend'), hedge_ret, left_index=True, right_index=True, how='left')
        total_ret = total_ret.merge(macro_buy_df, left_index=True, right_index=True, how='left')
        col1, col2 = total_ret.columns[0], total_ret.columns[1]

        total_ret['total_ret'] = total_ret.apply(trend_helper._calc_total_port, args=(col1, col2), axis=1)
        total_daily_ret = total_ret['total_ret']

        # Save plot to "report" directory
        spy = get_spy(start_date='2005-01-01', end_date=self.current_date)
        qs.reports.html(total_daily_ret, spy, output=dir_path / 'report.html')

        # Total portfolio allocation weights
        macro_buy_df = macro_buy_df.loc[macro_buy_df.index.get_level_values('date') == macro_buy_df.index.get_level_values('date').unique().max()]
        if macro_buy_df.values[0]:
            trend_factor = 0.5
            hedge_factor = 0.5
        else:
            trend_factor = 0.25
            hedge_factor = 0.75

        # Retrieve stocks to long/short tomorrow (only get 'ticker') and hedge stocks
        latest_bond_re_port = bond_re_port.loc[bond_re_port.index.get_level_values('date') == bond_re_port.index.get_level_values('date').unique().max()]
        hedge_ticker = latest_bond_re_port.index.get_level_values('ticker').unique().tolist()
        hedge_weight = (latest_bond_re_port['norm_inv_vol'] * hedge_factor * self.allocate).tolist()
        long = [stock_pair[0] for stock_pair in pred_return.iloc[-1]['longStocks']]
        short = [stock_pair[0] for stock_pair in pred_return.iloc[-1]['shortStocks']]
        # Retrieve weights for long/short and multiply by self.allocate and trend_factor for strategic asset allocation
        long_weight = (long_weights[-1] * trend_factor * self.allocate).tolist()
        short_weight = (short_weight[-1] * trend_factor * self.allocate).tolist()
        # Combine
        long = long + hedge_ticker
        long_weight = long_weight + hedge_weight

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
        filename = get_live_stock() / 'trade_stock_ml_trend.parquet.brotli'
        combined_df.to_parquet(filename, compression='brotli')