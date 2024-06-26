import shutil
import quantstats as qs

from scipy.stats import spearmanr

from class_model.model_lregression import ModelLRegression
from class_model.model_test import ModelTest
from class_model.model_prep import ModelPrep
from class_strat.strat import Strategy
from core.operation import *

class StratMLRetLR(Strategy):
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
        impute = 'cross_median'
        total_time = time.time()

        stock = read_stock(get_large(live) / 'permno_live.csv')

        start_time = time.time()

        lr_params = {
            'alpha':      {'optuna': ('suggest_float', 1e-5, 1),     'gridsearch': [1e-3, 1e-4, 1e-5],      'default': 0.01},
            'l1_ratio':   {'optuna': ('suggest_float', 1e-5, 1),     'gridsearch': [1e-3, 1e-4, 1e-5],      'default': 0.005},
        }

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------MODEL---------------------------------------------------------------------------------------------
        format_end = date.today().strftime('%Y%m%d')
        model_name = f'lregression_{format_end}'
        tune = 'default'

        alpha = ModelLRegression(live=live, model_name=model_name, tuning=tune, plot_hist=False, pred='price', model='elastic', stock='permno',
                                 lookahead=1, trend=0, opt='ewo', outlier=False, train_len=504, valid_len=21, test_len=21, **lr_params)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------GENERAL-------------------------------------------------------------------------------------------
        ret = ModelPrep(live=live, factor_name='factor_ret', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret, impute=impute)
        del ret

        ret_comp = ModelPrep(live=live, factor_name='factor_ret_comp', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ret_comp, normalize=normalize, impute=impute)
        del ret_comp

        talib = ModelPrep(live=live, factor_name='factor_talib', group='permno', interval='D', kind='price', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(talib, normalize=normalize, impute=impute)
        del talib

        volume = ModelPrep(live=live, factor_name='factor_volume', group='permno', interval='D', kind='price', div=False, stock=stock, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(volume, normalize=normalize, impute=impute)
        del volume

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------PCA-----------------------------------------------------------------------------------------------
        load_ret = ModelPrep(live=live, factor_name='factor_load_ret', group='permno', interval='D', kind='loading', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(load_ret, normalize=normalize, impute=impute)
        del load_ret

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------INDUSTRY------------------------------------------------------------------------------------------
        ind_mom = ModelPrep(live=live, factor_name='factor_ind_mom', group='permno', interval='D', kind='ind', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        alpha.add_factor(ind_mom, normalize=normalize, impute=impute)
        del ind_mom

        # # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # -----------------------------------------------------------------------------OPEN ASSET----------------------------------------------------------------------------------------
        # net_debt_finance = ModelPrep(live=live, factor_name='factor_net_debt_finance', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(net_debt_finance, normalize=normalize, impute=impute)
        # del net_debt_finance
        #
        # chtax = ModelPrep(live=live, factor_name='factor_chtax', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(chtax, normalize=normalize, impute=impute)
        # del chtax
        #
        # asset_growth = ModelPrep(live=live, factor_name='factor_asset_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(asset_growth, normalize=normalize, impute=impute)
        # del asset_growth
        #
        # mom_season = ModelPrep(live=live, factor_name='factor_mom_season', group='permno', interval='D', kind='mom', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(mom_season, normalize=normalize, impute=impute)
        # del mom_season
        #
        # noa = ModelPrep(live=live, factor_name='factor_noa', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(noa, normalize=normalize, impute=impute)
        # del noa
        #
        # invest_ppe = ModelPrep(live=live, factor_name='factor_invest_ppe_inv', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(invest_ppe, normalize=normalize, impute=impute)
        # del invest_ppe
        #
        # inv_growth = ModelPrep(live=live, factor_name='factor_inv_growth', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(inv_growth, normalize=normalize, impute=impute)
        # del inv_growth
        #
        # comp_debt = ModelPrep(live=live, factor_name='factor_comp_debt', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(comp_debt, normalize=normalize, impute=impute)
        # del comp_debt
        #
        # cheq = ModelPrep(live=live, factor_name='factor_cheq', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(cheq, normalize=normalize, impute=impute)
        # del cheq
        #
        # xfin = ModelPrep(live=live, factor_name='factor_xfin', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(xfin, normalize=normalize, impute=impute)
        # del xfin
        #
        # emmult = ModelPrep(live=live, factor_name='factor_emmult', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(emmult, normalize=normalize, impute=impute)
        # del emmult
        #
        # accrual = ModelPrep(live=live, factor_name='factor_accrual', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(accrual, normalize=normalize, impute=impute)
        # del accrual
        #
        # pcttoacc = ModelPrep(live=live, factor_name='factor_pcttotacc', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(pcttoacc, normalize=normalize, impute=impute)
        # del pcttoacc
        #
        # accrual_bm = ModelPrep(live=live, factor_name='factor_accrual_bm', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(accrual_bm, normalize=normalize, impute=impute)
        # del accrual_bm
        #
        # grcapx = ModelPrep(live=live, factor_name='factor_grcapx', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(grcapx, normalize=normalize, impute=impute)
        # del grcapx
        #
        # gradexp = ModelPrep(live=live, factor_name='factor_gradexp', group='permno', interval='M', kind='fundamental', stock=stock, div=False, start=self.start_model, end=self.current_date, save=True).prep()
        # alpha.add_factor(gradexp, normalize=normalize, impute=impute)
        # del gradexp

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
        model_name = f"lregression_{date.today().strftime('%Y%m%d')}"
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
        filename = get_live_stock() / 'trade_stock_ml_ret_lr.parquet.brotli'
        combined_df.to_parquet(filename, compression='brotli')