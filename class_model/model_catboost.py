from class_model.model_train import ModelTrain
from core.operation import *

from itertools import product
from scipy.stats import spearmanr
from datetime import timedelta
from catboost import CatBoostRegressor, CatBoostClassifier
from catboost import Pool
from sklearn.metrics import accuracy_score

import os
import time
import shap
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

class ModelCatboost(ModelTrain):
    def __init__(self,
                 live: bool = None,
                 model_name: str = None,
                 tuning: [str, int] = None,
                 shap: bool = False,
                 plot_loss: bool = False,
                 plot_hist: bool = False,
                 pred: str = 'price',
                 stock: str = None,
                 lookahead: int = 1,
                 trend: int = 0,
                 opt: str = None,
                 outlier: bool = False,
                 early: bool = True,
                 train_len: int = None,
                 valid_len: int = None,
                 test_len: int = None,
                 **kwargs
                 ):

        '''
        live (bool): Get historical data or live data
        model_name (str): Model name
        tuning (str): Type of parameter to use (i.e., default, optuna, etc.)
        shap (bool): Save shap plot or not
        plot_loss (bool): Plot training and validation curve after each window training or not
        plot_hist (bool): Plot actual returns and predicted returns after each window training or not
        pred (str): Predict for price returns or price movement
        stock (str): Name of index for stocks ('permno' or 'ticker')
        lookahead (int): Lookahead period to predict for
        trend (int): Size of rolling window to calculate trend (for price movement predictions)
        opt (str): Type of training optimization ('ewo' or 'wfo')
        weight (float): Weight for sample weight training
        outlier (bool): Handle outlier data in label data or not
        early (bool): Train with early stopping or not
        train_len (int): Train length for model training
        valid_len (int): Validation length for model training
        test_len (int): Prediction length for model training
        kwargs (dict): Model parameters to feed into model
        '''

        super().__init__(live, model_name, tuning, pred, stock, lookahead, trend, opt, outlier, None, train_len, **kwargs)
        self.data = None
        self.live = live
        self.model_name = model_name
        self.categorical = []
        self.tuning = tuning
        self.shap = shap
        self.plot_loss = plot_loss
        self.plot_hist = plot_hist
        self.pred = pred
        self.stock = stock
        self.lookahead = lookahead
        self.trend = trend
        self.opt = opt
        self.outlier = outlier
        self.early = early
        self.train_len = train_len
        self.valid_len = valid_len
        self.test_len = test_len
        self.actual_return = None
        self.parameter_specs = kwargs

        assert self.opt in ['ewo', 'wfo']
        assert self.pred == 'price' or self.pred == 'sign', ValueError('must use either "price" or "sign" for pred parameter')
        assert self.tuning[0] == 'optuna' or self.tuning[0] == 'gridsearch' or self.tuning == 'default' or self.tuning == 'best', \
            ValueError("must use either ['optuna', num_trials=int], ['gridsearch', num_search=int], or 'default' for tuning parameter")
        assert self.lookahead >= 1, ValueError('lookahead must be greater than 0')
        if self.early:
            assert self.valid_len > 0, ValueError("valid_len must be > 0 if early is set to True")

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------HELPER FUNCTIONS---------------------------------------------------------------------------------------
    # Save beeswarm SHAP plot
    def plot_beeswarm_cb(self, sv, X, key, i):
        shap.summary_plot(sv, X, max_display=80, show=False)
        plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}' / f'beeswarm_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
        plt.close()

    # Save waterfall SHAP plot
    def plot_waterfall_cb(self, sv, key, i):
        shap.plots.waterfall(sv[0], max_display=80, show=False)
        plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}' / f'waterfall_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
        plt.close()

    # Save catboost gain importance
    @staticmethod
    def get_feature_gain(model):
        fi = model.get_feature_importance(type="FeatureImportance")
        feature_names = model.feature_names_
        return pd.Series(fi / fi.sum(), index=feature_names)

    # Save catboost gain importance
    @staticmethod
    def get_feature_split(model):
        fi = model.get_feature_importance(type="PredictionValuesChange")
        feature_names = model.feature_names_
        return pd.Series(fi / fi.sum(), index=feature_names)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------EXECUTION CODE-------------------------------------------------------------------------------
    # Model Training Execution
    def catboost(self, export_key, param_name_train, param_val_train, base_params, data_train, catboost_data_train, metric_cols, num_iterations, ret):
        # Get the parameters
        params = dict(zip(param_name_train, param_val_train))
        params.update(base_params)
        # Set up cross-validation and track time
        T = 0
        track_wfo = time.time()

        # Creates the indices for wfo or ewo
        opt_pred, gain, split = [], [], []
        if self.opt == 'wfo':
            n_splits = (get_timeframe_length(data_train) - self.train_len) // self.test_len
            opt = self._wfo(data=data_train, n_splits=n_splits, lookahead=self.lookahead, train_period_length=self.train_len, test_period_length=self.test_len)
        elif self.opt == 'ewo':
            n_splits = (get_timeframe_length(data_train)) // self.test_len
            opt = self._ewo(data=data_train, n_splits=n_splits, lookahead=self.lookahead, train_period_length=self.train_len, test_period_length=self.test_len)

        # Execute training
        print("Train model......")
        print("-" * 60)
        # Iterate over wfo periods
        for i, (train_idx, test_idx) in enumerate(opt):
            start_train = data_train.index.get_level_values('date')[train_idx].min().strftime('%Y-%m-%d')
            end_train = data_train.index.get_level_values('date')[train_idx].max().strftime('%Y-%m-%d')
            start_test = data_train.index.get_level_values('date')[test_idx].min().strftime('%Y-%m-%d')
            end_test = data_train.index.get_level_values('date')[test_idx].max().strftime('%Y-%m-%d')
            print(f'Training from {start_train} to {end_train} || Testing from {start_test} to {end_test}:')
            # If early stopping is set to True
            if self.early:
                # Select train subset save last self.valid_len for validation
                catboost_train = catboost_data_train.slice(train_idx[:-self.valid_len].tolist())
                catboost_val = catboost_data_train.slice(train_idx[-self.valid_len:].tolist())
                # Early stop on RMSE or AUC
                print('Start training......')
                track_early_stopping = time.time()
                if self.pred == 'price':
                    model = CatBoostRegressor(iterations=1000, eval_metric='RMSE', **params)
                elif self.pred == 'sign':
                    model = CatBoostClassifier(iterations=1000, eval_metric='AUC', **params)

                # Train model
                model.fit(catboost_train, eval_set=catboost_val, early_stopping_rounds=100, use_best_model=True)

                # Print training time
                print('Train time:', round(time.time() - track_early_stopping, 2), 'seconds')

                # Plot the loss for each training period if self.plot_loss is True
                if self.plot_loss:
                    evals_result = model.get_evals_result()
                    train_rmse = evals_result['learn']['RMSE']
                    val_rmse = evals_result['validation']['RMSE']
                    plt.plot(train_rmse, label='Train RMSE')
                    plt.plot(val_rmse, label='Validation RMSE')
                    plt.xlabel('Iterations')
                    plt.ylabel('RMSE')
                    plt.legend()
                    plt.show()
            else:
                # No early stop training
                catboost_train = catboost_data_train.slice(train_idx.tolist())

                # Early stop on RMSE or AUC
                track_early_stopping = time.time()
                if self.pred == 'price':
                    model = CatBoostRegressor(iterations=1000, eval_metric='RMSE', **params)
                elif self.pred == 'sign':
                    model = CatBoostClassifier(iterations=1000, eval_metric='AUC', **params)

                # Train model
                model.fit(catboost_train)
                print('Train time:', round(time.time() - track_early_stopping, 2), 'seconds')

            # Capture predictions
            test_set = data_train.iloc[test_idx, :]
            test_factors = test_set.loc[:, model.feature_names_]
            test_ret = test_set.loc[:, ret]
            # Predict for price or sign
            print('Predicting......')
            if self.pred == 'price':
                # Get the model predictions using different number of trees from the model
                test_pred_ret = {str(n): model.predict(test_factors) for n in num_iterations}
            elif self.pred == 'sign':
                # Get the model predictions using different number of trees from the model
                test_pred_ret = {str(n): model.predict(test_factors) for n in num_iterations}

            # Create a prediction dataset
            opt_pred.append(test_ret.assign(**test_pred_ret).assign(i=i))

            # Plot histogram plot for each training period if self.plot_hist is True
            if self.plot_hist:
                self._plot_histo(test_ret.assign(**test_pred_ret).assign(i=i), self.actual_return, data_train.iloc[train_idx].index.get_level_values('date')[-1] + timedelta(days=5))

            # Save SHAP plot if self.shap is set to True
            if self.shap:
                # Save the SHAP plots for training period at start, middle, and end
                if i == 0 or i == n_splits // 2 or i == n_splits - 1:
                    print('Exporting beeswarm and waterfall SHAP plot......')
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(Pool(test_factors, test_ret, cat_features=self.categorical))
                    sv_wf = explainer(test_factors)
                    self.plot_beeswarm_cb(sv, test_factors, export_key, i)
                    self.plot_waterfall_cb(sv_wf, export_key, i)
                    print("-" * 60)

            # Save feature gain and split
            if i == 0:
                gain = self.get_feature_gain(model).to_frame()
                split = self.get_feature_split(model).to_frame()
            else:
                gain[i] = self.get_feature_gain(model)
                split[i] = self.get_feature_split(model)
            print("-" * 60)

        # Create the dataset that assigns the number of trees used for prediction to its respective prediction
        params = {key: value.__name__ if callable(value) else value for key, value in params.items()}
        all_pred_ret = pd.concat(opt_pred).assign(**params)

        # Compute IC or AS per day cross-sectionally
        by_day = all_pred_ret.groupby(level='date')
        if self.pred == 'price':
            ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x[ret], x[str(n)])[0]).to_frame(n) for n in num_iterations], axis=1)
            daily_ic_mean = list(ic_by_day.mean())
        elif self.pred == 'sign':
            as_by_day = pd.concat([by_day.apply(lambda x: accuracy_score(x[ret], x[str(n)].apply(lambda p: 1 if p > 0.5 else 0))).to_frame(n) for n in num_iterations], axis=1)
            daily_as_mean = list(as_by_day.mean())

        # Record training time for this model training period
        t = time.time() - track_wfo
        T += t
        # Save and print metrics
        if self.pred == 'price':
            metrics = pd.Series(list(param_val_train) + [t] + daily_ic_mean, index=metric_cols)
            metrics = metrics.to_frame().T
            msg = f'\t{format_time(T)} ({t:3.0f} seconds) | daily_metric max: {ic_by_day.mean().max(): 6.2%}'
            ic_by_day.columns = ic_by_day.columns.map(str)
            ic_by_day.assign(**params).to_parquet(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / 'daily_metric.parquet.brotli', compression='brotli')
        elif self.pred == 'sign':
            metrics = pd.Series(list(param_val_train) + [t] + daily_as_mean, index=metric_cols)
            metrics = metrics.to_frame().T
            msg = f'\t{format_time(T)} ({t:3.0f} seconds) | daily_metric max: {as_by_day.mean().max(): 6.2%}'
            as_by_day.columns = as_by_day.columns.map(str)
            as_by_day.assign(**params).to_parquet(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / 'daily_metric.parquet.brotli', compression='brotli')

        # Print params
        for metric_name in param_name_train:
            msg += f" | {metric_name}: {metrics[metric_name].iloc[0]}"
        print(msg)
        print("-" * 60)

        # Export actual returns, metrics, gain, split, and predictions
        self.actual_return.to_parquet(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / 'returns.parquet.brotli', compression='brotli')
        metrics.to_parquet(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / 'metrics.parquet.brotli', compression='brotli')
        gain.T.describe().T.assign(**params).to_parquet(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / 'gain.parquet.brotli', compression='brotli')
        split.T.describe().T.assign(**params).to_parquet(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / 'split.parquet.brotli', compression='brotli')
        all_pred_ret.to_parquet(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / 'predictions.parquet.brotli', compression='brotli')

        # If optuna is true, optimize for dailyIC mean
        if self.tuning[0] == 'optuna':
            if self.pred == 'price':
                objective_value = ic_by_day.mean().max()
                return objective_value
            elif self.pred == 'sign':
                objective_value = as_by_day.mean().max()
                return objective_value

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------INHERITED FUNCTION------------------------------------------------------------------------------
    # Model to train (this function is created so that it can be used for gridsearch, optuna, and default training)
    def train_option(self, trial):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------TUNING PARAMETERS----------------------------------------------------------------------------------------
        # Base params used for training
        if self.pred == 'price':
            base_params = dict(task_type='GPU', verbose=0, loss_function='RMSE', grow_policy='Lossguide', devices='0')
        elif self.pred == 'sign':
            base_params = dict(task_type='GPU', verbose=0, loss_function='Logloss', devices='0')

        # Get parameters and set num_iterations used for prediction
        params = self._get_parameters(self)
        param_names = list(params.keys())
        num_iterations = [1000]
        # This will be used for the metric dataset during training
        metric_cols = (param_names + ['time'] + ["daily_metric_" + str(n) for n in num_iterations])

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------CREATING LABELS--------------------------------------------------------------------------------------
        # Create forward returns
        self._create_fwd()
        # Get list of target returns and factors used for prediction
        ret = sorted(self.data.filter(like='target').columns)
        factors = [col for col in self.data.columns if col not in ret]

        # Get start date and end date for train data
        data_train = self.data.loc[:, factors + ret]
        start_date_train = str(data_train.index.get_level_values('date').unique()[0].date())
        end_date_train = str(data_train.index.get_level_values('date').unique()[-1].date())
        # Print the start date and end date for train period
        print("Train: " + str(start_date_train) + " --> " + str(end_date_train))
        print("-" * 60)

        # Create pool dataset for train
        catboost_data_train = Pool(data=data_train.drop(ret, axis=1), label=data_train[ret], cat_features=self.categorical)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------TRAINING MODEL---------------------------------------------------------------------------------------
        # Train model with optuna, gridsearch, or default
        if self.tuning[0] == 'optuna':
            # Get param values and print the key (formatted params)
            param_vals = list(params.values())
            key = '_'.join([str(float(p)) for p in param_vals])
            print(f'Key: {key}')
            # Create the directory
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
            # Train model and return the optimization metric for optuna
            return self.catboost(key, param_names, param_vals, base_params, data_train, catboost_data_train, metric_cols, num_iterations, ret)
        elif self.tuning[0] == 'gridsearch':
            # Get and create all possible combinations of params
            cv_params = list(product(*params.values()))
            n_params = len(cv_params)
            # Randomized grid search
            cvp = np.random.choice(list(range(n_params)), size=int(n_params / 2), replace=False)
            cv_params_ = [cv_params[i] for i in cvp][:self.tuning[1]]
            print("Number of gridsearch iterations: " + str(len(cv_params_)))
            # Iterate over (shuffled) hyperparameter combinations
            for p, param_vals in enumerate(cv_params_):
                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                # Create the directory
                os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
                # Train model
                self.catboost(key, param_names, param_vals, base_params, data_train, catboost_data_train, metric_cols, num_iterations, ret)
        elif self.tuning == 'default':
            # Get param values and print the key (formatted params)
            param_vals = list(params.values())
            key = '_'.join([str(float(p)) for p in param_vals])
            print(f'Key: {key}')
            # Create the directory
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
            # Train model
            self.catboost(key, param_names, param_vals, base_params, data_train, catboost_data_train, metric_cols, num_iterations, ret)
        elif self.tuning == 'best':
            # Get list of best params
            for param_set in params:
                # Get param values and print the key (formatted params)
                param_vals = list(param_set.values())
                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                # Create the directory
                os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
                # Train model
                self.catboost(key, param_names, param_vals, base_params, data_train, catboost_data_train, metric_cols, num_iterations, ret)