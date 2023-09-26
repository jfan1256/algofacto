from typing import Optional

import pandas as pd

from functions.utils.func import *
from itertools import product
from scipy.stats import spearmanr
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from catboost import Pool
from sklearn.metrics import accuracy_score
from typing import Optional, Union, List

import os
import time
import shap
import shutil
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


class AlphaModel:
    def __init__(self, model_name: str = None,
                 tuning: [str, int] = None,
                 plot_loss: bool = False,
                 plot_hist: bool = False,
                 pred: str = 'price',
                 stock: str = None,
                 lookahead: int = 1,
                 incr: bool = False,
                 opt: str = None,
                 weight: bool = False,
                 outlier: bool = False,
                 early: bool = True,
                 pretrain_len: Optional[int] = None,
                 train_len: int = None,
                 valid_len: int = None,
                 test_len: int = None,
                 **kwargs):

        self.data = None
        self.model_name = model_name
        self.categorical = []
        self.tuning = tuning
        self.plot_loss = plot_loss
        self.plot_hist = plot_hist
        self.pred = pred
        self.stock = stock
        self.lookahead = lookahead
        self.incr = incr
        self.opt = opt
        self.weight = weight
        self.outlier = outlier
        self.early = early
        self.pretrain_len = pretrain_len
        self.train_len = train_len
        self.valid_len = valid_len
        self.test_len = test_len
        self.actual_return = None
        self.parameter_specs = kwargs

        if self.opt == 'ewo':
            assert self.incr == False, ValueError("incr must be set to False if opt is ewo")
            assert self.pretrain_len == 0, ValueError("pretrain_len must be 0 if opt is ewo")
        assert self.pred == 'price' or self.pred == 'sign', ValueError('Must use either "price" or "sign" for pred parameter')
        if self.incr:
            assert self.pretrain_len > 0, ValueError("Pretrain_len must be greater than 0 to use incremental training")
        assert self.tuning[0] == 'optuna' or self.tuning[0] == 'gridsearch' or self.tuning == 'default', \
            ValueError("Must use either ['optuna', num_trials=int], ['gridsearch', num_search=int], or 'default' for tuning parameter")
        assert self.lookahead>=1, ValueError('Must be greater than 0')
        if self.early == False:
            assert self.plot_loss == False, ValueError("Cannot plot validation loss if early is set to False")
            assert self.valid_len == 0, ValueError("Must set valid_len to 0 is early is set to False")
        if self.early:
            assert self.valid_len > 0, ValueError("valid_len must be > 0 if early is set to True")


    # Renumber Category data to consecutive numbers
    @staticmethod
    def renumber_cat(factor):
        category_mapping = {}
        for col in factor.columns:
            unique_categories = factor[col].unique()
            category_mapping[col] = {category: i for i, category in enumerate(unique_categories)}
            factor[col] = factor[col].map(category_mapping[col])
        factor = factor.astype(int)
        return factor

    def create_fwd(self):
        if self.outlier:
            if self.pred == 'sign':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'TARGET_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}'].shift(-self.lookahead)
                self.data = self.data.dropna(subset=[f'TARGET_{self.lookahead}D'])
                self.data[f'TARGET_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'TARGET_{self.lookahead}D'].apply(lambda x: np.sign(x))
            elif self.pred == 'price':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'TARGET_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}'].shift(-self.lookahead)
                condition = self.data[f'TARGET_{self.lookahead}D'].abs() > 0.05
                self.data.loc[condition, f'TARGET_{self.lookahead}D'] = np.nan
                self.data = self.data.dropna(subset=[f'TARGET_{self.lookahead}D'])
        else:
            if self.pred == 'sign':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'TARGET_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}'].shift(-self.lookahead)
                self.data = self.data.dropna(subset=[f'TARGET_{self.lookahead}D'])
                self.data[f'TARGET_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'TARGET_{self.lookahead}D'].apply(lambda x: np.sign(x))
            elif self.pred == 'price':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'TARGET_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}'].shift(-self.lookahead)
                self.data = self.data.dropna(subset=[f'TARGET_{self.lookahead}D'])

    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Adding to {self.model_name}', animation=spinner_animation)
    def add_factor(self, factor, categorical=False, normalize=False):
        # If self.data is None then concat, else join
        def condition_add():
            if self.data is None:
                self.data = pd.concat([self.data, factor], axis=1)
            else:
                self.data = pd.merge(self.data, factor, left_index=True, right_index=True, how='left')
                self.data = self.data.loc[~self.data.index.duplicated(keep='first')]

        if categorical:
            self.categorical = self.categorical + factor.columns.tolist()
            factor = self.renumber_cat(factor)
            """self.data = pd.concat([self.data, factor], axis=1)"""
            condition_add()
        else:
            if normalize:
                factor = factor.replace([np.inf, -np.inf], np.nan)
                scaler = MinMaxScaler((-1, 1))

                def normalize(group):
                    value = scaler.fit_transform(group)
                    return pd.DataFrame(value, index=group.index, columns=group.columns)

                factor = factor.groupby(level='date').apply(normalize)
                """self.data = pd.concat([self.data, factor], axis=1)"""
                condition_add()

            else:
                """self.data = pd.concat([self.data, factor], axis=1)"""
                condition_add()

    @staticmethod
    def pretrain(data, pretrain_len):
        unique_dates = data.index.get_level_values('date').unique()
        days = sorted(unique_dates)

        train_start = 0
        train_end = pretrain_len - 1

        dates = data.reset_index()[['date']]
        pre_train_idx = dates[(days[train_start] <= dates.date) & (dates.date <= days[train_end])].index
        return pre_train_idx

    @staticmethod
    def wfo(data, n_splits, lookahead, train_period_length, test_period_length):
        unique_dates = data.index.get_level_values('date').unique()
        days = sorted(unique_dates)
        split_idx = []

        for i in range(n_splits):
            train_start_idx = i * test_period_length
            train_end_idx = train_start_idx + train_period_length - 1
            test_start_idx = train_end_idx + lookahead
            test_end_idx = test_start_idx + test_period_length - 1

            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = data.reset_index()[['date']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(days[train_start] <= dates.date) & (dates.date <= days[train_end])].index
            test_idx = dates[(days[test_start] <= dates.date) & (dates.date <= days[test_end])].index
            yield train_idx, test_idx

    @staticmethod
    def ewo(data, n_splits, lookahead, train_period_length, test_period_length):
        unique_dates = data.index.get_level_values('date').unique()
        days = sorted(unique_dates)
        split_idx = []

        # Start training from the first available data point
        train_start_idx = 0

        for i in range(n_splits):
            train_end_idx = train_start_idx + i * test_period_length + train_period_length - 1

            test_start_idx = train_end_idx + lookahead
            test_end_idx = test_start_idx + test_period_length - 1

            if test_end_idx >= len(days):
                break

            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = data.reset_index()[['date']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(days[train_start] <= dates.date) & (dates.date <= days[train_end])].index
            test_idx = dates[(days[test_start] <= dates.date) & (dates.date <= days[test_end])].index
            yield train_idx, test_idx

    @staticmethod
    def get_feature_gain(model):
        fi = model.feature_importance(importance_type='gain')
        return pd.Series(fi / fi.sum(), index=model.feature_name())

    @staticmethod
    def get_feature_split(model):
        fi = model.feature_importance(importance_type='split')
        return pd.Series(fi / fi.sum(), index=model.feature_name())

    def plot_beeswarm_gbm(self, sv, explainer, X, key, i):
        if self.pred == 'price':
            shap.plots.beeswarm(sv, max_display=80, show=False)
            plt.savefig(str(get_result() / f'{self.model_name}' / f'params_{key}' / f'beeswarm_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
            plt.close()
        elif self.pred == 'sign':
            values = explainer.shap_values(X)
            shap.summary_plot(values, feature_names=X.columns.tolist(), max_display=80, show=False)
            plt.savefig(str(get_result() / f'{self.model_name}' / f'params_{key}' / f'beeswarm_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
            plt.close()

    def plot_waterfall_gbm(self, sv, key, i):
        shap.plots.waterfall(sv[0], max_display=80, show=False)
        plt.savefig(str(get_result() / f'{self.model_name}' / f'params_{key}' / f'waterfall_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
        plt.close()

    def plot_beeswarm_cb(self, sv, X, key, i):
        shap.summary_plot(sv, X, max_display=80, show=False)
        plt.savefig(str(get_result() / f'{self.model_name}' / f'params_{key}' / f'beeswarm_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
        plt.close()

    def plot_waterfall_cb(self, sv, key, i):
        shap.plots.waterfall(sv[0], max_display=80, show=False)
        plt.savefig(str(get_result() / f'{self.model_name}' / f'params_{key}' / f'waterfall_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
        plt.close()

    @staticmethod
    def plot_histo(pred, ret, date):
        pred = pred['150'].loc[pred.index.get_level_values('date') == date]
        ret = ret.RET_01.loc[ret.index.get_level_values('date') == date]
        plt.hist(pred, bins='auto', edgecolor='black', alpha=0.5, label=f"Pred: {date}")
        plt.hist(ret, bins='auto', edgecolor='red', alpha=0.5, label=f"Real: {date}")
        plt.title('Histogram of values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def lightgbm(self):
        print('List of categorical inputs:')
        print(self.categorical)
        def model_training(trial):
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------FUNCTIONS WITHIN MODEL_TRAINING------------------------------------------------------------------------------
            def custom_loss(y_pred, dataset):
                y_true = dataset.get_label()
                penalty = 0.1
                grad = 2 * (y_pred - y_true)
                hess = 2 * np.ones_like(y_true)
                # Penalizing both sign predictions
                grad += penalty * (np.sign(y_pred) - np.sign(y_true))
                return grad, hess

            def custom_eval(y_pred, dataset):
                y_true = dataset.get_label()
                loss = (y_pred - y_true) ** 2 + 0.2 * (np.sign(y_pred) - np.sign(y_true)) ** 2
                return 'custom_loss', loss.mean(), False

            def get_parameters():
                if self.tuning[0] == 'optuna':
                    params = {}
                    for key, spec in self.parameter_specs.items():
                        method_name = spec['optuna'][0]
                        if method_name == 'suggest_float':
                            low, high, *rest = spec['optuna'][1:]
                            log_val = rest[0] if rest else False
                            params[key] = trial.suggest_float(key, low, high, log=log_val)
                        else:
                            params[key] = getattr(trial, method_name)(key, *spec['optuna'][1:])
                    return params
                elif self.tuning[0] == 'gridsearch':
                    return {key: spec['gridsearch'] for key, spec in self.parameter_specs.items()}
                elif self.tuning == 'default':
                    return {key: spec['default'] for key, spec in self.parameter_specs.items()}

            def plot_avg_loss(folds_data):
                # Determine the maximum length across all folds
                max_len_training = max(len(fold['training']['l2']) for fold in folds_data)
                max_len_valid_1 = max(len(fold['valid_1']['l2']) for fold in folds_data)

                # Initialize cumulative loss values
                cumulative_losses = {
                    'training': np.zeros(max_len_training),
                    'valid_1': np.zeros(max_len_valid_1)
                }
                counts_training = np.zeros(max_len_training)
                counts_valid_1 = np.zeros(max_len_valid_1)

                # Sum up the losses for each epoch across folds and count non-NaN values
                for fold_data in folds_data:
                    training_vals = np.array(fold_data['training']['l2'])
                    valid_1_vals = np.array(fold_data['valid_1']['l2'])

                    # Pad arrays with NaN
                    training_vals = np.pad(training_vals, (0, max_len_training - len(training_vals)), constant_values=np.nan)
                    valid_1_vals = np.pad(valid_1_vals, (0, max_len_valid_1 - len(valid_1_vals)), constant_values=np.nan)

                    # Update cumulative sums and counts
                    valid_mask_training = ~np.isnan(training_vals)
                    valid_mask_valid_1 = ~np.isnan(valid_1_vals)

                    cumulative_losses['training'][valid_mask_training] += training_vals[valid_mask_training]
                    cumulative_losses['valid_1'][valid_mask_valid_1] += valid_1_vals[valid_mask_valid_1]

                    counts_training[valid_mask_training] += 1
                    counts_valid_1[valid_mask_valid_1] += 1

                # Calculate the average loss for each epoch across folds
                avg_losses = {
                    'training': {'l2': (cumulative_losses['training'] / counts_training).tolist()},
                    'valid_1': {'l2': (cumulative_losses['valid_1'] / counts_valid_1).tolist()}
                }

                evals_result = {'training': avg_losses['training'], 'valid_1': avg_losses['valid_1']}
                return evals_result

            def train_model(export_key, param_name_train, param_val_train):
                params = dict(zip(param_name_train, param_val_train))
                params.update(base_params)
                # Set up cross-validation and track time
                T = 0

                log_loss = []
                track_wfo = time.time()

                if self.pretrain_len > 0:
                    # Pre-train model
                    print("Pretrain model......")
                    print("-" * 60)
                    lgb_pretrain = lgb_data_pretrain.subset(used_indices=pretrain_idx.tolist()).construct()
                    prev_model = lgb.train(params=params, train_set=lgb_pretrain, num_boost_round=1000)

                opt_pred, gain, split = [], [], []
                if self.opt == 'wfo':
                    n_splits = (get_timeframe_length(data_train) - self.train_len) // self.test_len
                    opt = self.wfo(data=data_train, n_splits=n_splits, lookahead=self.lookahead, train_period_length=self.train_len, test_period_length=self.test_len)
                elif self.opt == 'ewo':
                    n_splits = (get_timeframe_length(data_train)) // self.test_len
                    opt = self.ewo(data=data_train, n_splits=n_splits, lookahead=self.lookahead, train_period_length=self.train_len, test_period_length=self.test_len)

                print("Train model......")
                print("-" * 60)

                # Iterate over wfo periods
                for i, (train_idx, test_idx) in enumerate(opt):
                    if self.early:
                        # Select train subset save last 30 for validation
                        lgb_train = lgb_data_train.subset(used_indices=train_idx.tolist()[:-self.valid_len]).construct()
                        lgb_val = lgb_data_train.subset(used_indices=train_idx.tolist()[-self.valid_len:]).construct()
                        lgb_early_stop = lgb.early_stopping(100)

                        # Early stop on MSE
                        track_training = time.time()
                        evals = {}
                        if self.incr:
                            model = lgb.train(init_model=prev_model, params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=1000,
                                              callbacks=[lgb_early_stop, lgb.record_evaluation(evals)])

                            # Retrain on entire dataset with less num_boost_round
                            whole_set = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                            model = lgb.train(init_model=model, params=params, train_set=whole_set, num_boost_round=150)

                            # Store this model for next fold
                            prev_model = model
                        else:
                            model = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=1000, callbacks=[lgb_early_stop, lgb.record_evaluation(evals)])

                            # Retrain on entire dataset with less num_boost_round
                            whole_set = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                            model = lgb.train(init_model=model, params=params, train_set=whole_set, num_boost_round=150)
                    else:
                        # Select train subset
                        lgb_train = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                        track_training = time.time()
                        if self.incr:
                            model = lgb.train(init_model=prev_model, params=params, train_set=lgb_train, num_boost_round=1000)

                            # Store this model for next fold
                            prev_model = model
                        else:
                            model = lgb.train(params=params, train_set=lgb_train, num_boost_round=1000)


                    print('Train time:', round(time.time() - track_training, 2), 'seconds')

                    if self.early:
                        # Capture training loss and validation loss
                        log_loss.append(evals)
                        if self.plot_loss:
                            lgb.plot_metric(evals)
                            plt.show()

                    # Capture predictions
                    test_set = data_train.iloc[test_idx, :]
                    test_factors = test_set.loc[:, model.feature_name()]
                    test_ret = test_set.loc[:, ret]
                    if self.pred == 'price':
                        print('Predicting......')
                        test_pred_ret = {str(n): model.predict(test_factors, num_iteration=n) for n in num_iterations}
                        print("-" * 60)
                    elif self.pred == 'sign':
                        print('Predicting......')
                        test_pred_ret = {str(n): (2 * (model.predict(test_factors, num_iteration=n) >= 0.5).astype(int) - 1) for n in num_iterations}
                        print("-" * 60)

                    # Record predictions for each fold
                    opt_pred.append(test_ret.assign(**test_pred_ret).assign(i=i))

                    # Plot hist
                    if self.plot_hist:
                        self.plot_histo(test_ret.assign(**test_pred_ret).assign(i=i), self.actual_return, data_train.iloc[train_idx].index.get_level_values('date')[-1] + timedelta(days=5))

                    # Record shap value
                    if i == 0 or i == n_splits // 2 or i == n_splits - 1:
                        print('Exporting beeswarm and waterfall SHAP plot......')
                        explainer = shap.TreeExplainer(model)
                        sv = explainer(test_factors)
                        self.plot_beeswarm_gbm(sv, explainer, test_factors, export_key, i)
                        if self.pred == 'price':
                            self.plot_waterfall_gbm(sv, export_key, i)
                        print("-" * 60)

                    # Log feature importance and SHAP values
                    if i == 0:
                        gain = self.get_feature_gain(model).to_frame()
                        split = self.get_feature_split(model).to_frame()
                    else:
                        gain[i] = self.get_feature_gain(model)
                        split[i] = self.get_feature_split(model)

                if self.early:
                    #Plot training loss and validation loss and track training time
                    eval_results = plot_avg_loss(log_loss)
                    lgb.plot_metric(eval_results)
                    plt.savefig(str(get_result() / f'{self.model_name}' / f'params_{export_key}' / f'loss.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
                    plt.close()

                # Combine fold results and set params string
                params = {name: val.__name__ if callable(val) else val for name, val in params.items()}
                all_pred_ret = pd.concat(opt_pred).assign(**params)

                # Compute IC or AS per day
                by_day = all_pred_ret.groupby(level='date')
                if self.pred == 'price':
                    ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x[ret], x[str(n)])[0]).to_frame(n) for n in num_iterations], axis=1)
                    daily_ic_mean = list(ic_by_day.mean())
                elif self.pred == 'sign':
                    as_by_day = pd.concat([by_day.apply(lambda x: accuracy_score(x[ret], x[str(n)])).to_frame(n) for n in num_iterations], axis=1)
                    daily_as_mean = list(as_by_day.mean())

                # Record training time for this model
                t = time.time() - track_wfo
                T += t

                # Collect and print metrics
                if self.pred == 'price':
                    metrics = pd.Series(list(param_val_train) + [t] + daily_ic_mean, index=metric_cols)
                    metrics = metrics.to_frame().T
                    msg = f'\t{format_time(T)} ({t:3.0f} seconds) | dIC_mean max: {ic_by_day.mean().max(): 6.2%}'
                    ic_by_day.columns = ic_by_day.columns.map(str)
                    ic_by_day.assign(**params).to_parquet(get_result() / f'{self.model_name}' / f'params_{export_key}' / 'dailyIC.parquet.brotli', compression='brotli')
                elif self.pred == 'sign':
                    metrics = pd.Series(list(param_val_train) + [t] + daily_as_mean, index=metric_cols)
                    metrics = metrics.to_frame().T
                    msg = f'\t{format_time(T)} ({t:3.0f} seconds) | dAS_mean max: {as_by_day.mean().max(): 6.2%}'
                    as_by_day.columns = as_by_day.columns.map(str)
                    as_by_day.assign(**params).to_parquet(get_result() / f'{self.model_name}' / f'params_{export_key}' / 'dailyAS.parquet.brotli', compression='brotli')

                # Print params
                for metric_name in param_name_train:
                    msg += f" | {metric_name}: {metrics[metric_name].iloc[0]}"
                print(msg)
                print("-" * 60)

                # Export results
                self.actual_return.to_parquet(get_result() / f'{self.model_name}' / f'params_{export_key}' / 'returns.parquet.brotli', compression='brotli')
                metrics.to_parquet(get_result() / f'{self.model_name}' / f'params_{export_key}' / 'metrics.parquet.brotli', compression='brotli')
                gain.T.describe().T.assign(**params).to_parquet(get_result() / f'{self.model_name}' / f'params_{export_key}' / 'gain.parquet.brotli', compression='brotli')
                split.T.describe().T.assign(**params).to_parquet(get_result() / f'{self.model_name}' / f'params_{export_key}' / 'split.parquet.brotli', compression='brotli')
                all_pred_ret.to_parquet(get_result() / f'{self.model_name}' / f'params_{export_key}' / 'predictions.parquet.brotli', compression='brotli')

                if self.tuning[0] == 'optuna':
                    if self.pred == 'price':
                        objective_value = ic_by_day.mean().max()
                        return objective_value
                    elif self.pred == 'sign':
                        objective_value = as_by_day.mean().max()
                        return objective_value

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------TUNING PARAMETERS----------------------------------------------------------------------------------------
            # Hyperparameter Options used for Optuna, GridSearch, or None
            if self.pred == 'price':
                base_params = dict(boosting='gbdt', device_type='gpu', gpu_platform_id=1, gpu_device_id=0, verbose=-1, gpu_use_dp=True, objective='regression', metric='mse')
            elif self.pred == 'sign':
                base_params = dict(boosting='gbdt', device_type='gpu', gpu_platform_id=1, gpu_device_id=0, verbose=-1, gpu_use_dp=True, objective='binary', metric='binary_logloss')

            params = get_parameters()
            param_names = list(params.keys())
            num_iterations = [150, 200, 250, 300, 400]
            metric_cols = (param_names + ['time'] + ["dIC_mean_" + str(n) for n in num_iterations])

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------CREATING LABELS--------------------------------------------------------------------------------------
            # Create forward returns
            self.create_fwd()

            # Get list of target and features
            ret = sorted(self.data.filter(like='TARGET').columns)
            factors = self.data.columns.difference(ret).tolist()

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------PRETRAIN OR NOT--------------------------------------------------------------------------------------
            # Pretrain or not
            if self.pretrain_len > 0:
                # Set labels and outcome data for pre_train
                data_pretrain = self.data.loc[:, factors + ret]
                pretrain_idx = self.pretrain(data=data_pretrain, pretrain_len=self.pretrain_len)
                print("-" * 60)
                print("Pretrain: " + str(data_pretrain.iloc[pretrain_idx].index.get_level_values('date')[0].date()) + " --> " + str(
                    data_pretrain.iloc[pretrain_idx].index.get_level_values('date')[-1].date()))

                # Get start date and end date for train data
                start_date_train = str(data_pretrain.iloc[pretrain_idx].index.get_level_values('date')[-1].date() + timedelta(days=1))
                end_date_train = str(data_pretrain.index.get_level_values('date')[-1].date())
                print("Train: " + str(start_date_train) + " --> " + str(end_date_train))
                print("-" * 60)
                data_train = set_timeframe(self.data.loc[:, factors + ret], start_date_train, end_date_train)

                # Create binary dataset
                lgb_data_pretrain = lgb.Dataset(data=data_pretrain.drop(ret, axis=1),
                                                label=data_pretrain[ret], categorical_feature=self.categorical,
                                                free_raw_data=False, params={'device_type': 'gpu'})

                lgb_data_train = lgb.Dataset(data=data_train.drop(ret, axis=1), label=data_train[ret],
                                             categorical_feature=self.categorical, free_raw_data=False, params={'device_type': 'gpu'})

                if self.weight:
                    # Emphasize extreme returns
                    threshold = 0.001
                    weight_pretrain = data_pretrain[ret[0]].map(lambda x: 1 if abs(x) > threshold else 0.1).values
                    weight_train = data_train[ret[0]].map(lambda x: 1 if abs(x) > threshold else 0.1).values
                    # Emphasize negative returns
                    """weight_pretrain = data_pretrain[ret[0]].map(lambda x: 1 if x < 0 else 0.1).values
                    weight_train = data_train[ret[0]].map(lambda x: 1 if x < 0 else 0.1).values"""
                    lgb_data_pretrain.set_weight(weight_pretrain)
                    lgb_data_train.set_weight(weight_train)

            else:
                # Get start date and end date for train data
                data_train = self.data.loc[:, factors + ret]
                start_date_train = str(data_train.index.get_level_values('date')[0].date())
                end_date_train = str(data_train.index.get_level_values('date')[-1].date())
                print("-" * 60)
                print("Train: " + str(start_date_train) + " --> " + str(end_date_train))
                print("-" * 60)

                # Create binary dataset
                lgb_data_train = lgb.Dataset(data=data_train.drop(ret, axis=1), label=data_train[ret],
                                             categorical_feature=self.categorical, free_raw_data=False, params={'device_type': 'gpu'})

                if self.weight:
                    # Emphasize extreme returns
                    threshold = 0.001
                    weight_train = data_train[ret[0]].map(lambda x: 1 if abs(x) > threshold else 0.1).values
                    # Emphasize negative returns
                    """weight_train = data_train[ret[0]].map(lambda x: 1 if x < 0 else 0.1).values"""
                    lgb_data_train.set_weight(weight_train)

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------TRAINING MODEL---------------------------------------------------------------------------------------
            # Training model with optuna, gridsearch, or none
            if self.tuning[0] == 'optuna':
                # Get param values
                param_vals = list(params.values())
                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                os.makedirs(get_result() / f'{self.model_name}' / f'params_{key}')

                # Train model and return IC score for optuna
                return train_model(key, param_names, param_vals)
            elif self.tuning[0] == 'gridsearch':
                # Create all possible combinations of params
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
                    os.makedirs(get_result() / f'{self.model_name}' / f'params_{key}')

                    # Train model
                    train_model(key, param_names, param_vals)
            elif self.tuning == 'default':
                # Get param values
                param_vals = list(params.values())
                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                os.makedirs(get_result() / f'{self.model_name}' / f'params_{key}')

                # Train model
                train_model(key, param_names, param_vals)

        # ===============================================================================================================================================================================
        # --------------------------------------------------------------------------MODEL_TRAINING()-------------------------------------------------------------------------------------
        if self.tuning[0] == 'optuna':
            # Create new directory named (modelName)
            shutil.rmtree(get_result() / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_result() / f'{self.model_name}')

            # Execute train
            study = optuna.create_study(direction="maximize")
            study.optimize(model_training, n_trials=self.tuning[1])
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial:")
            trial = study.best_trial
            print("     IC: {}".format(trial.value))
            print("     Params: ")
            for name, number in trial.params.items():
                print("         {}: {}".format(name, number))
            print("-" * 60)
        else:
            # Create new directory named (modelName)
            shutil.rmtree(get_result() / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_result() / f'{self.model_name}')

            # Execute train
            model_training(None)

    def catboost(self):
        assert not self.incr, 'Cannot run incremental training with catboost'
        # Create forward returns
        self.create_fwd()

        # Hyperparameter Options used for GridSearch
        base_params = dict(task_type='GPU', verbose=150)

        if self.tuning:
            max_depths = [3, 5, 7, 9]
            num_leaves_opts = [15]
            min_child_samples = [20, 250, 500]
            learning_rate_opts = [.15]
            l2_leaf_reg_opts = [0.01]
        else:
            max_depths = [10]
            num_leaves_opts = [15]
            min_child_samples = [60]
            learning_rate_opts = [.15]
            l2_leaf_reg_opts = [0.01]

        param_names = ['max_depth', 'learning_rate', 'num_leaves', 'min_child_samples', 'l2_leaf_reg']
        cv_params = list(product(max_depths, learning_rate_opts, num_leaves_opts, min_child_samples, l2_leaf_reg_opts))
        n_params = len(cv_params)

        # Get list of target and features
        ret = sorted(self.data.filter(like='TARGET').columns)
        factors = self.data.columns.difference(ret).tolist()

        # Iterations for mode boosting
        num_iterations = [150, 175, 200, 250, 300]
        """num_iterations = [150, 250, 500, 1000]"""

        # Metric column names for export
        if self.pred == 'price':
            metric_cols = (param_names + ['time'] + ["dIC_mean_" + str(n) for n in num_iterations])
        elif self.pred == 'sign':
            metric_cols = (param_names + ['time'] + ["dAS_mean_" + str(n) for n in num_iterations])

        # Create new directory named (modelName)
        shutil.rmtree(get_result() / f'{self.model_name}', ignore_errors=True)
        os.makedirs(get_result() / f'{self.model_name}')

        # Randomized grid search
        if n_params == 1:
            cvp = [0]
        else:
            cvp = np.random.choice(list(range(n_params)), size=int(n_params / 2), replace=False)
        cv_params_ = [cv_params[i] for i in cvp][:40]
        print("Number of grid search iterations: " + str(len(cv_params_)))

        # Get start date and end date for train data
        data_train = self.data.loc[:, factors + ret]
        start_date_train = str(data_train.index.get_level_values('date')[0].date())
        end_date_train = str(data_train.index.get_level_values('date')[-1].date())
        print("Train: " + str(start_date_train) + " --> " + str(end_date_train))
        print("-" * 60)

        # Pool dataset
        catboost_data_train = Pool(data=data_train.drop(ret, axis=1), label=data_train[ret], cat_features=self.categorical)

        # Set up cross-validation and track time
        T = 0
        n_splits = (get_timeframe_length(data_train) - self.train_len) // self.test_len

        # Iterate over (shuffled) hyperparameter combinations
        for p, param_vals in enumerate(cv_params_):
            key = '_'.join([str(p) for p in param_vals])
            os.makedirs(get_result() / f'{self.model_name}' / f'params_{key}')

            # Add additional Params
            params = dict(zip(param_names, param_vals))
            params.update(base_params)

            if self.pred == 'price':
                params['loss_function'] = 'RMSE'
                params['grow_policy'] = 'Lossguide'
            elif self.pred == 'sign':
                params['loss_function'] = 'Logloss'

            track_wfo = time.time()
            wfo_pred, fi = [], []
            wfo = self.wfo(data=data_train, n_splits=n_splits, lookahead=self.lookahead, train_period_length=self.train_len, test_period_length=self.test_len)

            print("Train model......")
            print("-" * 60)

            # Iterate over wfo periods
            for i, (train_idx, test_idx) in enumerate(wfo):
                # Select train subset save last 30 for validation
                catboost_train = catboost_data_train.slice(train_idx[:-30].tolist())
                catboost_val = catboost_data_train.slice(train_idx[-30:].tolist())

                # Early stop on MSE
                track_early_stopping = time.time()
                if self.pred == 'price':
                    model = CatBoostRegressor(iterations=1000, eval_metric='RMSE', **params)
                elif self.pred == 'sign':
                    model = CatBoostClassifier(iterations=1000, eval_metric='AUC', **params)
                model.fit(catboost_train, eval_set=catboost_val, early_stopping_rounds=100)
                print('Train time:', round(time.time() - track_early_stopping, 2), 'seconds')
                print("-" * 60)

                # Capture predictions
                test_set = data_train.iloc[test_idx, :]
                test_factors = test_set.loc[:, model.feature_names_]
                test_ret = test_set.loc[:, ret]
                if self.pred == 'price':
                    test_pred_ret = {str(n): model.predict(test_factors, ntree_end=n) for n in num_iterations}
                elif self.pred == 'sign':
                    test_pred_ret = {str(n): (model.predict(test_factors, ntree_end=n) >= 0.5).astype(int) for n in num_iterations}

                # Record predictions for each fold
                wfo_pred.append(test_ret.assign(**test_pred_ret).assign(i=i))

                # Record shap value
                if i == 0 or i == n_splits // 2 or i == n_splits - 1:
                    print('Exporting beeswarm and waterfall SHAP plot......')
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(Pool(test_factors, test_ret, cat_features=self.categorical))
                    sv_wf = explainer(test_factors)
                    self.plot_beeswarm_cb(sv, test_factors, key, i)
                    self.plot_waterfall_cb(sv_wf, key, i)
                    print("-" * 60)

            # Combine fold results and set params string
            params = {key: value.__name__ if callable(value) else value for key, value in params.items()}
            all_pred_ret = pd.concat(wfo_pred).assign(**params)

            # Compute IC or AS per day
            by_day = all_pred_ret.groupby(level='date')
            if self.pred == 'price':
                ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x[ret], x[str(n)])[0]).to_frame(n) for n in num_iterations], axis=1)
                daily_ic_mean = list(ic_by_day.mean())
            elif self.pred == 'sign':
                as_by_day = pd.concat([by_day.apply(lambda x: accuracy_score(x[ret], x[str(n)])).to_frame(n) for n in num_iterations], axis=1)
                daily_as_mean = list(as_by_day.mean())

            # Record training time for this model
            t = time.time() - track_wfo
            T += t

            # Collect and print metrics
            if self.pred == 'price':
                metrics = pd.Series(list(param_vals) + [t] + daily_ic_mean, index=metric_cols)
                metrics = metrics.to_frame().T
                msg = f'\t{p:3.0f} | {format_time(T)} ({t:3.0f} seconds) | dIC_mean max: {ic_by_day.mean().max(): 6.2%}'
                ic_by_day.columns = ic_by_day.columns.map(str)
                ic_by_day.assign(**params).to_parquet(get_result() / f'{self.model_name}' / f'params_{key}' / 'dailyIC.parquet.brotli', compression='brotli')
            elif self.pred == 'sign':
                metrics = pd.Series(list(param_vals) + [t] + daily_as_mean, index=metric_cols)
                metrics = metrics.to_frame().T
                msg = f'\t{p:3.0f} | {format_time(T)} ({t:3.0f} seconds) | dAS_mean max: {as_by_day.mean().max(): 6.2%}'
                as_by_day.columns = as_by_day.columns.map(str)
                as_by_day.assign(**params).to_parquet(get_result() / f'{self.model_name}' / f'params_{key}' / 'dailyAS.parquet.brotli', compression='brotli')

            for metric_name in param_names:
                value = metrics[metric_name].iloc[0]
                msg += f" | {metric_name}: {value}"
            print(msg)
            print("-" * 60)

            # Export results
            self.actual_return.to_parquet(get_result() / f'{self.model_name}' / f'params_{key}' / 'returns.parquet.brotli', compression='brotli')
            metrics.to_parquet(get_result() / f'{self.model_name}' / f'params_{key}' / 'metrics.parquet.brotli', compression='brotli')
            fi.T.describe().T.assign(**params).to_parquet(get_result() / f'{self.model_name}' / f'params_{key}' / 'fi.parquet.brotli', compression='brotli')
            all_pred_ret.to_parquet(get_result() / f'{self.model_name}' / f'params_{key}' / 'predictions.parquet.brotli', compression='brotli')
