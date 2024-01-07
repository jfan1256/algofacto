from core.operation import *

from itertools import product
from scipy.stats import spearmanr
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import Pool
from sklearn.metrics import accuracy_score
from typing import Optional

import os
import time
import shap
import shutil
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


class ModelTrain:
    def __init__(self,
                 live: bool = None,
                 model_name: str = None,
                 end: str = None,
                 tuning: [str, int] = None,
                 shap: bool = False,
                 plot_loss: bool = False,
                 plot_hist: bool = False,
                 pred: str = 'price',
                 stock: str = None,
                 lookahead: int = 1,
                 trend: int = 0,
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

        '''
        live (bool): Get historical data or live data
        model_name (str): Model name (lightgbm, randomforest, or catboost)
        end (str: YYYY-MM-DD): Final date for model training
        tuning (str): Type of parameter to use (i.e., default, optuna, etc.)
        shap (bool): Save shap plot or not
        plot_loss (bool): Plot training and validation curve after each window training or not
        plot_hist (bool): Plot actual returns and predicted returns after each window training or not
        pred (str): Predict for price returns or price movement
        stock (str): Name of index for stocks ('permno' or 'ticker')
        lookahead (int): Lookahead period to predict for
        trend (int): Size of rolling window to calculate trend (for price movement predictions)
        incr (bool): Perform incremental training or not (only for lightgbm model)
        opt (str): Type of training optimization ('ewo' or 'wfo')
        weight (float): Weight for sample weight training (only for lightgbm)
        outlier (bool): Handle outlier data in label data or not
        early (bool): Train with early stopping or not
        pretrain_len (int): Pretrain length for model training
        train_len (int): Train length for model training
        valid_len (int): Validation length for model training
        test_len (int): Prediction length for model training
        kwargs (dict): Model parameters to feed into model
        '''

        self.data = None
        self.live = live
        self.model_name = model_name
        self.end = end
        self.categorical = []
        self.tuning = tuning
        self.shap = shap
        self.plot_loss = plot_loss
        self.plot_hist = plot_hist
        self.pred = pred
        self.stock = stock
        self.lookahead = lookahead
        self.trend = trend
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
        assert self.pred == 'price' or self.pred == 'sign', ValueError('must use either "price" or "sign" for pred parameter')
        if self.incr == False:
            assert self.pretrain_len == 0, ValueError("pretrain_len must be 0 if incremental training is false")
        assert self.tuning[0] == 'optuna' or self.tuning[0] == 'gridsearch' or self.tuning == 'default' or self.tuning=='best', \
            ValueError("must use either ['optuna', num_trials=int], ['gridsearch', num_search=int], or 'default' for tuning parameter")
        assert self.lookahead>=1, ValueError('lookahead must be greater than 0')
        if self.early == False and 'randomforest' not in self.model_name:
            assert self.plot_loss == False, ValueError("cannot plot validation loss if early is set to False")
            assert self.valid_len == 0, ValueError("must set valid_len to 0 is early is set to False")
        if self.early:
            assert self.valid_len > 0, ValueError("valid_len must be > 0 if early is set to True")

    # Gets the required parameters from **params
    def _get_parameters(self, trial):
        # Options can either be optuna, gridsearch, or default
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
        elif self.tuning == 'best':
            num_sets = len(self.parameter_specs['max_depth']['best'])
            best_param_sets = []
            for i in range(num_sets):
                param_set = {key: params['best'][i] for key, params in self.parameter_specs.items()}
                best_param_sets.append(param_set)
            return best_param_sets

    # Renumber categorical data to consecutive numbers (Lightgbm requires this)
    @staticmethod
    def _renumber_cat(factor, date):
        category_mapping = {}
        max_categories = 259
        # Lightgbm can only handle max 259 different categories for one column (on GPU)
        for col in factor.columns:
            if date:
                # Set max bin size to 259
                max_rank = factor[col].max()
                bin_size = np.ceil(max_rank / max_categories) + 0.2
                max_compressed_rank = (max_rank + bin_size - 1) // bin_size
                factor[col] = np.ceil(factor[col] / bin_size)
                factor[col] = factor[col].apply(lambda x: min(x, max_compressed_rank))
                factor[col] = factor[col].replace({np.nan: -1, np.inf: max_compressed_rank}).astype(int)

            # Convert to lightgbm format, start from 0 (1, 2, 3.....)
            unique_categories = factor[col].unique()
            category_mapping[col] = {category: i for i, category in enumerate(unique_categories)}
            factor[col] = factor[col].map(category_mapping[col])
        factor = factor.astype(int)
        return factor

    # Creates the forward (by timestep self.lookahead) returns used for model training
    def _create_fwd(self):
        if self.outlier:
            if self.pred == 'sign':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'target_{self.lookahead}D'] = (
                    self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}']
                    .apply(lambda x: (1 + x).rolling(window=self.trend).apply(np.prod, raw=True) - 1)
                    .shift(-self.lookahead)
                    .apply(lambda x: 1 if x > 0 else 0)
                    .reset_index(level=0, drop=True)
                )
                self.data = remove_nan_before_end(self.data, f'target_{self.lookahead}D')
            elif self.pred == 'price':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'target_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}'].shift(-self.lookahead)
                self.data = remove_nan_before_end(self.data, f'target_{self.lookahead}D')
                condition = self.data[f'target_{self.lookahead}D'].abs() > 0.05
                self.data.loc[condition, f'target_{self.lookahead}D'] = np.nan
        else:
            if self.pred == 'sign':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'target_{self.lookahead}D'] = (
                    self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}']
                    .apply(lambda x: (1 + x).rolling(window=self.trend).apply(np.prod, raw=True) - 1)
                    .shift(-self.lookahead)
                    .apply(lambda x: 1 if x > 0 else 0)
                    .reset_index(level=0, drop=True)
                )
                self.data = remove_nan_before_end(self.data, f'target_{self.lookahead}D')
            elif self.pred == 'price':
                self.actual_return = self.data[[f'RET_{self.lookahead:02}']]
                self.data[f'target_{self.lookahead}D'] = self.data.groupby(level=self.stock)[f'RET_{self.lookahead:02}'].shift(-self.lookahead)
                self.data = remove_nan_before_end(self.data, f'target_{self.lookahead}D')

    # Creates the indices used for pretraining the model
    @staticmethod
    def _pretrain(data, pretrain_len):
        unique_dates = data.index.get_level_values('date').unique()
        days = sorted(unique_dates)
        train_start = 0
        train_end = pretrain_len - 1
        dates = data.reset_index()[['date']]
        pre_train_idx = dates[(days[train_start] <= dates.date) & (dates.date <= days[train_end])].index
        return pre_train_idx

    # Creates the indices used for set walk forward training
    @staticmethod
    def _wfo(data, n_splits, lookahead, train_period_length, test_period_length):
        unique_dates = data.index.get_level_values('date').unique()
        days = sorted(unique_dates)
        split_idx = []

        # Your existing loop to create split indices
        for i in range(n_splits):
            train_start_idx = i * test_period_length
            train_end_idx = train_start_idx + train_period_length - 1
            test_start_idx = train_end_idx + lookahead
            test_end_idx = test_start_idx + test_period_length - 1
            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        # Check if the last test_end date is not the last available date
        if days[test_end_idx] < days[-1]:
            # Use the previous test_end as the start of the new training period
            train_start_idx = test_end_idx + 1 - train_period_length
            train_end_idx = test_end_idx

            # Set the testing window to start right after the previous one ended
            test_start_idx = test_end_idx + 1
            test_end_idx = len(days) - 1  # Last available date

            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = data.reset_index()[['date']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(days[train_start] <= dates.date) & (dates.date <= days[train_end])].index
            test_idx = dates[(days[test_start] <= dates.date) & (dates.date <= days[test_end])].index
            yield train_idx, test_idx

    # Creates the indices used for expanded walk forward training
    @staticmethod
    def _ewo(data, n_splits, lookahead, train_period_length, test_period_length):
        # Extract the unique dates in the dataframe
        unique_dates = data.index.get_level_values('date').unique()
        days = sorted(unique_dates)
        split_idx = []

        # Start training from the first available data point
        train_start_idx = 0

        # Creates the indices used to split the data for training and testing
        for i in range(n_splits):
            train_end_idx = train_start_idx + i * test_period_length + train_period_length - 1
            test_start_idx = train_end_idx + lookahead
            test_end_idx = test_start_idx + test_period_length - 1

            # If the last index exceeds the number of days (index out of bounds), break the loop
            if test_end_idx >= len(days):
                test_end_idx = len(days) - 1
                split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])
                break

            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = data.reset_index()[['date']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(days[train_start] <= dates.date) & (dates.date <= days[train_end])].index
            test_idx = dates[(days[test_start] <= dates.date) & (dates.date <= days[test_end])].index
            yield train_idx, test_idx

    # Plots the distribution of predicted returns and actual returns during training in a histogram plot (this can be used to help diagnose overfitting)
    @staticmethod
    def _plot_histo(pred, ret, date):
        pred = pred[pred.columns[0]].loc[pred.index.get_level_values('date') == date]
        ret = ret.RET_01.loc[ret.index.get_level_values('date') == date]
        plt.hist(pred, bins='auto', edgecolor='black', alpha=0.5, label=f"Pred: {date}")
        plt.hist(ret, bins='auto', edgecolor='red', alpha=0.5, label=f"Real: {date}")
        plt.title('Histogram of values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    # Adds the created and prepped factors to the entire dataset that will be used for training
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Adding to {self.model_name}', animation=spinner_animation)
    def add_factor(self, factor, categorical=False, normalize=False, date=False):
        # If self.data is None then assign self.data to it, else merge the factor to self.data
        def condition_add(factor):
            if self.data is None:
                self.data = factor
            else:
                self.data = self.data.merge(factor, left_index=True, right_index=True, how='left')
                # The stupidest **** I've seen (pd.merge converts int datatypes to float????)
                if all(col in self.categorical for col in factor.columns):
                    self.data[factor.columns] = self.data[factor.columns].fillna(-9999).astype(int)

        # If the factor is categorical, then append the columns of factor to self.categorical list. This will be used to designate categorical columns for model training
        if categorical:
            self.categorical = self.categorical + factor.columns.tolist()
            # Must renumber categories for lightgbm
            if 'lightgbm' in self.model_name:
                # Make sure self.model_name has 'lightgbm' in it, or else it will not append categorical factors to the dataframe
                factor = self._renumber_cat(factor, date)
                condition_add(factor)
            # Must renumber fill NAN values for catboost
            elif 'catboost' in self.model_name:
                factor = (factor + 1) * 2
                factor = factor.fillna(-9999).astype(int)
                condition_add(factor)
        else:
            # If normalize is true, then you normalize using MinMaxScaler
            if normalize:
                factor = factor.replace([np.inf, -np.inf], np.nan)
                scaler = MinMaxScaler((-1, 1))
                # Normalization function to be applied to each stock
                def normalize(group):
                    value = scaler.fit_transform(group)
                    return pd.DataFrame(value, index=group.index, columns=group.columns)
                factor = factor.groupby(level='date').apply(normalize)
                condition_add(factor)
            else:
                # If normalize is false then just add the factor to the data
                condition_add(factor)

    # Training model with lightgbm
    def lightgbm(self):
        # Model to train (this function is created so that it can be used for gridsearch, optuna, and default training)
        def model_training(trial):
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------FUNCTIONS WITHIN MODEL_TRAINING------------------------------------------------------------------------------
            # Save beeswarm SHAP plot
            def plot_beeswarm_gbm(sv, explainer, X, key, i):
                if self.pred == 'price':
                    shap.plots.beeswarm(sv, max_display=80, show=False)
                    plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}' / f'beeswarm_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
                    plt.close()
                elif self.pred == 'sign':
                    values = explainer.shap_values(X)
                    shap.summary_plot(values, feature_names=X.columns.tolist(), max_display=80, show=False)
                    plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}' / f'beeswarm_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
                    plt.close()

            # Save waterfall SHAP plot
            def plot_waterfall_gbm(sv, key, i):
                shap.plots.waterfall(sv[0], max_display=80, show=False)
                plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}' / f'waterfall_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
                plt.close()

            # Save lightgbm gain importance
            def get_feature_gain(model):
                fi = model.feature_importance(importance_type='gain')
                return pd.Series(fi / fi.sum(), index=model.feature_name())

            # Save lightgbm split importance
            def get_feature_split(model):
                fi = model.feature_importance(importance_type='split')
                return pd.Series(fi / fi.sum(), index=model.feature_name())

            # Custom loss function
            def custom_loss(y_pred, dataset):
                y_true = dataset.get_label()
                penalty = 0.1
                grad = 2 * (y_pred - y_true)
                hess = 2 * np.ones_like(y_true)
                # Penalizing both sign predictions
                grad += penalty * (np.sign(y_pred) - np.sign(y_true))
                return grad, hess

            # Custom evaluation function
            def custom_eval(y_pred, dataset):
                y_true = dataset.get_label()
                loss = (y_pred - y_true) ** 2 + 0.2 * (np.sign(y_pred) - np.sign(y_true)) ** 2
                return 'custom_loss', loss.mean(), False

            # Saves the average loss of the training curve and learning curve in a plot
            def plot_avg_loss(folds_data):
                # Determine the maximum length across all folds
                max_len_training = max(len(fold['training']['l2']) for fold in folds_data)
                max_len_valid_1 = max(len(fold['valid_1']['l2']) for fold in folds_data)
                # Initialize cumulative loss values
                cumulative_losses = {'training': np.zeros(max_len_training), 'valid_1': np.zeros(max_len_valid_1)}
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
                avg_losses = {'training': {'l2': (cumulative_losses['training'] / counts_training).tolist()}, 'valid_1': {'l2': (cumulative_losses['valid_1'] / counts_valid_1).tolist()}}
                evals_result = {'training': avg_losses['training'], 'valid_1': avg_losses['valid_1']}
                return evals_result

            # Executes the optimization training process
            def train_model(export_key, param_name_train, param_val_train):
                # Get the parameters
                params = dict(zip(param_name_train, param_val_train))
                params.update(base_params)
                # Set up cross-validation and track time
                T = 0
                log_loss = []
                track_wfo = time.time()
                # If pretrain_len is greater than 0, pretrain the model
                if self.pretrain_len > 0:
                    # Pre-train model
                    print("-" * 60)
                    print("Pretrain model......")
                    print("-" * 60)
                    start_train = data_pretrain.index.get_level_values('date')[pretrain_idx].min().strftime('%Y-%m-%d')
                    end_train = data_pretrain.index.get_level_values('date')[pretrain_idx].max().strftime('%Y-%m-%d')
                    print(f'Pretraining from {start_train} --> {end_train}:')
                    lgb_pretrain = lgb_data_pretrain.subset(used_indices=pretrain_idx.tolist()).construct()
                    prev_model = lgb.train(params=params, train_set=lgb_pretrain, num_boost_round=1000)

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
                    print(f'Training from {start_train} --> {end_train} || Testing from {start_test} --> {end_test}:')
                    # If early stopping is set to True
                    if self.early:
                        # Select train subset save last self.valid_len for validation
                        lgb_train = lgb_data_train.subset(used_indices=train_idx.tolist()[:-self.valid_len]).construct()
                        lgb_val = lgb_data_train.subset(used_indices=train_idx.tolist()[-self.valid_len:]).construct()
                        lgb_early_stop = lgb.early_stopping(100)
                        # Early stop on MSE or binary_logloss
                        print('Start training......')
                        track_training = time.time()
                        evals = {}
                        if self.incr:
                            if self.pretrain_len>0:
                                # Pretrain the model
                                model = lgb.train(init_model=prev_model, params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=1000,
                                                  callbacks=[lgb_early_stop, lgb.record_evaluation(evals)])
                                """# Retrain on entire dataset with less num_boost_round
                                whole_set = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                                model = lgb.train(init_model=model, params=params, train_set=whole_set, num_boost_round=150)"""
                                # Store this model for next fold
                                prev_model = model
                            else:
                                if i == 0:
                                    # First model to be trained
                                    model = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=1000,
                                                      callbacks=[lgb_early_stop, lgb.record_evaluation(evals)])
                                    """# Retrain on entire dataset with less num_boost_round
                                    whole_set = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                                    model = lgb.train(init_model=model, params=params, train_set=whole_set, num_boost_round=150)"""
                                    # Store this model for next fold
                                    prev_model = model
                                else:
                                    # Model to be trained after the first model
                                    model = lgb.train(init_model=prev_model, params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=1000,
                                                      callbacks=[lgb_early_stop, lgb.record_evaluation(evals)])
                                    """# Retrain on entire dataset with less num_boost_round
                                    whole_set = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                                    model = lgb.train(init_model=model, params=params, train_set=whole_set, num_boost_round=150)"""
                                    # Store this model for next fold
                                    prev_model = model

                        else:
                            # No incremental training
                            model = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=1000, callbacks=[lgb_early_stop, lgb.record_evaluation(evals)])
                            """# Retrain on entire dataset with less num_boost_round
                            whole_set = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                            model = lgb.train(init_model=model, params=params, train_set=whole_set, num_boost_round=150)"""
                    else:
                        # No early stop training
                        # Select train subset
                        lgb_train = lgb_data_train.subset(used_indices=train_idx.tolist()).construct()
                        print('Start training......')
                        track_training = time.time()
                        if self.incr:
                            if self.pretrain_len > 0:
                                model = lgb.train(init_model=prev_model, params=params, train_set=lgb_train, num_boost_round=1000)
                                # Store this model for next fold
                                prev_model = model
                            else:
                                if i == 0:
                                    # First model to be trained
                                    model = lgb.train(params=params, train_set=lgb_train, num_boost_round=1000)
                                    # Store this model for next fold
                                    prev_model = model
                                else:
                                    model = lgb.train(init_model=prev_model, params=params, train_set=lgb_train, num_boost_round=1000)
                                    # Store this model for next fold
                                    prev_model = model
                        else:
                            # No incremental training
                            model = lgb.train(params=params, train_set=lgb_train, num_boost_round=1000)

                    # Print training time
                    print('Train time:', round(time.time() - track_training, 2), 'seconds')
                    if self.early:
                        # Capture training loss and validation loss and plot the loss for each training period if self.plot_loss is True
                        log_loss.append(evals)
                        if self.plot_loss:
                            lgb.plot_metric(evals)
                            plt.show()

                    # Capture predictions
                    test_set = data_train.iloc[test_idx]
                    test_factors = test_set[model.feature_name()]
                    test_ret = test_set[ret]
                    # Predict for price or sign
                    if self.pred == 'price':
                        print('Predicting......')
                        # Get the model predictions using different number of trees from the model
                        test_pred_ret = {str(n): model.predict(test_factors) if n == 1000 else model.predict(test_factors, num_iteration=n) for n in num_iterations}
                        print("-" * 60)
                    elif self.pred == 'sign':
                        print('Predicting......')
                        # Get the model predictions using different number of trees from the model
                        test_pred_ret = {str(n): model.predict(test_factors) if n == 1000 else model.predict(test_factors, num_iteration=n) for n in num_iterations}
                        print("-" * 60)

                    # Create a prediction dataset
                    opt_pred.append(test_ret.assign(**test_pred_ret).assign(i=i))

                    # Plot histogram plot for each training period if self.plot_hist is True
                    if self.plot_hist:
                        self._plot_histo(test_ret.assign(**test_pred_ret).assign(i=i), self.actual_return, data_train.iloc[train_idx].index.get_level_values('date')[-1] + timedelta(days=5))

                    if self.shap:
                        # Save the SHAP plots for training period at start, middle, and end
                        if i == 0 or i == n_splits // 2 or i == n_splits - 1:
                            print('Exporting beeswarm and waterfall SHAP plot......')
                            explainer = shap.TreeExplainer(model)
                            sv = explainer(test_factors)
                            plot_beeswarm_gbm(sv, explainer, test_factors, export_key, i)
                            if self.pred == 'price':
                                plot_waterfall_gbm(sv, export_key, i)

                    # Save feature gain and split
                    if i == 0:
                        gain = get_feature_gain(model).to_frame()
                        split = get_feature_split(model).to_frame()
                    else:
                        gain[i] = get_feature_gain(model)
                        split[i] = get_feature_split(model)
                    print("-" * 60)

                # If model was training with early stopping, save the average training curve and learning curve across all training periods
                if self.early and self.pred == 'price':
                    #Plot training loss and validation loss and track training time
                    eval_results = plot_avg_loss(log_loss)
                    lgb.plot_metric(eval_results)
                    plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{export_key}' / f'loss.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
                    plt.close()

                # Create the dataset that assigns the number of trees used for prediction to its respective prediction
                params = {name: val.__name__ if callable(val) else val for name, val in params.items()}
                all_pred_ret = pd.concat(opt_pred).assign(**params)

                # Compute IC or AS per day cross-sectionally
                by_day = all_pred_ret.groupby(level='date')
                if self.pred == 'price':
                    ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x[ret], x[str(n)])[0]).to_frame(n) for n in num_iterations], axis=1)
                    daily_ic_mean = list(ic_by_day.mean())
                elif self.pred == 'sign':
                    as_by_day = pd.concat([
                        by_day.apply(lambda x: accuracy_score(x[ret], x[str(n)].apply(lambda p: 1 if p > 0.5 else 0))).to_frame(n)
                        for n in num_iterations
                    ], axis=1)
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
            # ----------------------------------------------------------------------TUNING PARAMETERS----------------------------------------------------------------------------------------
            # Base params used for training
            if self.pred == 'price':
                base_params = dict(boosting='gbdt', device_type='gpu', gpu_platform_id=1, gpu_device_id=0, verbose=-1, gpu_use_dp=True, objective='regression', metric='mse', seed=42)
            elif self.pred == 'sign':
                base_params = dict(boosting='gbdt', device_type='gpu', gpu_platform_id=1, gpu_device_id=0, verbose=-1, gpu_use_dp=True, objective='binary', metric='binary_logloss', seed=42)

            # Get parameters and set num_iterations used for prediction
            params = self._get_parameters(trial)

            # self.tuning is a list so must handle this case
            if self.tuning == 'best':
                param_names = list(params[0].keys())
            else:
                param_names = list(params.keys())
            num_iterations = [150, 200, 300, 400, 500, 750, 1000]
            # This will be used for the metric dataset during training
            metric_cols = (param_names + ['time'] + ["daily_metric_" + str(n) for n in num_iterations])

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------CREATING LABELS--------------------------------------------------------------------------------------
            # Create forward returns
            self._create_fwd()
            # Get list of target returns and factors used for prediction
            ret = sorted(self.data.filter(like='target').columns)
            factors = [col for col in self.data.columns if col not in ret]

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------PRETRAIN OR NOT--------------------------------------------------------------------------------------
            # Create the lightgbm datasets used for training
            if self.pretrain_len > 0:
                # Set factors and target return data for pre_train
                data_pretrain = self.data.loc[:, factors + ret]
                pretrain_idx = self._pretrain(data=data_pretrain, pretrain_len=self.pretrain_len)
                # Print the start date and end date for pretrain period
                print("-" * 60)
                print("Pretrain: " + str(data_pretrain.iloc[pretrain_idx].index.get_level_values('date').unique()[0].date()) + " --> " + str(
                    data_pretrain.iloc[pretrain_idx].index.get_level_values('date').unique()[-1].date()))
                # Get start date and end date for train data from pretrain index (essentially add one to last pretrain index)
                start_date_train = str(data_pretrain.iloc[pretrain_idx].index.get_level_values('date').unique()[-1].date() + timedelta(days=1))
                end_date_train = str(data_pretrain.index.get_level_values('date').unique()[-1].date())
                # Print the start date and end date for train period
                print("Train: " + str(start_date_train) + " --> " + str(end_date_train))
                print("-" * 60)
                data_train = set_timeframe(self.data.loc[:, factors + ret], start_date_train, end_date_train)
                # Create binary dataset for pretrain and train
                lgb_data_pretrain = lgb.Dataset(data=data_pretrain.drop(ret, axis=1),
                                                label=data_pretrain[ret], categorical_feature=self.categorical,
                                                free_raw_data=False, params={'device_type': 'gpu'})
                lgb_data_train = lgb.Dataset(data=data_train.drop(ret, axis=1), label=data_train[ret],
                                             categorical_feature=self.categorical, free_raw_data=False, params={'device_type': 'gpu'})

                # Set sample weights for training if self.weight is True
                if self.weight:
                    # Emphasize extreme returns (model focuses more on these anomalies)
                    threshold = 0.001
                    weight_pretrain = data_pretrain[ret[0]].map(lambda x: 1 if abs(x) > threshold else 0.1).values
                    weight_train = data_train[ret[0]].map(lambda x: 1 if abs(x) > threshold else 0.1).values
                    # Emphasize negative returns (model focuses more on these negative returns)
                    """weight_pretrain = data_pretrain[ret[0]].map(lambda x: 1 if x < 0 else 0.1).values
                    weight_train = data_train[ret[0]].map(lambda x: 1 if x < 0 else 0.1).values"""
                    # Set the weight
                    lgb_data_pretrain.set_weight(weight_pretrain)
                    lgb_data_train.set_weight(weight_train)

            else:
                # Print the start date and end date for train period
                data_train = self.data.loc[:, factors + ret]
                start_date_train = str(data_train.index.get_level_values('date').unique()[0].date())
                end_date_train = str(data_train.index.get_level_values('date').unique()[-1].date())
                print("-" * 60)
                print("Train: " + str(start_date_train) + " --> " + str(end_date_train))
                print("-" * 60)

                # Create binary dataset for pretrain and train
                lgb_data_train = lgb.Dataset(data=data_train.drop(ret, axis=1), label=data_train[ret],
                                             categorical_feature=self.categorical, free_raw_data=False, params={'device_type': 'gpu'})

                # Set sample weights for training if self.weight is True
                if self.weight:
                    # Emphasize extreme returns (model focuses more on these anomalies)
                    threshold = 0.001
                    weight_train = data_train[ret[0]].map(lambda x: 1 if abs(x) > threshold else 0.1).values
                    # Emphasize negative returns (model focuses more on these negative returns)
                    """weight_train = data_train[ret[0]].map(lambda x: 1 if x < 0 else 0.1).values"""
                    # Set the weight
                    lgb_data_train.set_weight(weight_train)

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------TRAINING MODEL---------------------------------------------------------------------------------------
            # Train model with optuna, gridsearch, or default
            if self.tuning[0] == 'optuna':
                # Get param values and print the key (formatted params)
                param_vals = list(params.values())
                # Round param vals (this is to maintain consistency)
                param_vals = [round(p, 7) if isinstance(p, float) else p for p in param_vals]

                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                # Create the directory
                os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
                # Train model and return the optimization metric for optuna
                return train_model(key, param_names, param_vals)
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
                    train_model(key, param_names, param_vals)
            elif self.tuning == 'default':
                # Get param values and print the key (formatted params)
                param_vals = list(params.values())
                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                # Create the directory
                os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
                # Train model
                train_model(key, param_names, param_vals)
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
                    train_model(key, param_names, param_vals)

        # ===============================================================================================================================================================================
        # --------------------------------------------------------------------------MODEL_TRAINING()-------------------------------------------------------------------------------------
        print('List of categorical inputs:')
        print(self.categorical)
        print(f'Length: {len(self.categorical)}')
        if self.tuning[0] == 'optuna':
            # Create new directory/override directory named self.model_name
            shutil.rmtree(get_ml_result(self.live, self.model_name) / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}')
            # Create study that maximizes metric score
            study = optuna.create_study(direction="maximize")
            # Execute train
            study.optimize(model_training, n_trials=self.tuning[1])
            # Print the best model's result
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial:")
            trial = study.best_trial
            print("     Metric: {}".format(trial.value))
            print("     Params: ")
            for name, number in trial.params.items():
                print("         {}: {}".format(name, number))
            print("-" * 60)
        else:
            # Create new directory/override directory named self.model_name
            shutil.rmtree(get_ml_result(self.live, self.model_name) / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}')
            # Execute train
            model_training(None)

    def catboost(self):
        # Model to train (this function is created so that it can be used for gridsearch, optuna, and default training)
        def model_training(trial):
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------FUNCTIONS WITHIN MODEL_TRAINING------------------------------------------------------------------------------
            # Save beeswarm SHAP plot
            def plot_beeswarm_cb(sv, X, key, i):
                shap.summary_plot(sv, X, max_display=80, show=False)
                plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}' / f'beeswarm_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
                plt.close()

            # Save waterfall SHAP plot
            def plot_waterfall_cb(sv, key, i):
                shap.plots.waterfall(sv[0], max_display=80, show=False)
                plt.savefig(str(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}' / f'waterfall_{i}.png'), dpi=700, format="png", bbox_inches='tight', pad_inches=1)
                plt.close()

            # Save catboost gain importance
            def get_feature_gain(model):
                fi = model.get_feature_importance(type="FeatureImportance")
                feature_names = model.feature_names_
                return pd.Series(fi / fi.sum(), index=feature_names)

            # Save catboost gain importance
            def get_feature_split(model):
                fi = model.get_feature_importance(type="PredictionValuesChange")
                feature_names = model.feature_names_
                return pd.Series(fi / fi.sum(), index=feature_names)

            # Executes the optimization training process
            def train_model(export_key, param_name_train, param_val_train):
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
                            plot_beeswarm_cb(sv, test_factors, key, i)
                            plot_waterfall_cb(sv_wf, key, i)
                            print("-" * 60)

                    # Save feature gain and split
                    if i == 0:
                        gain = get_feature_gain(model).to_frame()
                        split = get_feature_split(model).to_frame()
                    else:
                        gain[i] = get_feature_gain(model)
                        split[i] = get_feature_split(model)
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
                    as_by_day = pd.concat([
                        by_day.apply(lambda x: accuracy_score(x[ret], x[str(n)].apply(lambda p: 1 if p > 0.5 else 0))).to_frame(n)
                        for n in num_iterations
                    ], axis=1)
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
                return train_model(key, param_names, param_vals)
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
                    train_model(key, param_names, param_vals)
            elif self.tuning == 'default':
                # Get param values and print the key (formatted params)
                param_vals = list(params.values())
                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                # Create the directory
                os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
                # Train model
                train_model(key, param_names, param_vals)
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
                    train_model(key, param_names, param_vals)

        # ===============================================================================================================================================================================
        # --------------------------------------------------------------------------MODEL_TRAINING()-------------------------------------------------------------------------------------
        # Cannot run incremental training with catboost
        assert not self.incr, 'Cannot run incremental training with catboost'
        print('List of categorical inputs:')
        print(self.categorical)
        print(f'Length: {len(self.categorical)}')
        if self.tuning[0] == 'optuna':
            # Create new directory/override directory named self.model_name
            shutil.rmtree(get_ml_result(self.live, self.model_name) / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}')
            # Create study that maximizes metric score
            study = optuna.create_study(direction="maximize")
            # Execute train
            study.optimize(model_training, n_trials=self.tuning[1])
            # Print the best model's result
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial:")
            trial = study.best_trial
            print("     Metric: {}".format(trial.value))
            print("     Params: ")
            for name, number in trial.params.items():
                print("         {}: {}".format(name, number))
            print("-" * 60)
        else:
            # Create new directory/override directory named self.model_name
            shutil.rmtree(get_ml_result(self.live, self.model_name) / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}')
            # Execute train
            model_training(None)


    def randomforest(self):
        # Model to train (this function is created so that it can be used for gridsearch, optuna, and default training)
        def model_training(trial):
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------FUNCTIONS WITHIN MODEL_TRAINING------------------------------------------------------------------------------
            # Executes the optimization training process
            def train_model(export_key, param_name_train, param_val_train):
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

                    # Select train subset save last self.valid_len for validation
                    rf_train = data_train.iloc[train_idx[:-self.valid_len]]
                    rf_val = data_train.iloc[train_idx[-self.valid_len:]]
                    # Split into Train/Validation
                    X_train = rf_train.drop(ret, axis=1)
                    y_train = rf_train[ret]
                    X_val = rf_val.drop(ret, axis=1)
                    y_val = rf_val[ret]
                    # Early stop on RMSE or AUC
                    print('Start training......')
                    track_early_stopping = time.time()
                    if self.pred == 'price':
                        model = RandomForestRegressor(**params)
                    elif self.pred == 'sign':
                        model = RandomForestClassifier(**params)
                    # Train model
                    model.fit(X_train, y_train)

                    # Print training time
                    print('Train time:', round(time.time() - track_early_stopping, 2), 'seconds')

                    # Evaluate model performance on validation set
                    train_score = model.score(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    print(f'Train Score: {train_score}, Validation Score: {val_score}')

                    # Plotting feature importance if self.plot_loss is True
                    if self.plot_loss:
                        importances = model.feature_importances_
                        indices = np.argsort(importances)
                        plt.title('Feature Importances')
                        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                        plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
                        plt.xlabel('Relative Importance')
                        plt.show()

                    # Capture predictions
                    test_set = data_train.iloc[test_idx, :]
                    test_factors = test_set.loc[:, factors]
                    test_ret = test_set.loc[:, ret]
                    # Predict for price or sign
                    print('Predicting......')
                    if self.pred == 'price':
                        # Get the model predictions
                        test_pred_ret = model.predict(test_factors)
                    elif self.pred == 'sign':
                        # Get the model predictions
                        test_pred_ret = model.predict_proba(test_factors)[:, 1]

                    # Create a DataFrame for predictions
                    pred_col_name = '0'
                    test_ret[pred_col_name] = test_pred_ret
                    # Append the prediction DataFrame
                    opt_pred.append(test_ret.assign(i=i))

                    # Plot histogram plot for each training period if self.plot_hist is True
                    if self.plot_hist:
                        self._plot_histo(test_ret, self.actual_return, data_train.iloc[train_idx].index.get_level_values('date')[-1] + timedelta(days=5))

                    print("-" * 60)

                # Create the dataset with model predictions
                params = {key: value.__name__ if callable(value) else value for key, value in params.items()}
                all_pred_ret = pd.concat(opt_pred).assign(**params)

                # Compute IC or AS per day cross-sectionally
                by_day = all_pred_ret.groupby(level='date')
                if self.pred == 'price':
                    # Spearman Rank Correlation
                    ic_by_day = by_day.apply(lambda x: spearmanr(x[ret], x['0'])[0]).to_frame('0')
                    daily_ic_mean = ic_by_day.mean().tolist()
                elif self.pred == 'sign':
                    # Accuracy Score
                    as_by_day = by_day.apply(lambda x: accuracy_score(x[ret], x['0'].map(lambda p: 1 if p > 0.5 else 0))).to_frame('0')
                    daily_as_mean = as_by_day.mean().tolist()

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
            # ----------------------------------------------------------------------TUNING PARAMETERS----------------------------------------------------------------------------------------
            # Base params used for training
            base_params = {'n_jobs': -1, 'random_state': 42}

            # Get parameters and set num_iterations used for prediction
            params = self._get_parameters(self)
            param_names = list(params.keys())
            # This will be used for the metric dataset during training
            metric_cols = (param_names + ['time'] + ['daily_metric_0'])

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------CREATING LABELS--------------------------------------------------------------------------------------
            # Create forward returns
            self._create_fwd()
            # Get list of target returns and factors used for prediction
            ret = sorted(self.data.filter(like='target').columns)
            factors = [col for col in self.data.columns if col not in ret]

            # Get start date and end date for train data
            data_train = self.data.loc[:, factors + ret]
            # Fillna with -9999 because Random Forest cannot handle NAN values
            data_train = data_train.fillna(-9999)
            start_date_train = str(data_train.index.get_level_values('date').unique()[0].date())
            end_date_train = str(data_train.index.get_level_values('date').unique()[-1].date())
            # Print the start date and end date for train period
            print("Train: " + str(start_date_train) + " --> " + str(end_date_train))
            print("-" * 60)

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
                return train_model(key, param_names, param_vals)
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
                    train_model(key, param_names, param_vals)
            elif self.tuning == 'default':
                # Get param values and print the key (formatted params)
                param_vals = list(params.values())
                key = '_'.join([str(float(p)) for p in param_vals])
                print(f'Key: {key}')
                # Create the directory
                os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
                # Train model
                train_model(key, param_names, param_vals)
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
                    train_model(key, param_names, param_vals)

        # ===============================================================================================================================================================================
        # --------------------------------------------------------------------------MODEL_TRAINING()-------------------------------------------------------------------------------------
        # Cannot run incremental training with catboost
        assert not self.incr, 'Cannot run incremental training with random forest'
        assert not self.early, 'Cannot run early stopping with random forest'
        assert not self.shap, 'Cannot plot SHAP for random forest'
        print('List of categorical inputs:')
        print(self.categorical)
        print(f'Length: {len(self.categorical)}')
        if self.tuning[0] == 'optuna':
            # Create new directory/override directory named self.model_name
            shutil.rmtree(get_ml_result(self.live, self.model_name) / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}')
            # Create study that maximizes metric score
            study = optuna.create_study(direction="maximize")
            # Execute train
            study.optimize(model_training, n_trials=self.tuning[1])
            # Print the best model's result
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial:")
            trial = study.best_trial
            print("     Metric: {}".format(trial.value))
            print("     Params: ")
            for name, number in trial.params.items():
                print("         {}: {}".format(name, number))
            print("-" * 60)
        else:
            # Create new directory/override directory named self.model_name
            shutil.rmtree(get_ml_result(self.live, self.model_name) / f'{self.model_name}', ignore_errors=True)
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}')
            # Execute train
            model_training(None)
            
