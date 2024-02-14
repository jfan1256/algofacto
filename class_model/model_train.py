import shutil
import os
import optuna

from sklearn.preprocessing import MinMaxScaler
from typing import Optional

from core.operation import *

import warnings

warnings.filterwarnings('ignore')

class ModelTrain:
    def __init__(self,
                 live: bool = None,
                 model_name: str = None,
                 tuning: [str, int] = None,
                 pred: str = 'price',
                 stock: str = None,
                 lookahead: int = 1,
                 trend: int = 0,
                 opt: str = None,
                 outlier: bool = False,
                 pretrain_len: Optional[int] = None,
                 train_len: int = None,
                 **kwargs):

        '''
        live (bool): Get historical data or live data
        model_name (str): Model name (lightgbm, randomforest, or catboost)
        tuning (str): Type of parameter to use (i.e., default, optuna, etc.)
        pred (str): Predict for price returns or price movement
        stock (str): Name of index for stocks ('permno' or 'ticker')
        lookahead (int): Lookahead period to predict for
        trend (int): Size of rolling window to calculate trend (for price movement predictions)
        opt (str): Type of training optimization ('ewo' or 'wfo')
        outlier (bool): Handle outlier data in label data or not
        pretrain_len (int): Pretrain length for model training
        train_len (int): Train length for model training
        test_len (int): Prediction length for model training
        kwargs (dict): Model parameters to feed into model
        '''

        self.data = None
        self.live = live
        self.model_name = model_name
        self.categorical = []
        self.tuning = tuning
        self.pred = pred
        self.stock = stock
        self.lookahead = lookahead
        self.trend = trend
        self.opt = opt
        self.outlier = outlier
        self.pretrain_len = pretrain_len
        self.train_len = train_len
        self.actual_return = None
        self.parameter_specs = kwargs

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------PLOT---------------------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------GET PARAMETERS---------------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------ADD FACTORS (FEATURES FOR TRAINING)------------------------------------------------------------------
    # Renumber categorical data to consecutive numbers (Lightgbm requires this)
    @staticmethod
    def _renumber_cat(factor, compress):
        category_mapping = {}
        max_categories = 259
        # Lightgbm can only handle max 259 different categories for one column (on GPU)
        for col in factor.columns:
            if compress:
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

    # Adds the created and prepped factors to the entire dataset that will be used for training
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Adding to {self.model_name}', animation=spinner_animation)
    def add_factor(self, factor, categorical=False, normalize=None, impute=None, compress=False):
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
                factor = self._renumber_cat(factor, compress)

            # Must renumber fill NAN values for catboost
            elif 'catboost' in self.model_name:
                factor = (factor + 1) * 2
                factor = factor.fillna(-9999).astype(int)

            # Add factor
            condition_add(factor)
        else:
            # Min-Max Normalization
            if normalize == 'min_max_normalize':
                def minmax_normalize(group):
                    value = scaler.fit_transform(group)
                    return pd.DataFrame(value, index=group.index, columns=group.columns)

                # Remove infinite values
                factor = factor.replace([np.inf, -np.inf], np.nan)
                # Min-Max Scalar Normalization
                scaler = MinMaxScaler((-1, 1))
                # Normalize dataset
                factor = factor.groupby(level='date').apply(minmax_normalize).reset_index(level=0, drop=True)
                factor = factor.sort_index(level=['permno', 'date'])

            # Rank Normalization
            elif normalize == 'rank_normalize':
                # Rank-normalization function to be applied to each stock
                def rank_normalize(group):
                    # Separate the RET_01 column if it exists in the group
                    if 'RET_01' in group.columns:
                        ret_01 = group[['RET_01']]
                        group = group.drop(columns=['RET_01'], axis=1)
                    else:
                        ret_01 = None
                    # Rank the remaining data
                    ranks = group.rank(method='average', na_option='keep')
                    # Scale ranks to [-1, 1] range
                    min_rank, max_rank = ranks.min(), ranks.max()
                    scaled_ranks = -1 + 2.0 * (ranks - min_rank) / (max_rank - min_rank)
                    scaled_ranks_df = pd.DataFrame(scaled_ranks, index=group.index, columns=group.columns)
                    # Re-include the RET_01 column if it was initially present
                    if ret_01 is not None:
                        scaled_ranks_df = pd.concat([ret_01, scaled_ranks_df], axis=1)
                    return scaled_ranks_df

                # Remove infinite values
                factor = factor.replace([np.inf, -np.inf], np.nan)
                # Normalize dataset
                factor = factor.groupby(level='date').apply(rank_normalize).reset_index(level=0, drop=True)
                factor = factor.sort_index(level=['permno', 'date'])

            # Impute missing data
            if impute == "cross_median":
                # Cross-sectional median imputation
                daily_median = factor.groupby(level='date').transform('median')
                factor = factor.fillna(daily_median)

            # Add factor
            condition_add(factor)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------CREATE FORWARD RETURNS (LABELS FOR TRAINING-------------------------------------------------------------
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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------CREATE PRETRAIN, TRAIN, TEST IDX FOR TRAINING---------------------------------------------------------------
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

        # Check if the last test_end date is not the last available date (this handles cases where the len(test_period) != test_period_length)
        if days[test_end_idx] < days[-1]:
            # Use the previous test_end as the start of the new training period
            train_start_idx = test_end_idx + 1 - train_period_length
            train_end_idx = test_end_idx

            # Set the testing window to start right after the previous one ended
            test_start_idx = test_end_idx + 1
            test_end_idx = len(days) - 1

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

            # If test_start_idx exceeds the number of days (index out of bounds), break the loop
            if test_start_idx >= len(days):
                break

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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------INHERITED FUNCTION (FOR STUDENT)--------------------------------------------------------------------
    # Model to train (this function is created so that it can be used for gridsearch, optuna, and default training)
    def train_option(self, trial):
        return

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------EXECUTE TRAINING------------------------------------------------------------------------------
    # Execute train function
    def exec_train(self):
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
            study.optimize(self.train_option, n_trials=self.tuning[1])
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
            self.train_option(None)
