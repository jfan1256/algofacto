from class_model.model_train import ModelTrain
from core.operation import *

from itertools import product
from scipy.stats import spearmanr
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

import os
import time
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

class ModelRandomforest(ModelTrain):
    def __init__(self, 
                 live: bool = None, 
                 model_name: str = None, 
                 tuning: [str, int] = None,
                 plot_loss: bool = False, 
                 plot_hist: bool = False, 
                 pred: str = 'price',
                 stock: str = None, 
                 lookahead: int = 1, 
                 trend: int = 0, 
                 opt: str = None, 
                 outlier: bool = False, 
                 train_len: int = None,
                 valid_len: int = None, 
                 test_len: int = None, 
                 **kwargs
                 ):

        '''
        live (bool): Get historical data or live data
        model_name (str): Model name
        tuning (str): Type of parameter to use (i.e., default, optuna, etc.)
        plot_loss (bool): Plot training and validation curve after each window training or not
        plot_hist (bool): Plot actual returns and predicted returns after each window training or not
        pred (str): Predict for price returns or price movement
        stock (str): Name of index for stocks ('permno' or 'ticker')
        lookahead (int): Lookahead period to predict for
        trend (int): Size of rolling window to calculate trend (for price movement predictions)
        opt (str): Type of training optimization ('ewo' or 'wfo')
        outlier (bool): Handle outlier data in label data or not
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
        self.plot_loss = plot_loss
        self.plot_hist = plot_hist
        self.pred = pred
        self.stock = stock
        self.lookahead = lookahead
        self.trend = trend
        self.opt = opt
        self.outlier = outlier
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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------EXECUTION CODE-------------------------------------------------------------------------------
    # Model Training Execution
    def randomforest(self, export_key, param_name_train, param_val_train, base_params, data_train, ret, factors, metric_cols):
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
    # -------------------------------------------------------------------------------INHERITED FUNCTION------------------------------------------------------------------------------
    # Model to train (this function is created so that it can be used for gridsearch, optuna, and default training)
    def train_option(self, trial):
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
            return self.randomforest(key, param_names, param_vals, base_params, data_train, ret, factors, metric_cols)
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
                self.randomforest(key, param_names, param_vals, base_params, data_train, ret, factors, metric_cols)
        elif self.tuning == 'default':
            # Get param values and print the key (formatted params)
            param_vals = list(params.values())
            key = '_'.join([str(float(p)) for p in param_vals])
            print(f'Key: {key}')
            # Create the directory
            os.makedirs(get_ml_result(self.live, self.model_name) / f'{self.model_name}' / f'params_{key}')
            # Train model
            self.randomforest(key, param_names, param_vals, base_params, data_train, ret, factors, metric_cols)
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
                self.randomforest(key, param_names, param_vals, base_params, data_train, ret, factors, metric_cols)