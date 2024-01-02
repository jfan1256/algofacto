from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor


class FactorTrendFactor(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 live: bool = None,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 stock: Optional[Union[List[str], str]] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 join: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(live, file_name, skip, start, end, stock, batch_size, splice_size, group, join, general, window)
        factor_data = pd.read_parquet(get_parquet(self.live) / 'data_price.parquet.brotli')
        factor_data = get_stocks_data(factor_data, self.stock)
        lag_data = factor_data.copy(deep=True)
        # Create Lag Price Rolling Mean Predictors
        T = [1, 21, 126, 252]
        for lag in T:
            # Calculate rolling mean
            rolling_mean = lag_data.groupby('permno')['Close'].rolling(window=lag).mean().reset_index(level=0, drop=True)
            # Normalize the rolling mean by the daily closing price
            normalized_rolling_mean = rolling_mean / lag_data['Close']
            # Store the normalized rolling mean in the dataframe
            lag_data[f'ma_{lag}'] = normalized_rolling_mean

        # Convert the start date to a datetime object
        start = datetime.strptime(self.start, '%Y-%m-%d')
        # Add two years to the start date
        start = start + timedelta(days=252 * 2)
        # Convert the end date back to a string in the desired format
        start = start.strftime('%Y-%m-%d')

        lag_data = lag_data.replace([np.inf, -np.inf], np.nan)
        lag_data = lag_data.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        columns = lag_data.columns
        lag_data = set_timeframe(lag_data, start, self.end)
        lag_data = lag_data.unstack('permno')
        lag_data = lag_data.fillna(0)

        factor_data = create_return(factor_data, windows=[1])
        factor_data = factor_data[[f'RET_01']]
        factor_data = set_timeframe(factor_data, start, self.end)
        factor_data = factor_data['RET_01'].unstack('permno')
        factor_data = factor_data.fillna(0)

        betas = {}
        dates = factor_data.index

        # Iterate over each moving average column
        for ma_column in columns:
            betas_for_ma = []
            print(f'\nProcessing: {ma_column} column')
            # For each date in our dataset
            for date in dates:
                # Extract the dependent variable (returns) for the given date
                y = factor_data.loc[date]
                X = lag_data[ma_column].loc[date]
                model = sm.OLS(y, sm.add_constant(X), missing='drop').fit()
                betas_for_ma.append(model.params[date])

            # Store the betas in the dictionary
            betas[ma_column] = betas_for_ma

        # Convert the dictionary to a DataFrame
        rolling_betas = pd.DataFrame(betas, index=dates)
        rolling_betas = rolling_betas.rolling(window=21).mean()
        total = []
        for col in rolling_betas.columns:
            result = lag_data[col].multiply(rolling_betas[col], axis=0).stack().swaplevel().to_frame()
            result.columns = [col]
            total.append(result)
        total = pd.concat(total, axis=1)
        total = total.sort_index(level=['permno', 'date'])
        total['trend_factor'] = total.sum(axis=1)
        total['trend_factor'] = total['trend_factor'] - total['trend_factor'].mean()
        self.factor_data = total[['trend_factor']]


    """@ray.remote
    def function(self, splice_data):
        betas = {}
        dates = self.factor_data.index
        # Iterate over each moving average column
        for ma_column in self.columns:
            betas_for_ma = []

            # For each date in our dataset
            for date in dates:
                # Extract the dependent variable (returns) for the given date
                y = self.factor_data.loc[date]
                X = self.lag_data[ma_column].loc[date]
                model = sm.OLS(y, sm.add_constant(X), missing='drop').fit()
                betas_for_ma.append(model.params[date])

            # Store the betas in the dictionary
            betas[ma_column] = betas_for_ma

        # Convert the dictionary to a DataFrame
        rolling_betas = pd.DataFrame(betas, index=dates)
        # Calculate E[B]
        rolling_betas = rolling_betas.rolling(window=60).mean()

        # Multiple E[B] * MA per column
        total = []
        for col in rolling_betas.columns:
            result = self.lag_data[col].multiply(rolling_betas[col], axis=0).stack().swaplevel().to_frame()
            result.columns = [col]
            total.append(result)
        total = pd.concat(total, axis=1)
        total = total.sort_index(level=['permno', 'date'])

        # Calculate TrendFactor
        total['TrendFactor'] = total.sum(axis=1)
        # Center around 0
        total['TrendFactor'] = total['TrendFactor'] - total['TrendFactor'].mean()
        splice_data = total
        return splice_data"""