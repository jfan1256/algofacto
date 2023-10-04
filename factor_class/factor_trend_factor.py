from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorTrendFactor(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
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
        super().__init__(file_name, skip, start, end, stock, batch_size, splice_size, group, join, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        lag_data = self.factor_data
        # Create Lag Price Rolling Mean Predictors
        self.T = [3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
        for lag in self.T:
            lag_data[lag] = lag_data['Close'].rolling(window=lag).mean()
        lag_data = lag_data.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        self.lag = lag_data
        self.lag = set_timeframe(self.lag, self.start, self.end)

    @ray.remote
    def function(self, splice_data):
        T = [1]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        ret = f'RET_01'
        # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
        window = 120
        collect = []
        for indicator, df in splice_data.groupby(splice_data.index.names[0], group_keys=False):
            lag_data = get_stock_data(self.lag, indicator)
            lag_data = lag_data.unstack(self.group).swaplevel(axis=1)
            lag_data.columns = ['_'.join(str(col)).strip() for col in lag_data.columns.values]
            lag_data = lag_data.fillna(0)
            factor_col = lag_data.columns
            model_data = df[[ret]].merge(lag_data, on='date').dropna()
            collect_betas = []

            # Run Univariate Regression on each Lagged Price Rolling Mean
            for i, col in enumerate(factor_col):
                rolling_ols = RollingOLS(endog=model_data[ret], exog=sm.add_constant(model_data[col]), window=window)
                factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})
                # Compute predictions of stock's return
                beta_coef = factor_model[col]
                beta_coef = beta_coef.rolling(window=self.T[i]).mean()
                collect_betas.append(beta_coef)

            betas = pd.concat(collect_betas, axis=1)
            result = (betas * lag_data).sum(axis=1)
            result = result.to_frame()
            result[self.group] = indicator
            result = result.reset_index().set_index([self.group, 'date'])
            result.columns = ['TrendFactor']
            collect.append(result)
        splice_data = pd.concat(collect, axis=0)
        return splice_data