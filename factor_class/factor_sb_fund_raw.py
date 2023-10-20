from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBFundRaw(Factor):
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
        self.fama_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_fama.parquet.brotli')
        self.fund_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_fund_raw_q.parquet.brotli')
        self.factor_col = self.fund_data.columns[:-1]
        self.factor_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_price.parquet.brotli')
        self.factor_data = self.factor_data.reindex(self.fund_data.index, method='ffill')

    @ray.remote
    def function(self, splice_data):
        splice_data = create_return(splice_data, [1])

        t = 1
        ret = f'RET_{t:02}'
        name = 'FUND_RAW'

        # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
        windows = [9]
        for window in windows:
            betas = []
            for ticker, df in splice_data.groupby('ticker', group_keys=False):
                fund_data = get_stock_data(self.fund_data, ticker)
                fund_data = fund_data.reset_index('ticker').drop('ticker', axis=1)
                # Add risk-free rate
                fund_data = pd.concat([fund_data, self.fama_data['RF']], axis=1)

                # Set time frame
                fund_data = fund_data.loc[self.start:self.end]
                fund_data = fund_data.fillna(0)

                model_data = df[[ret]].merge(fund_data, on='date').dropna()
                model_data[ret] -= model_data.RF
                rolling_ols = RollingOLS(endog=model_data[ret], exog=sm.add_constant(model_data[self.factor_col]), window=window)
                factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

                # Compute predictions of ticker's return
                alpha = factor_model['ALPHA']
                beta_coef = factor_model[self.factor_col]
                factor_ret = model_data[self.factor_col]

                predictions = []
                for index, row in factor_ret.iterrows():
                    predictions.append(row @ beta_coef.loc[index] + alpha.loc[index])

                result = factor_model.assign(ticker=ticker).set_index('ticker', append=True).swaplevel()
                result['PRED'] = predictions
                betas.append(result)

            betas = pd.concat(betas)
            betas = betas.rename(columns=lambda x: f'{x}_{name}_{window:02}')
            splice_data = splice_data.join(betas)

        return splice_data