from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorSBInverse(Factor):
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
        self.factor_data = pd.read_parquet(get_parquet(self.live) / 'data_price.parquet.brotli')
        self.risk_free = pd.read_parquet(get_parquet(self.live) / 'data_rf.parquet.brotli')

        # Read in trade_live market data
        overall_df = get_data_fmp(ticker_list=['IWR', 'IVV', 'QQQ', 'SPY', 'VNQ'], start=self.start, current_date=self.end)
        T = [1]
        overall_df = overall_df[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
        overall_df = overall_df.rename(columns={'Adj Close': 'Close'})

        overall_df = create_return(overall_df, T)
        overall_df = overall_df.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        overall_df = overall_df.unstack('ticker').swaplevel(axis=1)
        overall_df.columns = ['_'.join(col).strip() for col in overall_df.columns.values]

        self.overall_data = overall_df
        self.overall_data = pd.concat([self.overall_data, self.risk_free['RF']], axis=1)
        self.overall_data = self.overall_data.loc[self.start:self.end]
        self.overall_data['RF'] = self.overall_data['RF'].ffill()
        self.overall_data = self.overall_data.fillna(0)
        self.factor_col = self.overall_data.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        T = [1]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        for t in T:
            ret = f'RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [21, 126]
            for window in windows:
                # betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.overall_data, factor_col=self.factor_col, window=window, name=f'inv_spy_{t:02}', ret=ret)
                # prefixes = ['IWR', 'IVV', 'QQQ', 'SPY', 'VNQ']
                # betas = betas[[col for col in betas.columns if any(col.startswith(prefix) for prefix in prefixes)]]
                # betas = betas.rdiv(1)

                betas = rolling_ols_parallel(data=splice_data, ret=ret, factor_data=self.overall_data, factor_cols=self.factor_col.tolist(), window=window, name=f'spy_{t:02}')
                splice_data = splice_data.join(betas)

        return splice_data