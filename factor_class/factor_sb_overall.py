from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBOverall(Factor):
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
        self.fama_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_fama.parquet.brotli')
        overall_df = yf.download(['IWR', 'IVV', 'QQQ', 'SPY', 'VNQ'], start=self.start, end=self.end)
        overall_df = overall_df.stack().swaplevel().sort_index()
        overall_df.index.names = ['ticker', 'date']
        overall_df = overall_df.astype(float)
        T = [1]
        overall_df = create_return(overall_df, T)
        overall_df = overall_df.drop(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        overall_df = overall_df.unstack('ticker').swaplevel(axis=1)
        overall_df.columns = ['_'.join(col).strip() for col in overall_df.columns.values]
        self.overall_data = overall_df
        self.overall_data = pd.concat([self.overall_data, self.fama_data['RF']], axis=1)
        self.overall_data = self.overall_data.loc[self.start:self.end]
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
            windows = [30, 60]
            for window in windows:
                betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.overall_data, factor_col=self.factor_col, window=window, name=f'{t:02}_OVERALL', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data
