from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBInd(Factor):
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
        self.factor_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_price.parquet.brotli')
        self.fama_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_fama.parquet.brotli')
        ind_df = yf.download(['KBE', 'XBI', 'KCE', 'KIE', 'XME', 'XOP', 'XPH', 'XRT', 'XSD'], start=self.start, end=self.end)
        ind_df = ind_df.stack().swaplevel().sort_index()
        ind_df.index.names = ['ticker', 'date']
        ind_df = ind_df.astype(float)
        T = [1]
        ind_df = create_return(ind_df, T)
        ind_df = ind_df.drop(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        ind_df = ind_df.unstack('ticker').swaplevel(axis=1)
        ind_df.columns = ['_'.join(col).strip() for col in ind_df.columns.values]

        # # Execute Rolling PCA
        # window_size = 60
        # num_components = 5
        # sector_df = rolling_pca(data=sector_df, window_size=window_size, num_components=num_components, name='sector')

        self.sector_data = ind_df
        self.sector_data = pd.concat([self.sector_data, self.fama_data['RF']], axis=1)
        self.sector_data = self.sector_data.loc[self.start:self.end]
        self.sector_data = self.sector_data.fillna(0)
        self.factor_col = self.sector_data.columns[:-1]

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
                # betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.sector_data, factor_col=self.factor_col, window=window, name=f'{t:02}_IND', ret=ret)
                betas = rolling_ols_parallel(data=splice_data, ret=ret, factor_data=self.sector_data, factor_cols=self.factor_col.tolist(), window=window, name=f'ind_{t:02}')
                splice_data = splice_data.join(betas)

        return splice_data
