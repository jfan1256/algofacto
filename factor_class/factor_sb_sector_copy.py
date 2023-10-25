from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBSectorCopy(Factor):
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
        sector_df = yf.download(['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU'], start=self.start, end=self.end)
        sector_df = sector_df.stack().swaplevel().sort_index()
        sector_df.index.names = ['ticker', 'date']
        sector_df = sector_df.astype(float)
        T = [1]
        sector_df = create_return(sector_df, T)
        sector_df = sector_df.drop(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        sector_df = sector_df.unstack('ticker').swaplevel(axis=1)
        sector_df.columns = ['_'.join(col).strip() for col in sector_df.columns.values]

        # # Execute Rolling PCA
        # window_size = 60
        # num_components = 5
        # sector_df = rolling_pca(data=sector_df, window_size=window_size, num_components=num_components, name='sector')

        self.sector_data = sector_df
        self.sector_data = self.sector_data.loc[self.start:self.end]
        self.sector_data = self.sector_data.fillna(0)
        self.factor_col = self.sector_data.columns

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
                betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.sector_data, factor_col=self.factor_col, window=window, name=f'sector_{t:02}', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data