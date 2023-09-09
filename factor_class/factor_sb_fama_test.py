from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBFamaTest(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 ticker: List[str] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.fama_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_fama.parquet.brotli')
        self.fama_data = self.fama_data.loc[self.start:self.end]
        self.fama_data = self.fama_data.fillna(0)
        self.factor_col = self.fama_data.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        T = [1, 6, 30]
        splice_data = create_return(splice_data, T)
        for t in T:
            ret = f'RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [60]
            for window in windows:
                betas = rolling_ols_sb(price=splice_data, factor_data=self.fama_data, factor_col=self.factor_col, window=window, name=f'{t:02}_FAMA_TEST', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data