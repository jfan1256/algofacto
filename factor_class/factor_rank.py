from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRank(Factor):
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

    @ray.remote
    def function(self, splice_data):
        T = [1, 6, 30]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)
        for t in T:
            splice_data[f'RANK_{t:02}'] = splice_data[f'RET_{t:02}'].groupby('date').rank()
            splice_data = splice_data.drop([f'RET_{t:02}'], axis=1)
        return splice_data
