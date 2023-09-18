from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorVolatility(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 ticker: Optional[Union[List[str], str]] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')

    @ray.remote
    def function(self, splice_data):
        T = [1, 2, 3, 4, 5, 10, 20, 40, 60, 120, 210]
        ret = create_return(splice_data, windows=T)
        for t in T:
            splice_data[f'Volatility_{t:02}'] = ret.groupby('ticker')[f'RET_{t:02}'].rolling(window=60).std().reset_index(level=0, drop=True)
            splice_data = splice_data.drop(f'RET_{t:02}', axis=1)
        return splice_data

