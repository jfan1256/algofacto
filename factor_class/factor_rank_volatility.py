from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRankVolatility(Factor):
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

    @ray.remote
    def function(self, splice_data):
        T = [1, 2, 3, 4, 5, 10]
        splice_data = splice_data.fillna(0)
        ret = create_return(splice_data, windows=T)
        for t in T:
            splice_data[f'VOLATILITY_{t:02}'] = ret.groupby(self.group)[f'RET_{t:02}'].rolling(window=60).std().reset_index(level=0, drop=True)
            splice_data[f'RANK_VOLATILITY_{t:02}'] = splice_data[f'VOLATILITY_{t:02}'].groupby('date').rank()
            splice_data = splice_data.drop(f'RET_{t:02}', axis=1)
            splice_data = splice_data.drop(f'VOLATILITY_{t:02}', axis=1)
        return splice_data
