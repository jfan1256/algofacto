from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor


class FactorRetComp(Factor):
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

    @ray.remote
    def function(self, splice_data):
        T = [1, 5, 21, 126, 252]
        splice_data = create_return(splice_data, windows=[1])

        for t in T:
            splice_data[f'ret_comp_{t:01}'] = splice_data.groupby('permno')['RET_01'].rolling(window=t).apply(lambda x: (1 + x).prod() - 1, raw=True).reset_index(level=0, drop=True)

        splice_data = splice_data.drop('RET_01', axis=1)
        return splice_data
