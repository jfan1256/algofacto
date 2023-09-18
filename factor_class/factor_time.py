from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorTime(Factor):
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
        price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.factor_data = price_data

    @ray.remote
    def function(self, splice_data):
        splice_data['Month'] = splice_data.index.get_level_values('date').month
        splice_data['Weekday'] = splice_data.index.get_level_values('date').weekday
        # splice_data['Halloween'] = splice_data.index.get_level_values('date').map(lambda x: 1 if 5 <= x.month <= 10 else 0)
        return splice_data
