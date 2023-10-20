from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRetMax(Factor):
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

    @ray.remote
    def function(self, splice_data):
        splice_data = create_return(splice_data, [1])
        splice_data['time_avail_m'] = splice_data.index.get_level_values('date').to_period('M')
        splice_data['time_avail_m'] = splice_data['time_avail_m'].dt.to_timestamp() + pd.DateOffset(months=1)
        splice_data['ret_max'] = splice_data.groupby([self.group, 'time_avail_m'])['RET_01'].transform('max')
        splice_data = splice_data.reset_index()
        splice_data = splice_data.drop('date', axis=1)
        splice_data = splice_data.rename(columns={'time_avail_m': 'date'})
        splice_data = splice_data.set_index([self.group, 'date'])
        splice_data = splice_data[~splice_data.index.duplicated(keep='first')]
        splice_data = splice_data[['ret_max']]
        return splice_data
        