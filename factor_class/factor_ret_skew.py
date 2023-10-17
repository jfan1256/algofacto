from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRetSkew(Factor):
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
        T = [1]
        splice_data = create_return(splice_data, windows=T)
        splice_data['time_avail_m'] = splice_data.index.get_level_values('date').to_period('M')
        splice_data = splice_data.groupby([self.group, 'time_avail_m']).agg(ndays=('RET_01', 'size'), ret_skew=('RET_01', 'skew')).reset_index()
        splice_data.loc[splice_data['ndays'] < 15, 'ret_skew'] = None

        # splice_data = splice_data.reset_index()
        # splice_data = pd.merge(splice_data, skew_df[[self.group, 'time_avail_m', 'ret_skew']], on=[self.group, 'time_avail_m'], how='left')
        # splice_data = splice_data.set_index([self.group, 'date'])
        # splice_data = splice_data[['ret_skew']]

        # Convert the monthly period back to a datetime timestamp.
        splice_data['time_avail_m'] = splice_data['time_avail_m'].dt.to_timestamp()
        splice_data['time_avail_m'] = splice_data['time_avail_m'] + pd.offsets.MonthBegin(1)
        splice_data = splice_data.reset_index()
        splice_data = splice_data.rename(columns={'time_avail_m': 'date'})
        splice_data = splice_data.set_index([self.group, 'date'])
        splice_data = splice_data[~splice_data.index.duplicated(keep='first')]
        return splice_data[['ret_skew']]
