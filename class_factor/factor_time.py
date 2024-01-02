from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorTime(Factor):
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
        price_data = pd.read_parquet(get_parquet(self.live) / 'data_price.parquet.brotli')
        self.factor_data = price_data

    @ray.remote
    def function(self, splice_data):
        splice_data['month'] = splice_data.index.get_level_values('date').month
        splice_data['weekday'] = splice_data.index.get_level_values('date').weekday
        splice_data['is_halloween'] = splice_data.index.get_level_values('date').map(lambda x: 1 if 5 <= x.month <= 10 else 0)
        splice_data['is_january'] = (splice_data.index.get_level_values('date').month == 1).astype(int)
        splice_data['is_friday'] = (splice_data.index.get_level_values('date').dayofweek == 4).astype(int)
        splice_data['last_day'] = splice_data.index.get_level_values('date').to_period('M').to_timestamp(how='end')
        splice_data['begin_last_week'] = splice_data['last_day'] - pd.Timedelta(days=5)
        splice_data['in_last_week'] = (splice_data.index.get_level_values('date') >= splice_data['begin_last_week']) & (splice_data.index.get_level_values('date') <= splice_data['last_day'])
        splice_data['is_quarter_end_week'] = (splice_data['in_last_week'] & splice_data.index.get_level_values('date').month.isin([3, 6, 9, 12])).astype(int)
        splice_data['is_year_end_week'] = (splice_data['in_last_week'] & (splice_data.index.get_level_values('date').month == 12)).astype(int)
        splice_data = splice_data.drop(columns=['last_day', 'begin_last_week', 'in_last_week'], axis=1)

        # day = splice_data.index.get_level_values('date').day
        # splice_data['is_turn_of_month'] = ((day <= 3) | (day >= 28)).astype(int)
        # last_day_of_month = splice_data.index.get_level_values('date').to_period('M').to_timestamp('M')
        # days_to_month_end = (last_day_of_month - splice_data.index.get_level_values('date')).days
        # splice_data['is_month_end_week'] = (days_to_month_end < 5).astype(int)
        # weekday = splice_data.index.get_level_values('date').weekday
        # splice_data['is_monday'] = (weekday == 0).astype(int)
        return splice_data
