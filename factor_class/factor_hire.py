from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorHire(Factor):
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
        columns = ['emp']
        hire = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli', columns=columns)
        hire = get_stocks_data(hire, self.stock)
        def compute_hire(group):
            group['hire'] = (group['emp'] - group['emp'].shift(12)) / (0.5 * (group['emp'] + group['emp'].shift(12)))
            return group

        hire = hire.groupby('permno').apply(compute_hire).reset_index(level=0, drop=True)

        # Replace 'hire' with 0 where 'emp' or its lag by 12 periods is NaN
        hire.loc[hire['emp'].isna() | hire['emp'].shift(12).isna(), 'hire'] = 0
        self.factor_data = hire[['hire']]
