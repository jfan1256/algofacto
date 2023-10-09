from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorAccrual(Factor):
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
        columns = ['txpq', 'actq', 'dlcq', 'lctq', 'dpq', 'atq', 'dlcchy', 'cheq']
        accrual = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        accrual = get_stocks_data(accrual, self.stock)
        # Replace missing values in 'txp' with 0
        accrual['tempTXP'] = accrual['txpq'].fillna(0)
        def compute_accruals(group):
            group['Accruals'] = ((group['actq'] - group['actq'].shift(1))
                                - (group['cheq'] - group['cheq'].shift(1))
                                - ((group['lctq'] - group['lctq'].shift(1)) - (group['dlcq'] - group['dlcq'].shift(1)) - (group['tempTXP'] - group['tempTXP'].shift(1)))
                                - group['dpq']
                                ) / ((group['atq'] + group['atq'].shift(1)) / 2)
            return group

        # Apply the function to each permno group
        accrual = accrual.groupby('permno').apply(compute_accruals).reset_index(level=0, drop=True)
        self.factor_data = accrual[['Accruals']]
