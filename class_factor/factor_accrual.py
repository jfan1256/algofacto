from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorAccrual(Factor):
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
        columns = ['txp', 'act', 'lct', 'dp', 'at', 'dlc', 'che']
        accrual = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli', columns=columns)
        accrual = get_stocks_data(accrual, self.stock)
        # Replace missing values in 'txp' with 0
        accrual['tempTXP'] = accrual['txp'].fillna(0)

        def compute_accruals(group):
            group['accruals'] = ((group['act'] - group['act'].shift(12))
                                 - (group['che'] - group['che'].shift(12))
                                 - ((group['lct'] - group['lct'].shift(12)) - (group['dlc'] - group['dlc'].shift(12)) - (group['tempTXP'] - group['tempTXP'].shift(12)))
                                 - group['dp']
                                 ) / ((group['at'] + group['at'].shift(12)) / 2)
            return group

        # Apply the function to each permno group
        accrual = accrual.groupby('permno').apply(compute_accruals).reset_index(level=0, drop=True)
        self.factor_data = accrual[['accruals']]
