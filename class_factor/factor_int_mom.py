from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorIntMom(Factor):
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
        T = [1]
        splice_data = create_return(splice_data, windows=T)
        # Scaling factor for daily data
        scale_factor = 1

        def compute_intmom(group):
            group['int_mom'] = (1 + group['RET_01'].shift(7)) * (1 + group['RET_01'].shift(8)) * \
                              (1 + group['RET_01'].shift(9)) * (1 + group['RET_01'].shift(10)) * \
                              (1 + group['RET_01'].shift(11)) * (1 + group['RET_01'].shift(12)) - 1
            return group

        result = splice_data.groupby(self.group).apply(compute_intmom).reset_index(level=0, drop=True)
        splice_data = result[['int_mom']]
        return splice_data
