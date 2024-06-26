from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorMomSeasonShort(Factor):
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
        splice_data = create_return(splice_data, T)
        # Scaling factor for daily data
        scale_factor = 21
        def compute_shifted_return(group):
            group['mom_season_short'] = group['RET_01'].shift(1 * scale_factor)
            return group[['mom_season_short']]

        splice_data = splice_data.groupby(self.group).apply(compute_shifted_return).reset_index(level=0, drop=True)
        return splice_data
