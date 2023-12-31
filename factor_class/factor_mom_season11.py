from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorMomSeason11(Factor):
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

        def compute_mom(group):
            for n in range(131 * scale_factor, 180 * scale_factor, 12 * scale_factor):
                group[f'temp{n}'] = group['RET_01'].shift(n)

            group['retTemp1'] = group[[col for col in group.columns if 'temp' in col]].sum(axis=1, skipna=True)
            group['retTemp2'] = group[[col for col in group.columns if 'temp' in col]].count(axis=1)
            group['mom_season_11'] = group['retTemp1'] / group['retTemp2']
            return group[['mom_season_11']]

        splice_data = splice_data.groupby(self.group).apply(compute_mom).reset_index(level=0, drop=True)
        return splice_data
