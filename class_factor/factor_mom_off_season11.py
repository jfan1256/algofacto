from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorMomOffSeason11(Factor):
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

        # Grouping function to compute both mom_season and MomOffSeason
        def compute_mom(group):
            # Generating lagged returns for use in MomOffSeason calculation
            for n in range(131 * scale_factor, 180 * scale_factor, 12 * scale_factor):
                group[f'temp{n}'] = group['RET_01'].shift(n)

            group['retTemp1'] = group[[f'temp{i}' for i in range(131 * scale_factor, 180 * scale_factor, 12 * scale_factor)]].sum(axis=1, skipna=True)
            group['retTemp2'] = group[[f'temp{i}' for i in range(131 * scale_factor, 180 * scale_factor, 12 * scale_factor)]].notna().sum(axis=1)

            # Computing MomOffSeason
            group['retLagTemp'] = group['RET_01'].shift(21)
            group['retLagTemp_sum60'] = group['retLagTemp'].rolling(126).sum()
            group['retLagTemp_count60'] = group['retLagTemp'].rolling(126).count()
            group['mom_off_season11'] = (group['retLagTemp_sum60'] - group['retTemp1']) / (group['retLagTemp_count60'] - group['retTemp2'])

            return group[['mom_off_season11']]

        splice_data = splice_data.groupby('permno').apply(compute_mom).reset_index(level=0, drop=True)
        return splice_data






