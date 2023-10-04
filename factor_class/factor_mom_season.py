from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorMomSeason(Factor):
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
        splice_data = splice_data.fillna(0)
        # Scaling factor for daily data
        scale_factor = 21

        for n in range(23 * scale_factor, 60 * scale_factor, 12 * scale_factor):
            splice_data[f'temp{n}'] = splice_data['RET_01'].shift(n)
        # for n in range(2*scale_factor, 5*scale_factor, 1*scale_factor):
        #     splice_data[f'temp{n}'] = splice_data['RET_01'].shift(n)

        splice_data['retTemp1'] = splice_data[[col for col in splice_data.columns if 'temp' in col]].sum(axis=1, skipna=True)
        splice_data['retTemp2'] = splice_data[[col for col in splice_data.columns if 'temp' in col]].count(axis=1)
        splice_data['MomSeason'] = splice_data['retTemp1'] / splice_data['retTemp2']
        splice_data = splice_data[['MomSeason']]
        return splice_data
