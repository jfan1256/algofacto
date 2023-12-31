from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorAgeMom(Factor):
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
        splice_data['tempage'] = splice_data.groupby(self.group).cumcount() + 1
        def compound_return(group, day):
            compound_return = 1
            for i in range(1, day+1):
                compound_return *= (1 + group['RET_01'].shift(i))
            return compound_return - 1

        # Scaling factor for daily data
        scale_factor = 1

        splice_data['age_mom'] = splice_data.groupby(self.group).apply(compound_return, day=5*scale_factor).reset_index(level=0, drop=True)
        splice_data.loc[(splice_data['Close'].abs() < 5) | (splice_data['tempage'] < 12*scale_factor), 'age_mom'] = np.nan

        splice_data = splice_data[['age_mom']]
        return splice_data
