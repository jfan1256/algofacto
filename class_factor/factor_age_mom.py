from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor
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

        T = [1]
        self.factor_data = create_return(self.factor_data, windows=T)
        self.factor_data['tempage'] = self.factor_data.groupby(self.group).cumcount() + 1
        def compound_return(group, day):
            compound_return = 1
            for i in range(1, day+1):
                compound_return *= (1 + group['RET_01'].shift(i))
            return compound_return - 1

        # Scaling factor for daily data
        scale_factor = 1

        self.factor_data['age_mom'] = self.factor_data.groupby(self.group).apply(compound_return, day=5*scale_factor).reset_index(level=0, drop=True)
        self.factor_data.loc[(self.factor_data['Close'].abs() < 5) | (self.factor_data['tempage'] < 12*scale_factor), 'age_mom'] = np.nan

        self.factor_data = self.factor_data[['age_mom']]