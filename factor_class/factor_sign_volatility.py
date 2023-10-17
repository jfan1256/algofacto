from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSignVolatility(Factor):
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
        self.factor_data = pd.read_parquet(get_factor_data_dir() / 'factor_volatility.parquet.brotli')
        self.factor_data = self.factor_data.drop(['Open', 'High', 'Close', 'Low', 'Volume'], axis=1)
        self.factor_data = self.factor_data - self.factor_data.mean()

    @ray.remote
    def function(self, splice_data):
        for col in splice_data.columns:
            splice_data[f'sign_{col}'] = np.sign(splice_data[col])
            splice_data = splice_data.drop([col], axis=1)
        return splice_data
