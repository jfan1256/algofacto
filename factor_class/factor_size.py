from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSize(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing grax', animation=spinner_animation)
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
        outstanding = ['market_cap']
        price_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_crsp.parquet.brotli', columns=outstanding)
        price_data['size'] = np.log(price_data['market_cap'])
        self.factor_data = price_data[['size']]