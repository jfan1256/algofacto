from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorResidMom(Factor):
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
        crsp = pd.read_parquet(get_factor_data_dir() / 'factor_sb_fama.parquet.brotli')
        filtered_columns = [col for col in crsp.columns if col.startswith('resid')][:-2]
        self.factor_data = crsp[filtered_columns]