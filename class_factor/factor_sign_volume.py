from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorSignVolume(Factor):
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
        T = [1, 21, 126, 252]
        splice_data = create_volume(splice_data, windows=T)
        for t in T:
            splice_data[f'sign_volume_{t:02}'] = np.sign(splice_data[f'VOL_{t:02}'])
            splice_data = splice_data.drop([f'VOL_{t:02}'], axis=1)
        return splice_data
