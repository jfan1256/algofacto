from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorStreversal(Factor):
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
        T = [5, 10, 40, 60]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)
        # Streveral
        condition1 = (splice_data['RET_05'] > 0) & (splice_data['RET_60'] < 0) & (splice_data['RET_40'] > 0)
        splice_data['streversal'] = np.where(condition1, 1, 0)

        # Strong Momentum
        condition2 = (splice_data['RET_05'] > splice_data['RET_10']) & (splice_data['RET_10'] > splice_data['RET_40'])
        splice_data['strong_momentum'] = np.where(condition2, 1, 0)

        for t in T:
            splice_data = splice_data.drop([f'RET_{t:02}'], axis=1)

        return splice_data
