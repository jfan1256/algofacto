from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRFVolume(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 ticker: List[str] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.all_rf = pd.read_parquet(get_load_data_parquet_dir() / 'data_all_rf.parquet.brotli')

    @ray.remote
    def function(self, splice_data):
        self.all_rf = self.all_rf.loc[self.start:self.end]
        self.all_rf = self.all_rf.fillna(0)

        # Get factor columns and create returns
        factors = self.all_rf.columns[:-1]
        T = [1, 6, 30]
        splice_data = create_volume(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        for t in T:
            vol = f'VOL_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [60]
            for window in windows:
                betas = rolling_ols_residual(price=splice_data, factor_data=self.all_rf, factor_col=factors, window=window,
                                             name=f'{t:02}_RF_VOL', ret=vol)
                splice_data = splice_data.join(betas)
            splice_data = splice_data.drop(vol, axis=1)

        return splice_data
