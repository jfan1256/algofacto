from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRFSign(Factor):
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
        self.all_rf = pd.read_parquet(get_load_data_parquet_dir() / 'data_all_rf_test.parquet.brotli')
        self.all_rf = self.all_rf.loc[self.start:self.end]
        self.all_rf = self.all_rf.fillna(0)
        self.factor_col = self.all_rf.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        T = [1, 6, 30]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        for t in T:
            splice_data[f'SIGN_RET_{t:02}'] = np.sign(splice_data[f'RET_{t:02}'])
            sign = f'SIGN_RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [60]
            for window in windows:
                betas = rolling_ols_beta_res(price=splice_data, factor_data=self.all_rf, factor_col=self.factor_col, window=window, name=f'{t:02}_RF_SIGN', ret=sign)
                splice_data = splice_data.join(betas)
            splice_data = splice_data.drop(sign, axis=1)

        return splice_data
