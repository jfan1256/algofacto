from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBSPYInv(Factor):
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
        self.spy_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_spy.parquet.brotli')
        self.fama_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_fama.parquet.brotli')
        self.spy_data = pd.concat([self.spy_data, self.fama_data['RF']], axis=1)
        self.spy_data = self.spy_data.loc[self.start:self.end]
        self.spy_data = self.spy_data.fillna(0)
        self.factor_col = self.spy_data.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        splice_data = create_return(splice_data, windows=[1])

        t = 1
        ret = f'RET_{t:02}'
        # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
        windows = [30, 60]
        for window in windows:
            betas = rolling_ols_sb(price=splice_data, factor_data=self.spy_data, factor_col=self.factor_col, window=window,
                                   name='INV_SPY', ret=ret)
            betas = betas[[col for col in betas.columns if col.startswith('spyRet')]]
            betas = betas.rdiv(1)
            splice_data = splice_data.join(betas)

        return splice_data