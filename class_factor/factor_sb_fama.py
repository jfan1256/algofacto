from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorSBFama(Factor):
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
        self.risk_free = pd.read_parquet(get_parquet(self.live) / 'data_rf.parquet.brotli')
        self.risk_free = self.risk_free.loc[self.start:self.end]
        self.risk_free['RF'] = self.risk_free['RF'].ffill()
        self.risk_free = self.risk_free.fillna(0)
        self.factor_col = self.risk_free.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        T = [1]
        splice_data = create_return(splice_data, T)
        splice_data = splice_data.fillna(0)

        for t in T:
            ret = f'RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [21, 126]
            for window in windows:
                betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.risk_free, factor_col=self.factor_col, window=window, name=f'fama_{t:02}', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data