from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorNOA(Factor):
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
        columns = ['atq', 'ceqq', 'cheq', 'dcomq', 'dlttq', 'mibtq']
        noa = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_q.parquet.brotli', columns=columns)
        noa = get_stocks_data(noa, self.stock)
        noa['OA'] = noa['atq'] - noa['cheq']
        noa['OL'] = noa['atq'] - noa['dlttq'] - noa['mibtq'] - noa['dcomq'] - noa['ceqq']
        noa['noa'] = (noa['OA'] - noa['OL']) / noa.groupby('permno')['atq'].shift(6)
        noa = noa[['noa']]
        self.factor_data = noa


