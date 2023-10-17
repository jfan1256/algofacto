from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorChEQ(Factor):
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
        columns = ['ceqq', 'atq']
        equity = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        equity = get_stocks_data(equity, self.stock)
        equity['cheq'] = equity['ceqq'] / equity.groupby('permno')['ceqq'].shift(6)
        equity = equity[(equity['ceqq'] > 0) & (equity.groupby('permno')['ceqq'].shift(6) > 0)]
        self.factor_data = equity[['cheq']]
        #
        # columns = ['ceq', 'at']
        # equity = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns)
        # equity = get_stocks_data(equity, self.stock)
        # equity['cheq'] = equity['ceq'] / equity.groupby('permno')['ceq'].shift(12)
        # equity = equity[(equity['ceq'] > 0) & (equity.groupby('permno')['ceq'].shift(12) > 0)]
        # self.factor_data = equity[['cheq']]
