from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorXFIN(Factor):
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
        columns = ['sstky', 'dvy', 'atq', 'prstkcy', 'dltisy', 'dltry', 'dlcchy']
        finance = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        finance = get_stocks_data(finance, self.stock)
        finance['xfin'] = (finance['sstky'] - finance['dvy'] - finance['prstkcy'] + finance['dltisy'] - finance['dltry'] + finance['dlcchy']) / finance['atq']
        self.factor_data = finance[['xfin']]

        # columns = ['sstk', 'dv', 'at', 'prstkc', 'dltis', 'dltr', 'dlcch']
        # finance = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns)
        # finance = get_stocks_data(finance, self.stock)
        # finance['xfin'] = (finance['sstk'] - finance['dv'] - finance['prstkc'] + finance['dltis'] - finance['dltr'] + finance['dlcch']) / finance['at']
        # self.factor_data = finance[['xfin']]

