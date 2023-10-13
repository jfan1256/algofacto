from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorEmmult(Factor):
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
        columns = ['dlttq', 'dlcq', 'cheq', 'oibdpq', 'drcq']
        finance = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        outstanding = ['outstanding']
        price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_crsp.parquet.brotli', columns=outstanding)
        finance = finance.sort_index()
        price_data = price_data.sort_index()
        price_data_reindexed = price_data.reindex(finance.index, method='ffill')
        finance = finance.merge(price_data_reindexed, left_index=True, right_index=True)
        finance = get_stocks_data(finance, self.stock)
        finance['emmult'] = ((finance['outstanding']/1000) + finance['dlttq'] + finance['dlcq'] + finance['drcq'] - finance['cheq']) / finance['oibdpq']
        self.factor_data = finance[['emmult']]
