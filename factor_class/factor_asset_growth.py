from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorAssetGrowth(Factor):
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
        columns = ['atq']
        growth = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        growth = get_stocks_data(growth, self.stock)
        growth['asset_growth'] = (growth['atq'] - growth.groupby('permno')['atq'].shift(6)) / growth.groupby('permno')['atq'].shift(6)
        growth = growth[['asset_growth']]
        self.factor_data = growth

        # columns = ['at']
        # growth = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns)
        # growth = get_stocks_data(growth, self.stock)
        # growth['asset_growth'] = (growth['at'] - growth.groupby('permno')['at'].shift(12)) / growth.groupby('permno')['at'].shift(12)
        # growth = growth[['asset_growth']]
        # self.factor_data = growth
