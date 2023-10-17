from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorCHTax(Factor):
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
        columns = ['txtq', 'atq']
        tax = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        tax = get_stocks_data(tax, self.stock)
        tax['chtax'] = (tax['txtq'] - tax.groupby('permno')['txtq'].shift(6)) / tax.groupby('permno')['atq'].shift(6)
        tax = tax[['chtax']]
        self.factor_data = tax

        # columns = ['txtq']
        # tax = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        # annual = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=['at'])
        # tax = tax.merge(annual, left_index=True, right_index=True, how='left')
        # tax = get_stocks_data(tax, self.stock)
        # tax['chtax'] = (tax['txtq'] - tax.groupby('permno')['txtq'].shift(12)) / tax.groupby('permno')['at'].shift(12)
        # tax = tax[['chtax']]
        # self.factor_data = tax
