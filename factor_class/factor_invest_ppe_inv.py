from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorInvestPPEInv(Factor):
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
        columns = ['atq', 'invtq', 'ppegtq']
        invest = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        invest = get_stocks_data(invest, self.stock)

        invest['tempPPE'] = invest['ppegtq'] - invest.groupby('permno')['ppegtq'].shift(6)
        invest['tempInv'] = invest['invtq'] - invest.groupby('permno')['invtq'].shift(6)
        invest['invest_ppe_inv'] = (invest['tempPPE'] + invest['tempInv']) / invest.groupby('permno')['atq'].shift(6)
        invest = invest[['invest_ppe_inv']]
        self.factor_data = invest

        # columns = ['at', 'invt', 'ppegt']
        # invest = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns)
        # invest = get_stocks_data(invest, self.stock)
        #
        # invest['tempPPE'] = invest['ppegt'] - invest.groupby('permno')['ppegt'].shift(12)
        # invest['tempInv'] = invest['invt'] - invest.groupby('permno')['invt'].shift(12)
        # invest['invest_ppe_inv'] = (invest['tempPPE'] + invest['tempInv']) / invest.groupby('permno')['at'].shift(12)
        # invest = invest[['invest_ppe_inv']]
        # self.factor_data = invest

