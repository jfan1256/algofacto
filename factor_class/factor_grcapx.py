from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorGrcapx(Factor):
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
        # columns = ['ppentq', 'capxy']
        # grcapx = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        # grcapx = get_stocks_data(grcapx, self.stock)
        #
        # grcapx['FirmAge'] = grcapx.groupby('permno').cumcount() + 1
        #
        # grcapx['l12_ppent'] = grcapx.groupby('permno')['ppentq'].shift(12)
        # grcapx['l24_ppent'] = grcapx.groupby('permno')['ppentq'].shift(24)
        # grcapx['l36_ppent'] = grcapx.groupby('permno')['ppentq'].shift(36)
        # grcapx.loc[grcapx['capxy'].isna() & (grcapx['FirmAge'] >= 24), 'capxy'] = grcapx['ppentq'] - grcapx['l12_ppent']
        #
        # grcapx['grcapx'] = (grcapx['capxy'] - grcapx['l24_ppent']) / grcapx['l24_ppent']
        # grcapx['grcapx1y'] = (grcapx['l12_ppent'] - grcapx['l24_ppent']) / grcapx['l24_ppent']
        # grcapx['grcapx3y'] = grcapx['capxy'] / (grcapx['l12_ppent'] + grcapx['l24_ppent'] + grcapx['l36_ppent']) * 3
        # grcapx = grcapx[['grcapx', 'grcapx1y', 'grcapx3y']]
        # self.factor_data = grcapx

        columns = ['ppent', 'capx']
        grcapx = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns)
        grcapx = get_stocks_data(grcapx, self.stock)

        grcapx['FirmAge'] = grcapx.groupby('permno').cumcount() + 1

        grcapx['l12_ppent'] = grcapx.groupby('permno')['ppent'].shift(12)
        grcapx['l24_ppent'] = grcapx.groupby('permno')['ppent'].shift(24)
        grcapx['l36_ppent'] = grcapx.groupby('permno')['ppent'].shift(36)
        grcapx.loc[grcapx['capx'].isna() & (grcapx['FirmAge'] >= 24), 'capx'] = grcapx['ppent'] - grcapx['l12_ppent']

        grcapx['grcapx'] = (grcapx['capx'] - grcapx['l24_ppent']) / grcapx['l24_ppent']
        grcapx['grcapx1y'] = (grcapx['l12_ppent'] - grcapx['l24_ppent']) / grcapx['l24_ppent']
        grcapx['grcapx3y'] = grcapx['capx'] / (grcapx['l12_ppent'] + grcapx['l24_ppent'] + grcapx['l36_ppent']) * 3
        grcapx = grcapx[['grcapx', 'grcapx1y', 'grcapx3y']]
        self.factor_data = grcapx
