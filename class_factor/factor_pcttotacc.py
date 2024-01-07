from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorPctTotAcc(Factor):
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
        columns = ['sstk', 'ni', 'prstkc', 'dvt', 'oancf', 'fincf', 'ivncf']
        finance = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli', columns=columns)
        finance = get_stocks_data(finance, self.stock)
        finance['pct_tot_acc'] = (finance['ni'] - (finance['prstkc'] - finance['sstk'] + finance['dvt'] + finance['oancf'] + finance['fincf'] + finance['ivncf'])) / finance['ni'].abs()
        self.factor_data = finance[['pct_tot_acc']]