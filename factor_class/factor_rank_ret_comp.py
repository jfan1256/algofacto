from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRankRetComp(Factor):
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
        ret_comp = pd.read_parquet(get_factor_dir(self.live) / 'factor_ret_comp.parquet.brotli')
        ret_comp = get_stocks_data(ret_comp, self.stock)
        collect = []
        ret_comp = ret_comp.drop(['Open', 'High', 'Close', 'Low', 'Volume'], axis=1)
        for col in ret_comp:
            ret_comp[f'{col}_rank'] = ret_comp[col].groupby('date').rank(method='dense')
            rank = ret_comp[[f'{col}_rank']]
            collect.append(rank)

        self.factor_data = pd.concat(collect, axis=1)