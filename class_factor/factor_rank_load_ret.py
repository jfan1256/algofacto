from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorRankLoadRet(Factor):
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
        load_data = pd.read_parquet(get_factor(self.live) / 'factor_load_ret.parquet.brotli')
        load_data = get_stocks_data(load_data, self.stock)

        # Ranking by each column
        load_rank = load_data[[load_data.columns[0]]]
        for col in load_data.columns:
            load_rank[f'{col}_rank'] = load_data.groupby('date')[col].rank(method='dense')

        load_rank = load_rank.drop([load_data.columns[0]], axis=1)
        self.factor_data = load_rank
