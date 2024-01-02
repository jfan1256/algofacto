from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor


class FactorRankSBBond(Factor):
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
        sb_lag = pd.read_parquet(get_factor(self.live) / 'factor_sb_bond.parquet.brotli')
        sb_lag = get_stocks_data(sb_lag, self.stock)

        filtered_columns = [col for col in sb_lag.columns if not col.startswith(('ALPHA', 'PRED', 'EPSIL', 'RESID', 'IDIO', 'Open', 'Close', 'High', 'Low', 'Volume'))]

        # Ranking by each column
        sb_rank = sb_lag[[sb_lag.columns[0]]]
        for col in filtered_columns:
            sb_rank[f'{col}_rank'] = sb_lag.groupby('date')[col].rank(method='dense')

        sb_rank = sb_rank.drop([sb_lag.columns[0]], axis=1)
        self.factor_data = sb_rank
