from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRankSBLagBond(Factor):
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
        sb_lag = pd.read_parquet(get_factor_data_dir() / 'factor_sb_lag_bond.parquet.brotli')
        sb_lag = get_stocks_data(sb_lag, self.stock)

        filtered_columns = [col for col in sb_lag.columns if not col.startswith(('ALPHA', 'PRED', 'EPSIL', 'RESID', 'IDIO', 'Open', 'Close', 'High', 'Low', 'Volume'))]

        # Ranking by each column
        sb_rank = sb_lag[[sb_lag.columns[0]]]
        for col in filtered_columns:
            sb_rank[f'{col}_rank'] = sb_lag.groupby('date')[col].rank()

            bin_size = 3.4
            max_compressed_rank = (sb_rank[f'{col}_rank'].max() + bin_size - 1) // bin_size
            sb_rank[f'{col}_rank'] = np.ceil(sb_rank[f'{col}_rank'] / bin_size)
            sb_rank[f'{col}_rank'] = sb_rank[f'{col}_rank'].apply(lambda x: min(x, max_compressed_rank))
            sb_rank[f'{col}_rank'] = sb_rank[f'{col}_rank'].replace({np.nan: -1, np.inf: max_compressed_rank}).astype(int)

        sb_rank = sb_rank.drop([sb_lag.columns[0]], axis=1)
        self.factor_data = sb_rank
