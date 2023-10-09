from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRankLoadVolume(Factor):
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
        load_data = pd.read_parquet(get_factor_data_dir() / 'factor_load_volume.parquet.brotli')
        load_data = get_stocks_data(load_data, self.stock)

        # Ranking by each column
        load_rank = load_data[[load_data.columns[0]]]
        for col in load_data.columns:
            load_rank[f'{col}_rank'] = load_data.groupby('date')[col].rank()

            bin_size = 3.4
            max_compressed_rank = (load_rank[f'{col}_rank'].max() + bin_size - 1) // bin_size
            load_rank[f'{col}_rank'] = np.ceil(load_rank[f'{col}_rank'] / bin_size)
            load_rank[f'{col}_rank'] = load_rank[f'{col}_rank'].apply(lambda x: min(x, max_compressed_rank))
            load_rank[f'{col}_rank'] = load_rank[f'{col}_rank'].replace({np.nan: -1, np.inf: max_compressed_rank}).astype(int)

        load_rank = load_rank.drop([load_data.columns[0]], axis=1)
        self.factor_data = load_rank
