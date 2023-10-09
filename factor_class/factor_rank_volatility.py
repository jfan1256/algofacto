from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRankVolatility(Factor):
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
        crsp = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        crsp = get_stocks_data(crsp, self.stock)

        T = [1, 2, 5, 10, 30, 60]
        crsp = crsp.fillna(0)
        crsp = create_return(crsp, windows=T)
        collect = []
        for t in T:
            crsp[f'VOLATILITY_{t:02}'] = crsp.groupby('permno')[f'RET_{t:02}'].rolling(window=60).std().reset_index(level=0, drop=True)
            crsp[f'RANK_VOLATILITY_{t:02}'] = crsp[f'VOLATILITY_{t:02}'].groupby('date').rank()
            crsp = crsp.drop(f'RET_{t:02}', axis=1)
            crsp = crsp.drop(f'VOLATILITY_{t:02}', axis=1)

            bin_size = 3.4
            max_compressed_rank = (crsp[f'RANK_VOLATILITY_{t:02}'].max() + bin_size - 1) // bin_size
            crsp[f'RANK_VOLATILITY_{t:02}'] = np.ceil(crsp[f'RANK_VOLATILITY_{t:02}'] / bin_size)
            crsp[f'RANK_VOLATILITY_{t:02}'] = crsp[f'RANK_VOLATILITY_{t:02}'].apply(lambda x: min(x, max_compressed_rank))
            crsp[f'RANK_VOLATILITY_{t:02}'] = crsp[f'RANK_VOLATILITY_{t:02}'].replace({np.nan: -1, np.inf: max_compressed_rank}).astype(int)
            rank = crsp[[f'RANK_VOLATILITY_{t:02}']]
            collect.append(rank)
        self.factor_data = pd.concat(collect, axis=1)
