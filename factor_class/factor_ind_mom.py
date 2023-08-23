from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorIndMom(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 ticker: List[str] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)

        price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        ind_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_ind.parquet.brotli')
        combine = pd.concat([price_data, ind_data], axis=1)

        t = 1
        ret = create_return(combine, windows=[t])[[f'RET_{t:02}', 'Industry']]
        avg_ret = ret.groupby(['Industry', ret.index.get_level_values('date')])[f'RET_{t:02}'].mean()
        ret = ret.reset_index()
        ret = pd.merge(ret, avg_ret.rename('indRET').reset_index(), on=['Industry', 'date'], how='left')
        ret[f'IndMom_{t:02}'] = ret[f'RET_{t:02}'] / ret['indRET']
        ind_mom = ret.set_index(['ticker', 'date'])[[f'IndMom_{t:02}']]
        self.factor_data = ind_mom
