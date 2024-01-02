from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorRankIndMom(Factor):
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

        price_data = pd.read_parquet(get_parquet(self.live) / 'data_price.parquet.brotli')
        ind_data = pd.read_parquet(get_parquet(self.live) / 'data_ind.parquet.brotli')
        price_data = get_stocks_data(price_data, self.stock)
        ind_data = get_stocks_data(ind_data, self.stock)

        combine = pd.concat([price_data, ind_data], axis=1)

        T = [1, 21, 126, 252]
        ret = create_return(combine, windows=T)

        collect = []
        for t in T:
            ret[f'IndMom_{t:02}'] = ret.groupby(['Industry', 'date'])[f'RET_{t:02}'].transform('mean')
            ret[f'indMom_{t:02}_rank'] = ret.groupby(['date'])[f'IndMom_{t:02}'].rank(method='dense')
            ind_mom = ret[[f'indMom_{t:02}_rank']]
            collect.append(ind_mom)

        self.factor_data = pd.concat(collect, axis=1)

