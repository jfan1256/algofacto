from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorIndMomComp(Factor):
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

        price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        crsp = pd.read_parquet(get_load_data_parquet_dir() / 'data_crsp.parquet.brotli', columns=['market_cap'])
        ind_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_ind.parquet.brotli')
        combine = pd.concat([price_data, ind_data, crsp], axis=1)


        T = [1]
        ret = create_return(combine, windows=T)
        ret['Mom6m'] = ret.groupby('permno').apply(
            lambda group: (1 + group['RET_01'].shift(1)) * (1 + group['RET_01'].shift(2)) * \
                          (1 + group['RET_01'].shift(3)) * (1 + group['RET_01'].shift(4)) * \
                          (1 + group['RET_01'].shift(5)) - 1).reset_index(level=0, drop=True)

        ret['ind_mom_comp'] = ret.groupby(['Industry', 'date'])['Mom6m'].transform('mean')
        self.factor_data = ret[['ind_mom_comp']]

