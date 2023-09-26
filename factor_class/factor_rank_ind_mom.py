from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRankIndMom(Factor):
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
        ind_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_ind.parquet.brotli')
        combine = pd.concat([price_data, ind_data], axis=1)

        T = [1, 2, 5, 10, 30, 60]
        ret = create_return(combine, windows=T)
        collect = []

        for t in T:
            ret[f'IndMom_{t:02}'] = ret.groupby(['ind', 'date'])[f'RET_{t:02}'].transform('mean')
            ret[f'indMom_{t:02}_rank'] =  ret.groupby(['date'])[f'IndMom_{t:02}'].rank()

            bin_size = 2.5
            max_compressed_rank = (ret[f'indMom_{t:02}_rank'].max() + bin_size - 1) // bin_size
            ret[f'indMom_{t:02}_rank'] = np.ceil(ret[f'indMom_{t:02}_rank'] / bin_size)
            ret[f'indMom_{t:02}_rank'] = ret[f'indMom_{t:02}_rank'].apply(lambda x: min(x, max_compressed_rank))
            ret[f'indMom_{t:02}_rank'] = ret[f'indMom_{t:02}_rank'].replace({np.nan: -1, np.inf: max_compressed_rank}).astype(int)
            ind_mom = ret[[f'indMom_{t:02}_rank']]
            collect.append(ind_mom)

        self.factor_data = pd.concat(collect, axis=1)

