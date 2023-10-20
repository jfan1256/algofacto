from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRankIndVWRFama(Factor):
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

        ohclv = pd.read_parquet(get_parquet_dir(self.live) / 'data_price.parquet.brotli')
        ind = pd.read_parquet(get_parquet_dir(self.live) / 'data_ind_fama.parquet.brotli')
        out = pd.read_parquet(get_parquet_dir(self.live) / 'data_misc.parquet.brotli', columns=['outstanding'])

        ohclv = get_stocks_data(ohclv, self.stock)
        ind = get_stocks_data(ind, self.stock)
        out = get_stocks_data(out, self.stock)

        T = [1, 21, 126, 252]
        ind_data = pd.merge(ohclv, ind, left_index=True, right_index=True, how='left').merge(out, left_index=True, right_index=True, how='left')
        ind_data = create_return(ind_data, windows=T)
        collect = []
        ind_data['value_permno'] = ind_data['Close'] * ind_data['outstanding']
        ind_data['value_ind'] = ind_data.groupby(['IndustryFama', 'date'])['value_permno'].transform('sum')
        ind_data['vwr_weight'] = ind_data['value_permno'] / ind_data['value_ind']

        for t in T:
            ind_data[f'vwr_{t:02}'] = ind_data['vwr_weight'] * ind_data[f'RET_{t:02}']
            ind_data[f'rank_vwr_fama_{t:02}'] = ind_data.groupby(['IndustryFama', 'date'])[f'vwr_{t:02}'].rank(method='dense')
            collect.append(ind_data[[f'rank_vwr_fama_{t:02}']])

        self.factor_data = pd.concat(collect, axis=1)