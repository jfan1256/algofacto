from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorCondIndMom(Factor):
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
        combine = pd.concat([price_data, ind_data], axis=1)

        T = [1]
        ret = create_return(combine, windows=T)
        collect = []

        for t in T:
            grouped = ret.groupby(['Industry', ret.index.get_level_values('date')])

            # Compute average returns just once
            avg_ret = grouped[f'RET_{t:02}'].transform('mean')

            # Calculate shifted returns
            ret_shifted_1 = grouped[f'RET_{t:02}'].shift(1)
            ret_shifted_2 = grouped[f'RET_{t:02}'].shift(2)
            avg_ret_shifted_1 = avg_ret.shift(1)
            avg_ret_shifted_2 = avg_ret.shift(2)

            condition = (
                    (ret[f'RET_{t:02}'] > avg_ret) &
                    (ret_shifted_1 > avg_ret_shifted_1) &
                    (ret_shifted_2 > avg_ret_shifted_2)
            )

            ret[f'cond_ind_mom_{t:02}'] = np.where(condition, 1, 0)
            ind_mom = ret[[f'cond_ind_mom_{t:02}']]
            collect.append(ind_mom)

        self.factor_data = pd.concat(collect, axis=1)
