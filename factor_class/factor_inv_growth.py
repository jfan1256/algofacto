from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorInvGrowth(Factor):
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
        columns = ['invtq']
        inv_growth = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        inv_growth = get_stocks_data(inv_growth, stock)

        # Convert CPI to multiindex
        medianCPI = pd.read_csv(get_load_data_large_dir() / 'macro' / 'medianCPI.csv')
        medianCPI.columns = ['date', 'medCPI']
        medianCPI['date'] = pd.to_datetime(medianCPI['date']).dt.to_period('M').dt.to_timestamp('M')
        medianCPI['date'] = (medianCPI['date'] + pd.DateOffset(months=1))
        medianCPI = medianCPI.set_index('date')
        medianCPI = medianCPI[~medianCPI.index.duplicated(keep='first')]
        factor_values = pd.concat([medianCPI] * len(stock), ignore_index=True).values
        multi_index = pd.MultiIndex.from_product([stock, medianCPI.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=medianCPI.columns, index=multi_index)
        multi_index_factor.index = multi_index_factor.index.set_names(['permno', 'date'])

        inv_growth = inv_growth.merge(multi_index_factor, left_index=True, right_index=True, how='left')
        inv_growth['invtq'] = inv_growth['invtq'] / inv_growth['medCPI']
        inv_growth['InvGrowth'] = inv_growth['invtq'] / inv_growth.groupby('permno')['invtq'].shift(12) - 1
        inv_growth = inv_growth[['InvGrowth']]
        self.factor_data = inv_growth
