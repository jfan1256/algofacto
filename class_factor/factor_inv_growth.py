from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorInvGrowth(Factor):
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
        columns = ['invtq']
        inv_growth = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_q.parquet.brotli', columns=columns)
        inv_growth = get_stocks_data(inv_growth, self.stock)

        # Convert CPI to multiindex
        medianCPI = pd.read_csv(get_large(self.live) / 'macro' / 'medianCPI.csv')
        medianCPI.columns = ['date', 'medCPI']
        medianCPI['date'] = pd.to_datetime(medianCPI['date']).dt.to_period('M').dt.to_timestamp('M')

        # Shift date 1 month forward for backtest
        if self.live == False:
            medianCPI['date'] = (medianCPI['date'] + pd.DateOffset(months=1))

        medianCPI = medianCPI.set_index('date')
        medianCPI = medianCPI[~medianCPI.index.duplicated(keep='first')]
        factor_values = pd.concat([medianCPI] * len(self.stock), ignore_index=True).values
        multi_index = pd.MultiIndex.from_product([self.stock, medianCPI.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=medianCPI.columns, index=multi_index)
        multi_index_factor.index = multi_index_factor.index.set_names(['permno', 'date'])

        multi_index_factor = multi_index_factor.sort_index()
        inv_growth = inv_growth.sort_index()
        multi_index_factor_reindexed = multi_index_factor.reindex(inv_growth.index, method='ffill')
        inv_growth = inv_growth.merge(multi_index_factor_reindexed, left_index=True, right_index=True)
        inv_growth['invtq'] = inv_growth['invtq'] / inv_growth['medCPI']
        inv_growth['inv_growth'] = inv_growth['invtq'] / inv_growth.groupby('permno')['invtq'].shift(6) - 1
        inv_growth = inv_growth[['inv_growth']]
        self.factor_data = inv_growth