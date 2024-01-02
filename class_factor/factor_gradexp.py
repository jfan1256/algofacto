from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorGradexp(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing grax', animation=spinner_animation)
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
        columns = ['xad']
        grax = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli', columns=columns)
        outstanding = ['market_cap']
        price_data = pd.read_parquet(get_parquet(self.live) / 'data_misc.parquet.brotli', columns=outstanding)
        grax = grax.sort_index()
        price_data = price_data.sort_index()
        price_data_reindexed = price_data.reindex(grax.index, method='ffill')
        grax = grax.merge(price_data_reindexed, left_index=True, right_index=True)
        grax = get_stocks_data(grax, self.stock)
        grax['gradexp'] = np.log(grax['xad']) - np.log(grax.groupby('permno')['xad'].shift(12))
        grax['tempSize'] = grax.groupby('date')['market_cap'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1)
        grax.loc[(grax['xad'] < 0.1) | (grax['tempSize'] == 1), 'gradexp'] = np.nan
        self.factor_data = grax[['gradexp']]