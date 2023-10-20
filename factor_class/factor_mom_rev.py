from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorMomRev(Factor):
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
        self.factor_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_price.parquet.brotli')
        self.factor_data = get_stocks_data(self.factor_data, self.stock)
        self.factor_data = create_return(self.factor_data, [1])
        self.factor_data['Mom6m'] = self.factor_data.groupby('permno')['RET_01'].rolling(window=6).apply(lambda x: (1 + x).prod() - 1, raw=True).reset_index(level=0, drop=True)
        self.factor_data['Mom36m'] = self.factor_data.groupby('permno')['RET_01'].rolling(window=36).apply(lambda x: (1 + x).prod() - 1, raw=True).reset_index(level=0, drop=True)

        def custom_qcut(x):
            if len(x.dropna()) == 0:
                return pd.Series([None] * len(x), index=x.index)
            else:
                return pd.qcut(x, 5, labels=False, duplicates='drop') + 1

        self.factor_data['tempMom6'] = self.factor_data.groupby(['date'])['Mom6m'].transform(custom_qcut)
        self.factor_data['tempMom36'] = self.factor_data.groupby(['date'])['Mom36m'].transform(custom_qcut)

        self.factor_data['mom_rev'] = 0
        self.factor_data.loc[(self.factor_data['tempMom6'] == 5) & (self.factor_data['tempMom36'] == 1), 'mom_rev'] = 1
        self.factor_data.loc[(self.factor_data['tempMom6'] == 1) & (self.factor_data['tempMom36'] == 5), 'mom_rev'] = 0
        self.factor_data = self.factor_data[['mom_rev']]
        
