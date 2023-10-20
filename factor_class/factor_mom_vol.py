from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorMomVol(Factor):
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
        T = [1]
        self.factor_data = create_return(self.factor_data, windows=T)
        self.factor_data = get_stocks_data(self.factor_data, self.stock)

        def compute_momentum(x):
            return ((1 + x['RET_01'].shift(1)) * (1 + x['RET_01'].shift(2)) * (1 + x['RET_01'].shift(3))
                    * (1 + x['RET_01'].shift(4)) * (1 + x['RET_01'].shift(5)) - 1)

        self.factor_data['Mom6m'] = self.factor_data.groupby('permno').apply(compute_momentum).reset_index(level=0, drop=True)
        self.factor_data = self.factor_data.fillna(0)
        self.factor_data['catMom'] = self.factor_data.groupby('date')['Mom6m'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))
        self.factor_data['temp'] = self.factor_data.groupby('permno')['Volume'].transform(lambda x: x.rolling(window=6, min_periods=5).mean())
        self.factor_data = self.factor_data.fillna(0)
        self.factor_data['catVol'] = self.factor_data.groupby('date')['temp'].transform(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop'))
        self.factor_data['mom_vol'] = self.factor_data.apply(lambda x: x['catMom'] if x['catVol'] == 2 else None, axis=1)
        self.factor_data['mom_vol'] = self.factor_data.groupby(['permno', 'date']).cumcount().where(lambda x: x >= 24, self.factor_data['mom_vol'])
        self.factor_data = self.factor_data[['mom_vol']]
