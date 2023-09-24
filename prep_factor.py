from functions.utils.func import *
from functions.utils.system import *
from typing import Union, Optional, List

import warnings

warnings.filterwarnings('ignore')


class PrepFactor:
    def __init__(self, factor_name: str = None,
                 interval: str = 'D',
                 kind: str = 'price',
                 div: bool = False,
                 stock: Optional[Union[List[str], str]] = None,
                 start: str = '2006-01-01',
                 end: str = '2022-01-01',
                 save: bool = False):
        self.data = None
        self.interval = interval  # specify the interval of the data being prepped
        self.kind = kind  # type of factor (price, industry)
        self.div = div  # divide factor by closing price
        self.factor_name = factor_name
        self.stock = stock
        self.start = start  # Suggested start date is 2010
        self.end = end  # Suggested start date is 2022
        self.path = Path(get_load_data_prep_dir() / f'prep_{self.factor_name}.parquet.brotli')
        self.save = save

    def get_factor(self):
        data_all = pd.read_parquet(get_factor_data_dir() / f'{self.factor_name}.parquet.brotli')

        # Remove OHCLV columns in price factors
        if self.kind == 'price':
            data_all = data_all.drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1)

        # Set self.stock to all stocks in the data set
        if self.stock == 'all':
            self.stock = get_stock_idx(data_all)

        # Remove historical returns from factors except for factor_return
        if self.factor_name == 'factor_ret':
            self.data = data_all
            self.data = get_stocks_data(self.data, self.stock)
            return self.data
        else:
            data_all = data_all.loc[:, ~data_all.columns.str.startswith('RET')]
            self.data = data_all
            self.data = get_stocks_data(self.data, self.stock)
            return self.data

    def set_interval(self):
        if self.interval == 'D':
            return self.data
        else:
            # Resample from not daily to daily
            date_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_date.parquet.brotli')
            stocks = read_stock(get_load_data_large_dir() / 'permno_to_train.csv')
            date_data = set_timeframe(date_data, self.start, self.end)
            self.data = pd.merge(date_data.loc[stocks], self.data, left_index=True, right_index=True, how='left')
            self.data = self.data.loc[~self.data.index.duplicated(keep='first')]
            self.data = self.data.ffill()
            return self.data

    def div_price(self):
        if self.div:
            # Divid factor by closing price
            if self.div:
                price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
                self.data = pd.merge(self.data, price_data.Close.loc[self.stock], left_index=True, right_index=True, how='left')
                self.data = self.data.loc[~self.data.index.duplicated(keep='first')]
                self.data.iloc[:, :-1] = self.data.iloc[:, :-1].div(self.data.Close, axis=0)
                self.data = self.data.drop(['Close'], axis=1)
        return self.data
    def handle_data(self):
        # for column in self.data.columns:
        #     if self.data[column].isnull().sum() > 0.50 * len(self.data[column]):
        #         self.data = self.data.drop(column, axis=1)
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        return self.data

    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Creating {self.factor_name}', animation=spinner_animation, post_func=print_data_shape)
    def prep(self):
        if not self.path.exists() or self.save:
            self.data = self.get_factor()
            self.data = self.set_interval()
            self.data = self.div_price()
            self.data = set_timeframe(self.data, self.start, self.end)
            self.data = self.handle_data()
            self.data.to_parquet(self.path, compression='brotli')
            return self.data
        else:
            self.data = pd.read_parquet(self.path)
            self.data = get_stocks_data(self.data, self.stock)
            self.data = set_timeframe(self.data, self.start, self.end)

            return self.data
