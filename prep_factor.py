from functions.utils.func import *
from functions.utils.system import *
from typing import Union, Optional, List

import warnings

warnings.filterwarnings('ignore')


class PrepFactor:
    def __init__(self,
                 live: bool = None,
                 factor_name: str = None,
                 group: str = None,
                 interval: str = 'D',
                 kind: str = 'price',
                 div: bool = False,
                 stock: Optional[Union[List[str], str]] = None,
                 start: str = '2006-01-01',
                 end: str = '2022-01-01',
                 save: bool = False):
        self.data = None
        self.live = live
        self.group = group
        self.interval = interval
        self.kind = kind
        self.div = div
        self.factor_name = factor_name
        self.stock = stock
        self.start = start
        self.end = end
        self.path = Path(get_prep_dir(live) / f'prep_{self.factor_name}.parquet.brotli')
        self.save = save

    # Get factor data
    def _get_factor(self):
        # Read in factor data
        data_all = pd.read_parquet(get_factor_dir(self.live) / f'{self.factor_name}.parquet.brotli')
        # Remove OHCLV columns in price factors
        if self.kind == 'price':
            data_all = data_all.drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1)
        # Set self.stock to list of all stocks in dataframe
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

    # Set data interval
    def _set_interval(self):
        if self.interval == 'D':
            return self.data
        else:
            # Resample from not daily to daily
            date_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_date.parquet.brotli')
            date_data = set_timeframe(date_data, self.start, self.end)
            date_data = get_stocks_data(date_data, self.stock)
            self.data = pd.merge(date_data, self.data, left_index=True, right_index=True, how='left')
            self.data = self.data.loc[~self.data.index.duplicated(keep='first')]
            # Forward Fill by a maximum of 93 days
            # self.data = self.data.groupby(self.group).fillna(method='ffill', limit=93)
            # Forward Fill
            self.data = self.data.groupby(self.group).ffill()
            # If self.kind is set to 'fundamental', then calculate moving average
            if self.kind == 'fundamental':
                self.data = self.data.groupby(self.group).rolling(window=21).mean().reset_index(level=0, drop=True)
            return self.data

    # Divided by price
    def _div_price(self):
        # Divide factor by closing price
        if self.div:
            price_data = pd.read_parquet(get_parquet_dir(self.live) / 'data_price.parquet.brotli')
            self.data = pd.merge(self.data, price_data.Close.loc[self.stock], left_index=True, right_index=True, how='left')
            self.data = self.data.loc[~self.data.index.duplicated(keep='first')]
            self.data.iloc[:, :-1] = self.data.iloc[:, :-1].div(self.data.Close, axis=0)
            self.data = self.data.drop(['Close'], axis=1)
        return self.data


    # Handle various data errors
    def _handle_data(self):
        # Replace all infinite values with NAN
        self.data = self.data.replace([np.inf, -np.inf], np.nan)

        # Remove the last row of data for each stock that is delisted (the last data point of delisted stocks are NAN)
        self.data = remove_row_before_end(self.data, self.group, self.end)

        # Remove all duplicate indices
        self.data = self.data.loc[~self.data.index.duplicated(keep='first')]
        return self.data


    # Execute prepping process
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Creating {self.factor_name}', animation=spinner_animation, post_func=print_data_shape)
    def prep(self):
        # If no file exists for this factor, prep the factor and export it
        if not self.path.exists() or self.save:
            self.data = self._get_factor()
            self.data = self._set_interval()
            self.data = self._div_price()
            self.data = set_timeframe(self.data, self.start, self.end)
            self.data = self._handle_data()
            self.data.to_parquet(self.path, compression='brotli')
            return self.data
        else:
            # Directly accessed prepped factor file
            self.data = pd.read_parquet(self.path)
            self.data = get_stocks_data(self.data, self.stock)
            self.data = set_timeframe(self.data, self.start, self.end)
            return self.data
