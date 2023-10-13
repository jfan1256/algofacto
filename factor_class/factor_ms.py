from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorMS(Factor):
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
        columns = ['atq', 'xrdq', 'xsgaq', 'fqtr', 'niq', 'capxy', 'oancfy', 'sic', 'saleq']
        ms = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        ms = get_stocks_data(ms, self.stock)

        # Handle missing values
        ms['xrdq'].fillna(0, inplace=True)
        ms['xrdq'].fillna(0, inplace=True)

        # Create capxq and oancfq columns
        ms['capxq'] = np.where(ms['fqtr'] == 1, ms['capxy'], ms['capxy'] - ms['capxy'].shift(3))
        ms['oancfq'] = np.where(ms['fqtr'] == 1, ms['oancfy'], ms['oancfy'] - ms['oancfy'].shift(3))

        # Aggregate quarterly
        rolling_window = 12
        min_periods = 12

        for col in ['niq', 'xrdq', 'oancfq', 'capxq']:
            ms[f'{col}sum'] = ms.groupby(level='permno')[col].rolling(window=rolling_window, min_periods=min_periods).mean().reset_index(level=0, drop=True)

        # Update values
        ms[['niqsum', 'xrdqsum', 'capxqsum', 'oancfqsum']] *= 4

        # Signal Construction
        ms['atdenom'] = (ms['atq'] + ms['atq'].shift(3)) / 2
        ms['roa'] = ms['niqsum'] / ms['atdenom']
        ms['cfroa'] = ms['oancfqsum'] / ms['atdenom']

        # Compute medians by group
        for var in ['roa', 'cfroa']:
            ms[f'md_{var}'] = ms.groupby(['sic', 'date'])[var].transform('median')

        ms['m1'] = (ms['roa'] > ms['md_roa']).astype(int)
        ms['m2'] = (ms['cfroa'] > ms['md_cfroa']).astype(int)
        ms['m3'] = (ms['oancfqsum'] > ms['niqsum']).astype(int)

        # Further signal construction (naive extrapolation)
        ms['roaq'] = ms['niq'] / ms['atq']
        ms['sg'] = ms['saleq'] / ms['saleq'].shift(3)

        rolling_window = 48
        min_periods = 18

        # Calculate rolling volatility for roaq and sg
        ms['niVol'] = ms.groupby(level='permno')['roaq'].rolling(window=rolling_window, min_periods=min_periods).std().reset_index(level=0, drop=True)
        ms['revVol'] = ms.groupby(level='permno')['sg'].rolling(window=rolling_window, min_periods=min_periods).std().reset_index(level=0, drop=True)

        # Compute medians by group
        for var in ['niVol', 'revVol']:
            ms[f'md_{var}'] = ms.groupby(['sic', 'date'])[var].transform('median')

        ms['m4'] = (ms['niVol'] < ms['md_niVol']).astype(int)
        ms['m5'] = (ms['revVol'] < ms['md_revVol']).astype(int)

        # More signal construction (Conservatism)
        ms['atdenom2'] = ms['atq'].shift(3)
        ms['xrdint'] = ms['xrdqsum'] / ms['atdenom2']
        ms['capxint'] = ms['capxqsum'] / ms['atdenom2']
        ms['xrdqint'] = ms['xrdq'] / ms['atdenom2']

        # Compute medians by group
        for var in ['xrdint', 'capxint', 'xrdqint']:
            ms[f'md_{var}'] = ms.groupby(['sic', 'date'])[var].transform('median')

        ms['m6'] = (ms['xrdint'] > ms['md_xrdint']).astype(int)
        ms['m7'] = (ms['capxint'] > ms['md_capxint']).astype(int)
        ms['m8'] = (ms['xrdqint'] > ms['md_xrdqint']).astype(int)

        ms['tempMS'] = ms[['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']].sum(axis=1)
        ms['ms'] = ms['tempMS']
        ms.loc[(ms['tempMS'] >= 6) & (ms['tempMS'] <= 8), 'MS'] = 6
        ms.loc[ms['tempMS'] <= 1, 'ms'] = 1
        self.factor_data = ms[['ms']]
