from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorEarningDisparity(Factor):
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
        # Read in actual summary and summary statistic files from WRDS
        statistic = pd.read_csv(get_load_data_large_dir() / 'summary_statistic_unadj_ibes.csv')
        actual = pd.read_csv(get_load_data_large_dir() / 'summary_actual_unadj_ibes.csv')
        statistic.columns = statistic.columns.str.lower()
        actual.columns = actual.columns.str.lower()
        statistic = statistic.drop('ticker', axis=1)
        actual = actual.drop('ticker', axis=1)
        statistic['date'] = pd.to_datetime(statistic['statpers']).dt.to_period('M')
        statistic = statistic.rename(columns={'oftic': 'ticker'})
        actual = actual.rename(columns={'oftic': 'ticker'})
        actual['date'] = pd.to_datetime(actual['statpers']).dt.to_period('M')
        actual = actual.drop(['statpers'], axis=1)
        actual = actual.groupby(['ticker', 'date']).last().reset_index()
        combined = pd.merge(actual, statistic, on=['ticker', 'date'], how='right')
        # Prep IBES data for short term
        ibes_short = combined[combined['fpi'] == 1].copy()
        ibes_short = ibes_short.dropna(subset=['fpedats'])
        ibes_short['statpers'] = pd.to_datetime(ibes_short['statpers'])
        ibes_short['fpedats'] = pd.to_datetime(ibes_short['fpedats'])
        ibes_short = ibes_short[ibes_short['fpedats'] > ibes_short['statpers'] + pd.Timedelta(days=30)]
        # Convert ticker to permno
        ticker = pd.read_parquet(get_load_data_parquet_dir() / 'data_ticker.parquet.brotli')
        ticker = get_stocks_data(ticker, self.stock)
        ticker = ticker.reset_index()
        ticker['date'] = ticker['date'].dt.to_period('M')
        ticker = ticker.set_index(['permno', 'date'])
        ticker = ticker[~ticker.index.duplicated(keep='first')]
        ticker = ticker.reset_index()
        ticker.permno = ticker.permno.astype(int)
        ibes_permno = pd.merge(ticker, ibes_short, on=['ticker', 'date'], how='right')
        ibes_permno = ibes_permno.dropna(subset='permno').set_index(['permno', 'date'])
        ibes_permno = ibes_permno.sort_index(level=['permno', 'date'])
        # Calculate earning forecast disparity
        ibes_permno['tempShort'] = 100 * (ibes_permno['meanest'] - ibes_permno['fy0a']) / abs(ibes_permno['fy0a'])
        ibes_permno['earning_disparity'] = ibes_permno['fvyrgro'] - ibes_permno['tempShort']
        ibes_permno = ibes_permno.reset_index()
        ibes_permno.date = ibes_permno.date.dt.to_timestamp("M")
        ibes_permno = ibes_permno.set_index(['permno', 'date'])
        ibes_permno = ibes_permno[['earning_disparity']]
        self.factor_data = ibes_permno
