from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor


class FactorEarningStreak(Factor):
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
        # Read in actual summary and summary statistic files from WRDS
        actual = pd.read_csv(get_large(self.live) / 'summary_actual_adj_ibes.csv')
        statistic = pd.read_csv(get_large(self.live) / 'summary_statistic_adj_ibes.csv')
        actual.columns = actual.columns.str.lower()
        statistic.columns = statistic.columns.str.lower()
        actual = actual.set_index(['ticker', 'statpers'])
        statistic = statistic.set_index(['ticker', 'statpers'])
        # Merge on oftic and statpers indices
        combined = pd.merge(actual, statistic, left_index=True, right_index=True, how='left')
        combined = combined.reset_index()
        # Use actual release date as date of availability
        combined['date'] = pd.to_datetime(combined['anndats_act']).dt.to_period('M')
        # Sort the data
        combined = combined.sort_values(by=['ticker', 'date', 'anndats_act', 'statpers'])
        # Keep the last forecast before the actual release
        combined = combined.groupby(['ticker', 'date']).last().reset_index()
        # Define Surp (positive / negative surprise) and Streak (consistent Surp)
        combined['surp'] = (combined['actual'] - combined['meanest']) / combined['price']
        combined = combined.sort_values(by=['ticker', 'anndats_act'])
        combined['streak'] = (combined['surp'].gt(0) == combined.groupby('ticker')['surp'].shift(1).gt(0)).astype(int)
        # Convert to Positive Streak vs Negative Streak
        combined = combined[combined['streak'] == 1]
        # Convert ticker to permno
        ticker = pd.read_parquet(get_parquet(self.live) / 'data_ticker.parquet.brotli')
        ticker = get_stocks_data(ticker, self.stock)
        ticker = ticker.reset_index()
        ticker['date'] = ticker['date'].dt.to_period('M')
        ticker = ticker.set_index(['permno', 'date'])
        ticker = ticker[~ticker.index.duplicated(keep='first')]
        ticker = ticker.reset_index()
        ibes_permno = pd.merge(ticker, combined, on=['ticker', 'date'], how='right')
        ibes_permno = ibes_permno.dropna(subset='permno')
        ibes_permno.permno = ibes_permno.permno.astype(int)
        ibes_permno = ibes_permno.dropna(subset='permno').set_index(['permno', 'date'])
        ibes_permno = ibes_permno.sort_index(level=['permno', 'date'])
        # Replace anndats_act column's NaN values with previous non-NaN value
        ibes_permno['anndats_act'] = ibes_permno['anndats_act'].fillna(method='ffill')
        # Drop rows where anndats_act is NaN or the difference between time_avail_m and month of anndats_act is more than 6
        ibes_permno = ibes_permno.dropna(subset=['anndats_act'])
        index_timestamp = ibes_permno.index.get_level_values(1).to_timestamp()
        anndats_act_timestamp = pd.to_datetime(ibes_permno['anndats_act'])
        ibes_permno['month_diff'] = (index_timestamp - anndats_act_timestamp).dt.days / 30.44
        ibes_permno = ibes_permno[ibes_permno['month_diff'] <= 6]
        # Assign the signal
        ibes_permno['earning_streak'] = ibes_permno['surp'].where(ibes_permno['streak'] == 1).fillna(method='ffill')
        ibes_permno = ibes_permno.reset_index()
        ibes_permno.date = ibes_permno.date.dt.to_timestamp("M")
        ibes_permno = ibes_permno.set_index(['permno', 'date'])
        ibes_permno = ibes_permno[['earning_streak']]
        self.factor_data = ibes_permno