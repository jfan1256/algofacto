from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor
from scipy.stats.mstats import winsorize

class FactorAbnormalAccrual(Factor):
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
        columns = ['fopt', 'oancf', 'act', 'che', 'dlc', 'lct', 'ib', 'sale', 'at', 'ppegt', 'sich', 'fyear', 'ni']
        accrual = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli', columns=columns)
        accrual = get_stocks_data(accrual, stock)

        # Define a function to compute temp columns based on permno grouping
        def compute_temp_cols(group):
            group['tempCFO'] = group['oancf'].fillna(group['fopt'] - (group['act'] - group['act'].shift(1))
                                                      + (group['che'] - group['che'].shift(1))
                                                      - (group['lct'] - group['lct'].shift(1))
                                                      + (group['dlc'] - group['dlc'].shift(1)))

            group['tempAccruals'] = (group['ib'] - group['tempCFO']) / group['at'].shift(1)
            group['tempInvTA'] = 1 / group['at'].shift(1)
            group['tempDelRev'] = (group['sale'] - group['sale'].shift(1)) / group['at'].shift(1)
            group['tempPPE'] = group['ppegt'] / group['at'].shift(1)
            return group

        # Apply the function
        merged_data = accrual.groupby('permno').apply(compute_temp_cols).reset_index(level=0, drop=True)
        merged_data = merged_data.sort_values(by='fyear')

        # Winsorize Data
        for column in merged_data.columns:
            if column.startswith('temp'):
                merged_data[column] = winsorize(merged_data[column], limits=[0.001, 0.001])

        merged_data['sic2'] = merged_data['sich'] // 100

        def run_regression(group):
            # Check if the group size is too small or if there are missing values
            if group.shape[0] < 2 or group[['tempAccruals', 'tempInvTA', 'tempDelRev', 'tempPPE']].isnull().any().any():
                group['fitted'] = None
                return group

            X = group[['tempInvTA', 'tempDelRev', 'tempPPE']]
            X = sm.add_constant(X)  # Adds a constant column for intercept
            y = group['tempAccruals']

            # Run regression only if there's no missing data
            if not X.isnull().values.any() and not y.isnull().values.any():
                model = sm.OLS(y, X, missing='drop').fit()
                group['AbnormalAccruals'] = model.fittedvalues
            else:
                group['AbnormalAccruals'] = None

            return group

        result = merged_data.groupby(['fyear', 'sic2']).apply(run_regression).reset_index(level=[0, 1], drop=True)

        group_counts = result.groupby(['fyear', 'sic2']).size()
        drop_indices = group_counts[group_counts < 6].index
        for idx in drop_indices:
            result = result.drop(result[(result['fyear'] == idx[0]) & (result['sic2'] == idx[1])].index)

        result = result.rename(columns={'_residuals': 'AbnormalAccruals'})
        # Drop duplicates
        result = result.sort_values(by=['permno', 'fyear'])
        result = result.reset_index()
        result = result.drop_duplicates(subset=['permno', 'fyear'], keep='first')

        # Compute AbnormalAccrualsPercent
        result['abnormal_accrual_pct'] = result['AbnormalAccruals'] * result['at'].shift(1) / abs(result['ni'])
        result = result.sort_index(level=['permno', 'date'])
        result = result.reset_index()

        # Expanding to monthly
        result = result.loc[result.index.repeat(12)]
        result['date'] = result.groupby('permno').date.transform(lambda x: pd.date_range(start=x.iat[0], periods=len(x), freq='M'))
        result.set_index(['permno', 'date'], inplace=True)
        result = result.groupby(['permno', 'date']).last()
        result = result.sort_index(level=['permno', 'date'])
        self.factor_data = result[['abnormal_accrual_pct']]
