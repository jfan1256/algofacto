from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorAccrualBM(Factor):
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
        # columns = ['ceqq', 'actq', 'dlcq', 'cheq', 'lctq', 'atq', 'txpq']
        # accrual_bm = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        # outstanding = ['market_cap']
        # price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_crsp.parquet.brotli', columns=outstanding)
        # accrual_bm = accrual_bm.sort_index()
        # price_data = price_data.sort_index()
        # price_data_reindexed = price_data.reindex(accrual_bm.index, method='ffill')
        # accrual_bm = accrual_bm.merge(price_data_reindexed, left_index=True, right_index=True)
        # accrual_bm = get_stocks_data(accrual_bm, self.stock)
        # # Replace missing values in 'txp' with 0
        # accrual_bm['BM'] = np.log(accrual_bm['ceqq'] / accrual_bm['market_cap'])
        #
        # # Calculate 'tempacc' with groupby 'permno' for shifting
        # accrual_bm['tempacc'] = (
        #                         (accrual_bm.groupby('permno')['actq'].transform(lambda x: x - x.shift(12))) -
        #                         (accrual_bm.groupby('permno')['cheq'].transform(lambda x: x - x.shift(12))) -
        #                         ((accrual_bm.groupby('permno')['lctq'].transform(lambda x: x - x.shift(12))) -
        #                          (accrual_bm.groupby('permno')['dlcq'].transform(lambda x: x - x.shift(12))) -
        #                          (accrual_bm.groupby('permno')['txpq'].transform(lambda x: x - x.shift(12))))
        #                 ) / ((accrual_bm['atq'] + accrual_bm.groupby('permno')['atq'].transform(lambda x: x.shift(12))) / 2)
        #
        # # Calculate quantiles for 'BM' and 'tempacc' based on 'time_avail_m'
        # def safe_qcut(x):
        #     if len(x.dropna()) < 5:
        #         return pd.Series([np.nan] * len(x), index=x.index)
        #     else:
        #         return pd.qcut(x, 5, labels=False) + 1
        #
        # # Apply the safe_qcut function
        # accrual_bm['tempqBM'] = accrual_bm.groupby('date')['BM'].transform(safe_qcut)
        # accrual_bm['tempqAcc'] = accrual_bm.groupby('date')['tempacc'].transform(safe_qcut)
        #
        # # Construct 'AccrualsBM' column
        # accrual_bm['accruals_bm'] = np.where((accrual_bm['tempqBM'] == 5) & (accrual_bm['tempqAcc'] == 1), 1,
        #                             np.where((accrual_bm['tempqBM'] == 1) & (accrual_bm['tempqAcc'] == 5), 0, np.nan))
        # accrual_bm.loc[accrual_bm['ceqq'] < 0, 'accruals_bm'] = np.nan
        # self.factor_data = accrual_bm[['accruals_bm']]

        columns = ['ceq', 'act', 'dlc', 'che', 'lct', 'at', 'txp']
        accrual_bm = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns)
        outstanding = ['market_cap']
        price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_crsp.parquet.brotli', columns=outstanding)
        accrual_bm = accrual_bm.sort_index()
        price_data = price_data.sort_index()
        price_data_reindexed = price_data.reindex(accrual_bm.index, method='ffill')
        accrual_bm = accrual_bm.merge(price_data_reindexed, left_index=True, right_index=True)
        accrual_bm = get_stocks_data(accrual_bm, self.stock)
        # Replace missing values in 'txp' with 0
        accrual_bm['BM'] = np.log(accrual_bm['ceq'] / accrual_bm['market_cap'])

        # Calculate 'tempacc' with groupby 'permno' for shifting
        accrual_bm['tempacc'] = (
                                        (accrual_bm.groupby('permno')['act'].transform(lambda x: x - x.shift(12))) -
                                        (accrual_bm.groupby('permno')['che'].transform(lambda x: x - x.shift(12))) -
                                        ((accrual_bm.groupby('permno')['lct'].transform(lambda x: x - x.shift(12))) -
                                         (accrual_bm.groupby('permno')['dlc'].transform(lambda x: x - x.shift(12))) -
                                         (accrual_bm.groupby('permno')['txp'].transform(lambda x: x - x.shift(12))))
                                ) / ((accrual_bm['at'] + accrual_bm.groupby('permno')['at'].transform(lambda x: x.shift(12))) / 2)

        # Calculate quantiles for 'BM' and 'tempacc' based on 'time_avail_m'
        def safe_qcut(x):
            if len(x.dropna()) < 5:
                return pd.Series([np.nan] * len(x), index=x.index)
            else:
                return pd.qcut(x, 5, labels=False) + 1

        # Apply the safe_qcut function
        accrual_bm['tempqBM'] = accrual_bm.groupby('date')['BM'].transform(safe_qcut)
        accrual_bm['tempqAcc'] = accrual_bm.groupby('date')['tempacc'].transform(safe_qcut)

        # Construct 'AccrualsBM' column
        accrual_bm['accruals_bm'] = np.where((accrual_bm['tempqBM'] == 5) & (accrual_bm['tempqAcc'] == 1), 1,
                                             np.where((accrual_bm['tempqBM'] == 1) & (accrual_bm['tempqAcc'] == 5), 0, np.nan))
        accrual_bm.loc[accrual_bm['ceq'] < 0, 'accruals_bm'] = np.nan
        self.factor_data = accrual_bm[['accruals_bm']]
