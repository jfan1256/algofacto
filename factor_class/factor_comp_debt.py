from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorCompDebt(Factor):
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
        columns = ['dlttq', 'dlcq']
        debt = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_q.parquet.brotli', columns=columns)
        debt = get_stocks_data(debt, self.stock)
        debt['tempBD'] = debt['dlttq'] + debt['dlcq']
        debt['comp_debt_iss'] = np.log(debt['tempBD'] / debt.groupby('permno')['tempBD'].shift(6))
        debt = debt[['comp_debt_iss']]
        self.factor_data = debt

        # columns = ['dltt', 'dlc']
        # debt = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns)
        # debt = get_stocks_data(debt, self.stock)
        # debt['tempBD'] = debt['dltt'] + debt['dlc']
        # debt['comp_debt_iss'] = np.log(debt['tempBD'] / debt.groupby('permno')['tempBD'].shift(60))
        # debt = debt[['comp_debt_iss']]
        # self.factor_data = debt
