from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorNetDebtFinance(Factor):
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
        columns = ['dlcchy', 'dltisy', 'dltry', 'atq']
        net_debt = pd.read_parquet(get_parquet_dir(self.live) / 'data_fund_raw_q.parquet.brotli', columns=columns)
        net_debt = get_stocks_data(net_debt, self.stock)

        net_debt['net_debt_fin'] = (net_debt['dltisy'] - net_debt['dltry'] + net_debt['dlcchy']) / (0.5 * (net_debt['atq'] + net_debt.groupby('permno')['atq'].shift(6)))
        net_debt.loc[net_debt['net_debt_fin'].abs() > 1, 'net_debt_fin'] = None
        net_debt = net_debt[['net_debt_fin']]
        self.factor_data = net_debt
