from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorFundRaw(Factor):
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
        columns = ['atq', 'lctq', 'cheq', 'ivstq', 'ltq', 'ceqq', 'niq', 'saleq', 'cogsq', 'invtq', 'apq', 'prccq', 'cshoq', 'dpq', 'xintq', 'piq', 'revtq']
        fund_raw = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns)
        fund_raw = get_stocks_data(fund_raw, stock)
        fund_raw['current_ratio'] = fund_raw['atq'] / fund_raw['lctq']
        fund_raw['quick_ratio'] = (fund_raw['cheq'] + fund_raw['ivstq']) / fund_raw['lctq']
        fund_raw['cash_ratio'] = fund_raw['cheq'] / fund_raw['lctq']
        fund_raw['debt_equity_ratio'] = fund_raw['ltq'] / fund_raw['ceqq']
        fund_raw['equity_multiplier'] = fund_raw['atq'] / fund_raw['ceqq']
        fund_raw['return_on_assets'] = fund_raw['niq'] / fund_raw['atq']
        fund_raw['return_on_equity'] = fund_raw['niq'] / fund_raw['ceqq']
        fund_raw['gross_profit_margin'] = (fund_raw['saleq'] - fund_raw['cogsq']) / fund_raw['saleq']
        fund_raw['net_profit_margin'] = fund_raw['niq'] / fund_raw['saleq']
        fund_raw['asset_turnover'] = fund_raw['saleq'] / fund_raw['atq']
        fund_raw['inventory_turnover'] = fund_raw['cogsq'] / fund_raw['invtq']
        fund_raw['payable_turnover'] = fund_raw['cogsq'] / fund_raw['apq']
        fund_raw['book_to_market'] = fund_raw['ceqq'] / (fund_raw['prccq'] * fund_raw['cshoq'])
        fund_raw['price_to_earnings'] = fund_raw['prccq'] / (fund_raw['niq'] / fund_raw['cshoq'])
        fund_raw['ev_to_ebitda'] = (fund_raw['ltq'] + fund_raw['ceqq'] + (fund_raw['prccq'] * fund_raw['cshoq'])) / (fund_raw['niq'] + fund_raw['dpq'] + fund_raw['xintq'])
        fund_raw['debt_ratio'] = fund_raw['ltq'] / fund_raw['atq']
        fund_raw['roic'] = fund_raw['niq'] / (fund_raw['ceqq'] + fund_raw['ltq'])
        fund_raw['financial_leverage'] = fund_raw['atq'] / fund_raw['ceqq']
        fund_raw['interest_coverage_ratio'] = fund_raw['piq'] / fund_raw['xintq']
        fund_raw['roe'] = fund_raw['niq'] / fund_raw['ceqq']
        fund_raw['roa'] = fund_raw['niq'] / fund_raw['atq']
        fund_raw['gross_margin'] = (fund_raw['saleq'] - fund_raw['cogsq']) / fund_raw['saleq']
        fund_raw['net_margin'] = fund_raw['niq'] / fund_raw['revtq']
        fund_raw['asset_utilization'] = fund_raw['saleq'] / fund_raw['atq']
        fund_raw['days_of_inventory_on_hand'] = 365 / fund_raw['inventory_turnover']
        fund_raw['payables_turnover'] = fund_raw['cogsq'] / fund_raw['apq']
        fund_raw['days_of_payables_outstanding'] = 365 / fund_raw['payables_turnover']

        # Handling division by zero and replacing inf with NaN
        fund_raw = fund_raw.drop(columns, axis=1)
        fund_raw = fund_raw.replace([np.inf, -np.inf], np.nan)
        self.factor_data = fund_raw
