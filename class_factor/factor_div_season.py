from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorDivSeason(Factor):
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
        dividend = pd.read_parquet(get_parquet(self.live) / 'data_misc.parquet.brotli', columns=['distcd', 'divamt'])
        dividend = get_stocks_data(dividend, self.stock)
        # Convert to string and pad to 4 characters
        dividend['distcd_str'] = dividend['distcd'].astype(str).str.pad(4, fillchar=' ')

        # Extract individual characters and apply lambda function
        dividend['cd1'] = dividend['distcd_str'].str[0].apply(lambda x: float(x) if x.isnumeric() else pd.NA)
        dividend['cd2'] = dividend['distcd_str'].str[1].apply(lambda x: float(x) if x.isnumeric() else pd.NA)
        dividend['cd3'] = dividend['distcd_str'].str[2].apply(lambda x: float(x) if x.isnumeric() else pd.NA)
        dividend['cd4'] = dividend['distcd_str'].str[3].apply(lambda x: float(x) if x.isnumeric() else pd.NA)

        # Optional: Drop the temporary 'distcd_str' column
        dividend = dividend.drop(columns=['distcd_str'])

        # Calculating DivSeason
        dividend['divpaid'] = dividend['divamt'] > 0
        
        condition_3 = (dividend['cd3'].isin([3, 0, 1])) & (
                    dividend.groupby('permno')['divpaid'].shift(2) | dividend.groupby('permno')['divpaid'].shift(5) | dividend.groupby('permno')['divpaid'].shift(8) | dividend.groupby('permno')['divpaid'].shift(11))
        condition_4 = (dividend['cd3'] == 4) & (dividend.groupby('permno')['divpaid'].shift(5) | dividend.groupby('permno')['divpaid'].shift(11))
        condition_5 = (dividend['cd3'] == 5) & dividend.groupby('permno')['divpaid'].shift(11)

        dividend['div_season'] = 0
        dividend.loc[condition_3 | condition_4 | condition_5, 'div_season'] = 1
        self.factor_data = dividend[['div_season']]

