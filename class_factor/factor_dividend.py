from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorDividend(Factor):
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
        dividend = pd.read_parquet(get_parquet(self.live) / 'data_misc.parquet.brotli', columns=['paydt'])
        dividend = get_stocks_data(dividend, self.stock)
        dividend['paydt'] = pd.to_datetime(dividend['paydt'])
        self.factor_data = dividend

    @ray.remote
    def function(self, splice_data):
        def create_dividend(data):
            df = data.copy(deep=True)
            df['div_pay'] = 0

            for row in df.iterrows():
                if pd.notna(row[1]['paydt']):
                    current_date_year = row[1]['paydt'].year
                    current_date_month = row[1]['paydt'].month
                    mask = (df.index.get_level_values(1).year == current_date_year) & (df.index.get_level_values(1).month == current_date_month)
                    df.loc[mask, 'div_pay'] = 1

            return df[['div_pay']]

        collect = []
        for _, df in splice_data.groupby(self.group):
            collect.append(create_dividend(df))

        splice_data = pd.concat(collect, axis=0)
        return splice_data


