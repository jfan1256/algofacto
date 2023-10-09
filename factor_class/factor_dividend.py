from typing import Optional, Union, List

import pandas as pd
import ray

from functions.utils.func import *
from factor_class.factor import Factor


class FactorDividend(Factor):
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
        dividend = pd.read_parquet(get_load_data_parquet_dir() / 'data_crsp.parquet.brotli', columns=['paydt'])
        dividend = get_stocks_data(dividend, stock)
        dividend['paydt'] = pd.to_datetime(dividend['paydt'])
        self.factor_data = dividend

    @ray.remote
    def function(self, splice_data):
        # def create_expdate(data):
        #     df = data.copy(deep=True)
        #     for row in df.iterrows():
        #         if row[1]['divpaydt'] is pd.NaT:
        #             continue
        #         current_date_year = row[1]['divpaydt'].year
        #         current_date_month = row[1]['divpaydt'].month
        #         mask = (df.index.get_level_values(1).year == current_date_year) & (df.index.get_level_values(1).month == current_date_month)
        #         df.loc[mask, 'expdate'] = 0
        #
        #     try:
        #         df_expdate = df.copy(deep=True)[['expdate']]
        #     except:
        #         df['expdate'] = -1
        #         df_expdate = df[['expdate']]
        #         return df_expdate
        #
        #     counter = 0
        #     in_sequence = False
        #     last_month = df_expdate.index.get_level_values(1).month
        #
        #     for date, row in df_expdate.iterrows():
        #         current_month = date[1].month
        #
        #         if row['expdate'] == 0:
        #             in_sequence = True
        #             counter = 0
        #         elif pd.isna(row['expdate']) and in_sequence and counter < 12:
        #             if current_month != last_month:
        #                 counter += 1
        #             df_expdate.loc[date, 'expdate'] = counter
        #         else:
        #             in_sequence = False
        #
        #         last_month = current_month
        #     df_expdate = df_expdate.fillna(-1)
        #     return df_expdate
        #
        # collect = []
        # for _, df in splice_data.groupby(self.group):
        #     collect.append(create_expdate(df))
        #
        # splice_data = pd.concat(collect, axis=0)
        # return splice_data
        def create_dividend(data):
            df = data.copy(deep=True)
            df['Dividend'] = 0  # Initialize the column with 0

            # For rows with valid dividend dates, set the corresponding month's expdate to 1
            for row in df.iterrows():
                if pd.notna(row[1]['paydt']):  # Checking for not NaT
                    current_date_year = row[1]['paydt'].year
                    current_date_month = row[1]['paydt'].month
                    mask = (df.index.get_level_values(1).year == current_date_year) & (df.index.get_level_values(1).month == current_date_month)
                    df.loc[mask, 'Dividend'] = 1

            return df[['Dividend']]

        collect = []
        for _, df in splice_data.groupby(self.group):
            collect.append(create_dividend(df))

        splice_data = pd.concat(collect, axis=0)
        return splice_data


