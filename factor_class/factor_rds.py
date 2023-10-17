from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRDS(Factor):
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
        # columns_compustat = ['gvkey', 'ceqq', 'niq', 'dvpq', 'rectaq', 'cshoq', 'prccq', 'msaq', 'cdvcy']
        # rds = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw.parquet.brotli', columns=columns_compustat)
        # rds = get_stocks_data(rds, self.stock)
        # rds['year'] = rds.index.get_level_values('date').year
        # rds = rds.reset_index()
        #
        # columns_pension = ['gvkey', 'pcupsu', 'paddml']
        # pension = pd.read_parquet(get_load_data_parquet_dir() / 'data_pension.parquet.brotli', columns=columns_pension)
        # pension['year'] = pension.index.get_level_values('date').year
        #
        # rds = rds.merge(pension[['gvkey', 'year', 'pcupsu', 'paddml']], on=['gvkey', 'year'], how='left')
        #
        # rds['rectaq'] = rds['rectaq'].fillna(0)
        # rds = rds.set_index(['permno', 'date'])
        # rds = rds.sort_index(level=['permno', 'date'])
        #
        # def calculate_rds(group):
        #     group['DS'] = (group['msaq'] - group['msaq'].shift(1)) + \
        #                   (group['rectaq'] - group['rectaq'].shift(1)) + \
        #                   0.65 * (group['pcupsu'].sub(group['paddml']).clip(upper=0) -
        #                           group['pcupsu'].shift(1).sub(group['paddml'].shift(1)).clip(upper=0))
        #     group['rds'] = (group['ceqq'] - group['ceqq'].shift(1)) - group['DS'] - \
        #                    (group['niq'] - group['dvpq']) + group['cdvcy'] - \
        #                    group['prccq'] * (group['cshoq'] - group['cshoq'].shift(1))
        #     return group
        #
        # rds = rds.groupby('permno').apply(calculate_rds).reset_index(level=0, drop=True)
        #
        # self.factor_data = rds[['rds']]

        columns_compustat = ['gvkey', 'ceq', 'ni', 'dvp', 'recta', 'csho', 'prcc_f', 'msa', 'cdvc']
        rds = pd.read_parquet(get_load_data_parquet_dir() / 'data_fund_raw_a.parquet.brotli', columns=columns_compustat)
        rds = get_stocks_data(rds, self.stock)
        rds['year'] = rds.index.get_level_values('date').year
        rds = rds.reset_index()

        columns_pension = ['gvkey', 'pcupsu', 'paddml']
        pension = pd.read_parquet(get_load_data_parquet_dir() / 'data_pension.parquet.brotli', columns=columns_pension)
        pension['year'] = pension.index.get_level_values('date').year

        rds = rds.merge(pension[['gvkey', 'year', 'pcupsu', 'paddml']], on=['gvkey', 'year'], how='left')

        rds['recta'] = rds['recta'].fillna(0)
        rds = rds.set_index(['permno', 'date'])
        rds = rds.sort_index(level=['permno', 'date'])

        def calculate_rds(group):
            group['DS'] = (group['msa'] - group['msa'].shift(1)) + \
                          (group['recta'] - group['recta'].shift(1)) + \
                          0.65 * (group['pcupsu'].sub(group['paddml']).clip(upper=0) -
                                  group['pcupsu'].shift(1).sub(group['paddml'].shift(1)).clip(upper=0))
            group['rds'] = (group['ceq'] - group['ceq'].shift(1)) - group['DS'] - \
                           (group['ni'] - group['dvp']) + group['cdvc'] - \
                           group['prcc_f'] * (group['csho'] - group['csho'].shift(1))
            return group

        rds = rds.groupby('permno').apply(calculate_rds).reset_index(level=0, drop=True)

        self.factor_data = rds[['rds']]
