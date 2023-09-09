from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorMFRet(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 ticker: List[str] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.fama_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_fama.parquet.brotli')
        # self.macro_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_macro.parquet.brotli')
        self.etf_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_etf.parquet.brotli')
        bond_df = yf.download(['TLT', 'TIP', 'SHY'], start=self.start, end=self.end)
        bond_df = bond_df.stack().swaplevel().sort_index()
        bond_df.index.names = ['ticker', 'date']
        bond_df = bond_df.astype(float)
        T = [1, 6, 30]
        bond_df = create_return(bond_df, T)
        bond_df = bond_df.drop(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        bond_df = bond_df.unstack('ticker').swaplevel(axis=1)
        bond_df.columns = ['_'.join(col).strip() for col in bond_df.columns.values]
        self.bond_data = bond_df
        self.mf = pd.concat([self.etf_data, self.bond_data, self.fama_data], axis=1)
        self.mf = self.mf.loc[self.start:self.end]
        self.mf = self.mf.fillna(0)
        self.factor_col = self.mf.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        T = [1]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        for t in T:
            ret = f'RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [30, 60]
            for window in windows:
                betas = rolling_ols_residual(price=splice_data, factor_data=self.mf, factor_col=self.factor_col, window=window, name=f'{t:02}_MF_RET', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data