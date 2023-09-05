from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBSector(Factor):
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
        sector_df = yf.download(['XLSR', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU'], start=self.start, end=self.end)
        sector_df = sector_df.stack().swaplevel().sort_index()
        sector_df.index.names = ['ticker', 'date']
        sector_df = sector_df.astype(float)
        T = [1]
        sector_df = create_return(sector_df, T)
        sector_df = sector_df.drop(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        sector_df = sector_df.unstack('ticker').swaplevel(axis=1)
        sector_df.columns = ['_'.join(col).strip() for col in sector_df.columns.values]
        self.sector_data = sector_df

    @ray.remote
    def function(self, splice_data):
        # Add risk-free rate
        self.sector_data = pd.concat([self.sector_data, self.fama_data['RF']], axis=1)

        self.sector_data = self.sector_data.loc[self.start:self.end]
        self.sector_data = self.sector_data.fillna(0)

        # Get factor columns and create returns
        factors = self.sector_data.columns[:-1]
        T = [1]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        for t in T:
            ret = f'RET_{t:02}'

            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [60]
            for window in windows:
                betas = rolling_ols_sb(price=splice_data, factor_data=self.sector_data, factor_col=factors, window=window, name=f'{t:02}_SECTOR', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data