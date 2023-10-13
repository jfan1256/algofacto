from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRFRet(Factor):
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
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')

        # PCA
        pca_df = self.factor_data.copy(deep=True)
        # Create returns and convert ticker index to columns
        pca_df = create_return(pca_df, windows=[1])
        ret = pca_df[['RET_01']]
        ret = ret['RET_01'].unstack(pca_df.index.names[0])

        # Execute Rolling PCA
        window_size = 60
        num_components = 5
        pca_df = rolling_pca(data=ret, window_size=window_size, num_components=num_components, name='Return')

        # Sector ETF
        sector_df = yf.download(['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU'], start=self.start, end=self.end)
        sector_df = sector_df.stack().swaplevel().sort_index()
        sector_df.index.names = ['ticker', 'date']
        sector_df = sector_df.astype(float)
        T = [1]
        sector_df = create_return(sector_df, T)
        sector_df = sector_df.drop(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        sector_df = sector_df.unstack('ticker').swaplevel(axis=1)
        sector_df.columns = ['_'.join(col).strip() for col in sector_df.columns.values]

        # Lag Bond
        bond_df = yf.download(['TLT', 'TIP', 'SHY'], start=self.start, end=self.end)
        bond_df = bond_df.stack().swaplevel().sort_index()
        bond_df.index.names = ['ticker', 'date']
        bond_df = bond_df.astype(float)
        T = [1, 21, 126]
        bond_df = create_return(bond_df, T)
        bond_df = bond_df.drop(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
        bond_df = bond_df.unstack('ticker').swaplevel(axis=1)
        bond_df.columns = ['_'.join(col).strip() for col in bond_df.columns.values]

        # Fama
        fama_df = pd.read_parquet(get_load_data_parquet_dir() / 'data_fama.parquet.brotli')
        fama_df_no_rf = fama_df.drop('RF', axis=1)
        rf = fama_df[['RF']].loc[self.start:self.end]

        all_rf = pd.concat([pca_df, sector_df, bond_df, fama_df_no_rf], axis=1)
        all_rf = all_rf.loc[self.start:self.end]

        # Execute Rolling PCA
        window_size = 21
        num_components = 5
        self.all_rf = rolling_pca(data=all_rf, window_size=window_size, num_components=num_components, name='all_rf')
        self.all_rf = pd.concat([self.all_rf, rf], axis=1)
        self.all_rf = self.all_rf.fillna(0)
        self.factor_col = self.all_rf.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        T = [1, 21, 126]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        for t in T:
            ret = f'RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [21]
            for window in windows:
                betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.all_rf, factor_col=self.factor_col, window=window, name=f'{t:02}_RF_RET', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data
