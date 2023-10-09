from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorSBPCA(Factor):
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
        self.fama_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_fama.parquet.brotli')
        pca_ret = self.factor_data.copy(deep=True)
        # Create returns and convert ticker index to columns
        pca_ret = create_return(pca_ret, windows=[1])
        ret = pca_ret[['RET_01']]
        ret = ret['RET_01'].unstack(pca_ret.index.names[0])
        ret.iloc[0] = ret.iloc[0].fillna(0)

        # Execute Rolling PCA
        window_size = 60
        num_components = 5
        self.pca_data = rolling_pca(data=ret, window_size=window_size, num_components=num_components, name='Return')
        self.pca_data = pd.concat([self.pca_data, self.fama_data['RF']], axis=1)
        self.pca_data = self.pca_data.loc[self.start:self.end]
        self.pca_data = self.pca_data.fillna(0)
        self.factor_col = self.pca_data.columns[:-1]

    @ray.remote
    def function(self, splice_data):
        T = [1]
        splice_data = create_return(splice_data, T)
        splice_data = splice_data.fillna(0)

        for t in T:
            ret = f'RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [30, 60]
            for window in windows:
                betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.pca_data, factor_col=self.factor_col, window=window, name=f'{t:02}_PCA_RET', ret=ret)
                splice_data = splice_data.join(betas)

        return splice_data