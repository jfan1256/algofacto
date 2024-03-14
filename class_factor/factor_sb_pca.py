from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor

class FactorSBPCA(Factor):
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
        self.factor_data = pd.read_parquet(get_parquet(self.live) / 'data_price.parquet.brotli')
        self.risk_free = pd.read_parquet(get_parquet(self.live) / 'data_rf.parquet.brotli')
        pca_ret = self.factor_data.copy(deep=True)
        # Create returns and convert ticker index to columns
        pca_ret = create_return(pca_ret, windows=[1])
        ret = pca_ret[['RET_01']]
        ret = ret['RET_01'].unstack(pca_ret.index.names[0])

        # Execute Rolling PCA
        window_size = 21
        num_components = 5
        self.pca_data = rolling_pca(data=ret, window_size=window_size, num_components=num_components, name='Return')
        self.pca_data = pd.concat([self.pca_data, self.risk_free['RF']], axis=1)
        self.pca_data = self.pca_data.loc[self.start:self.end]
        self.pca_data['RF'] = self.pca_data['RF'].ffill()
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
            windows = [60]
            for window in windows:
                # betas = rolling_ols_beta_res_syn(price=splice_data, factor_data=self.pca_data, factor_col=self.factor_col, window=window, name=f'ret_pca_{t:02}', ret=ret)
                betas = rolling_ols_parallel(data=splice_data, ret=ret, factor_data=self.pca_data, factor_cols=self.factor_col.tolist(), window=window, name=f'ret_pca_{t:02}')
                splice_data = splice_data.join(betas)

        return splice_data