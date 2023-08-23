from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorLoadRet(Factor):
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
                 window: int = None,
                 component: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.component = component
        # Create returns and convert ticker index to columns
        self.factor_data = create_return(self.factor_data, windows=[1])
        self.factor_data = self.factor_data[[f'RET_01']]
        self.factor_data = self.factor_data['RET_01'].unstack('ticker')


    @ray.remote
    def function(self, splice_data):
        # Normalize data
        splice_data = (splice_data - splice_data.mean()) / splice_data.std()

        # Drop columns that have more than half of missing data
        splice_data = splice_data.drop(columns=splice_data.columns[splice_data.isna().sum() > len(splice_data) / 2])
        splice_data = splice_data.fillna(0)

        # Get loadings
        pca = PCA(n_components=self.component)
        pca.fit_transform(splice_data)
        loading = pca.components_.T * np.sqrt(pca.explained_variance_)
        # Create a dataframe that matches loadings to ticker
        cols = splice_data.columns
        date = splice_data.index[0]
        splice_data = pd.DataFrame(loading, columns=[f'ret_loading_{i + 1}' for i in range(5)], index=[[date] * len(cols), cols])
        splice_data.index.names = ['date', 'ticker']
        return splice_data
