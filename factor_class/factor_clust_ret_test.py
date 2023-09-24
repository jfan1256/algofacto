from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorClustRetTest(Factor):
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
                 window: int = None,
                 cluster: int = None):
        super().__init__(file_name, skip, start, end, stock, batch_size, splice_size, group, join, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.cluster = cluster
        # Create returns and convert stock index to columns
        window_size = 10
        self.factor_data = create_smooth_return(self.factor_data, windows=[1], window_size=window_size)
        self.factor_data = self.factor_data[[f'RET_01']]
        self.factor_data = self.factor_data['RET_01'].unstack(self.join)
        self.factor_data.iloc[:window_size + 1] = self.factor_data.iloc[:window_size + 1].fillna(0)

    @ray.remote
    def function(self, splice_data):
        # Normalize data
        splice_data = (splice_data - splice_data.mean()) / splice_data.std()

        # Drop columns that have more than half of missing data
        # splice_data = splice_data.drop(columns=splice_data.columns[splice_data.isna().sum() > len(splice_data) / 2])
        splice_data = splice_data.fillna(0)

        # Run kmeans
        kmeans = KMeans(n_clusters=self.cluster, init='k-means++', random_state=0, n_init=10)
        cluster = kmeans.fit_predict(splice_data.T)

        # Create a dataframe that matches cluster to ticker
        cols = splice_data.columns
        date = splice_data.index[0]
        splice_data = pd.DataFrame(cluster, columns=[f'ret_cluster'], index=[[date] * len(cols), cols])
        splice_data.index.names = ['date', self.join]
        return splice_data