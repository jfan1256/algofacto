from typing import Optional, Union, List

from functions.utils.func import *
from class_factor.factor import Factor


class FactorClustRetComp(Factor):
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
                 window: int = None,
                 cluster: int = None):
        super().__init__(live, file_name, skip, start, end, stock, batch_size, splice_size, group, join, general, window)
        self.factor_data = pd.read_parquet(get_factor(self.live) / 'factor_ret_comp.parquet.brotli')
        self.cluster = cluster
        self.factor_data = self.factor_data.drop(['Open', 'Close', 'Low', 'Volume', 'High'], axis=1)
        start_date = datetime.strptime(self.start, '%Y-%m-%d')
        new_start_date = start_date + timedelta(days=0)
        new_start_str = new_start_date.strftime('%Y-%m-%d')
        self.factor_data = set_timeframe(self.factor_data, new_start_str, self.end)
        self.factor_data = self.factor_data.unstack(self.join)

    @ray.remote
    def function(self, splice_data):
        clust_collect = []
        for i, col in enumerate(splice_data.columns.get_level_values(0).unique()):
            data = splice_data[col]
            # Normalize data
            data = (data - data.mean()) / data.std()

            # Drop columns that have more than half of missing data
            data = data.drop(columns=data.columns[data.isna().sum() > len(data) / 2])
            data = data.fillna(0)

            # Run kmeans
            kmeans = KMeans(n_clusters=self.cluster, init='k-means++', random_state=0, n_init=10)
            cluster = kmeans.fit_predict(data.T)

            # Create a dataframe that matches cluster to stock
            cols = data.columns
            date = data.index[0]
            data = pd.DataFrame(cluster, columns=[f'{col}_cluster'], index=[[date] * len(cols), cols])
            data.index.names = ['date', self.join]
            clust_collect.append(data)
            break

        splice_data = pd.concat(clust_collect, axis=1)
        return splice_data