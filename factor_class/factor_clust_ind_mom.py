from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorClustIndMom(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 ticker: Optional[Union[List[str], str]] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 general: bool = False,
                 window: int = None,
                 cluster: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        price_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        ind_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_ind.parquet.brotli')
        combine = pd.concat([price_data, ind_data], axis=1)

        t = 1
        ret = create_return(combine, windows=[t])[[f'RET_{t:02}', 'Industry']]
        avg_ret = ret.groupby(['Industry', ret.index.get_level_values('date')])[f'RET_{t:02}'].mean()
        ret = ret.reset_index()
        ret = pd.merge(ret, avg_ret.rename('indRET').reset_index(), on=['Industry', 'date'], how='left')
        ret[f'IndMom_{t:02}'] = ret[f'RET_{t:02}'] / ret['indRET']
        ind_mom = ret.set_index(['ticker', 'date'])[[f'IndMom_{t:02}']]
        self.factor_data = ind_mom
        self.cluster = cluster
        # Create returns and convert ticker index to columns
        self.factor_data = self.factor_data['IndMom_01'].unstack('ticker')

    @ray.remote
    def function(self, splice_data):
        # Normalize data
        splice_data = (splice_data - splice_data.mean()) / splice_data.std()

        # Drop columns that have more than half of missing data
        splice_data = splice_data.drop(columns=splice_data.columns[splice_data.isna().sum() > len(splice_data) / 2])
        splice_data = splice_data.fillna(0)

        # Run kmeans
        kmeans = KMeans(n_clusters=self.cluster, init='k-means++', random_state=0, n_init=10)
        cluster = kmeans.fit_predict(splice_data.T)

        # Create a dataframe that matches cluster to ticker
        cols = splice_data.columns
        date = splice_data.index[0]
        splice_data = pd.DataFrame(cluster, columns=[f'ind_mom_cluster'], index=[[date] * len(cols), cols])
        splice_data.index.names = ['date', 'ticker']
        return splice_data
