from typing import Optional, Union, List

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
                 stock: Optional[Union[List[str], str]] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 join: str = None,
                 general: bool = False,
                 window: int = None,
                 component: int = None):
        super().__init__(file_name, skip, start, end, stock, batch_size, splice_size, group, join, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.component = component
        # Create returns and convert stock index to columns
        self.factor_data = create_return(self.factor_data, windows=[1])
        self.factor_data = self.factor_data[[f'RET_01']]
        start_date = datetime.strptime(self.start, '%Y-%m-%d')
        new_start_date = start_date + timedelta(days=0)
        new_start_str = new_start_date.strftime('%Y-%m-%d')
        self.factor_data = set_timeframe(self.factor_data, new_start_str, self.end)
        self.factor_data = self.factor_data['RET_01'].unstack(self.join)


    @ray.remote
    def function(self, splice_data):
        # Normalize data
        splice_data = (splice_data - splice_data.mean()) / splice_data.std()

        # Drop columns that have more than half of missing data
        splice_data = splice_data.drop(columns=splice_data.columns[splice_data.isna().sum() > len(splice_data) / 2])
        splice_data = splice_data.fillna(0)

        # Get loadings
        pca = PCA(n_components=self.component, random_state=42)
        pca.fit_transform(splice_data)
        loading = pca.components_.T * np.sqrt(pca.explained_variance_)
        # Create a dataframe that matches loadings to stock
        cols = splice_data.columns
        date = splice_data.index[0]
        splice_data = pd.DataFrame(loading, columns=[f'load_ret_{i + 1}' for i in range(5)], index=[[date] * len(cols), cols])
        splice_data.index.names = ['date', self.join]
        return splice_data
