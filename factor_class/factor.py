from typing import List

from functions.utils.func import *
from functions.utils.system import *
from itertools import chain

from typing import Union, Optional, List

import warnings

warnings.filterwarnings('ignore')


class Factor:
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
                 window: int = None):
        self.factor_data = None
        self.file_name = file_name
        self.skip = skip
        self.ticker = ticker
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.splice_size = splice_size
        self.group = group
        self.general = general
        self.window = window

        assert group == 'ticker' or group == 'date' or group is None, ValueError('group parameter must be either ''ticker'' or ''date''')
        if self.group == 'date' and self.ticker:
            raise ValueError('if group parameter is set to date then ticker parameter must be None')

    # Get rolling window groups
    def _rolling_window(self):
        dates = self.factor_data.index
        cols = self.factor_data.columns
        np_data = self.factor_data.to_numpy()
        shape = (np_data.shape[0] - self.window + 1, self.window, np_data.shape[1])
        strides = (np_data.strides[0], np_data.strides[0], np_data.strides[1])
        window_data = np.lib.stride_tricks.as_strided(np_data, shape=shape, strides=strides)
        window_data_dict = {date: window_data[idx] for idx, date in enumerate(dates[self.window - 1:])}
        window_df = [pd.DataFrame(data=item, index=[key] * len(item), columns=cols) for key, item in window_data_dict.items()]
        return window_df

    # Creating multi index
    def _create_multi_index(self, tickers):
        factor_values = pd.concat([self.factor_data] * len(tickers), ignore_index=True).values
        multi_index = pd.MultiIndex.from_product([tickers, self.factor_data.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=self.factor_data.columns, index=multi_index)
        multi_index_factor.index = multi_index_factor.index.set_names(['ticker', 'date'])
        return multi_index_factor

    # Splice data into smaller dataframes of with size (splice_size)
    def _splice_data(self):
        data_spliced = {}
        splice = 1

        if self.general:
            if 'ticker' in self.factor_data.index.names:
                raise TypeError('if general parameter is set to True then there cannot be ''ticker'' in the index')
            self.factor_data = self._create_multi_index(self.ticker)

        if self.group == 'ticker':
            count = 0
            splice_all = []

            for _, df in self.factor_data.groupby('ticker', group_keys=False):
                splice_all.append(df)
                count += 1
                if count == self.splice_size:
                    name = f'splice{splice}'
                    data_spliced[name] = pd.concat(splice_all, axis=0)
                    splice_all = []
                    splice += 1
                    count = 0
            if splice_all:  # Concatenate any remaining data
                name = f'splice{splice}'
                data_spliced[name] = pd.concat(splice_all, axis=0)
            return data_spliced

        elif self.group == 'date':
            if 'ticker' in self.factor_data.index.names:
                raise ValueError('if group parameter is set to ''date'' then ''ticker'' cannot be in the index. Must only ''date'' in index')

            window_data = self._rolling_window()
            for i in range(0, len(window_data), self.splice_size):
                name = f'splice{splice}'
                data_spliced[name] = window_data[i:i + self.splice_size]
                splice += 1
            return data_spliced

    # Converts each splice into a batch of multiple splices with (batch_size)
    def _batch_data(self, splice_data):
        batch = []
        factor_batch = {}
        batch_num = 1
        count = 1
        for i, item in enumerate(splice_data):
            batch.append(splice_data[item])
            if count == self.batch_size:
                name = f'batch{batch_num}'
                if self.group == 'ticker':
                    factor_batch[name] = batch
                elif self.group == 'date':
                    factor_batch[name] = list(chain.from_iterable(batch))
                batch_num = batch_num + 1
                count = 0
                batch = []
            count = count + 1

        name = f'batch{batch_num}'  # Excess data
        if self.group == 'ticker':
            factor_batch[name] = batch
        elif self.group == 'date':
            factor_batch[name] = list(chain.from_iterable(batch))
        return factor_batch

    # Feed in batch data and (function) will execute on all splices within the batch at the same time
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Executing creation', animation=spinner_animation)
    def _execute_creation(self, operation, batch_data):
        results = ray.get([operation.remote(self, splice_data) for splice_data in batch_data])
        return results

    # Executes (function) across all batches and exports the created factor
    def parallel_processing(self, batch_data):
        print(f"Creating {self.file_name}...")
        start_time = time.time()
        nested_data_all = []

        for i, item in enumerate(batch_data):
            i += 1
            data = (self._execute_creation(self.function, batch_data[item]))
            nested_data_all.append(data)
            print(f"Completed batch: {str(i)}")

        flattened_data_all = list(chain.from_iterable(nested_data_all))
        factor_data = pd.concat(flattened_data_all, axis=0)
        factor_data = factor_data.reset_index(['ticker', 'date'])
        factor_data = factor_data.set_index(['ticker', 'date'])
        factor_data = factor_data.sort_index(level=['ticker', 'date'])
        print(f"Exporting {self.file_name}...")
        factor_data.to_parquet(get_root_dir() / f'factor_data/{self.file_name}.parquet.brotli', compression='brotli')
        elapsed_time = time.time() - start_time
        print(f"Time to create {self.file_name}: {round(elapsed_time)} seconds")
        print("-" * 60)

    @ray.remote
    def function(self, splice_data):
        return splice_data

    def create_factor(self):
        self.factor_data = set_timeframe(self.factor_data, self.start, self.end)
        if self.skip:
            print('Skipping splice and batch...')
            print(f"Exporting {self.file_name}...")
            if self.ticker != 'all':
                self.factor_data = self.factor_data.loc[self.ticker]
            self.factor_data.to_parquet(get_root_dir() / f'factor_data/{self.file_name}.parquet.brotli', compression='brotli')
            print("-" * 60)
        else:
            if not self.general and self.group != 'date':
                if self.ticker != 'all':
                    self.factor_data = self.factor_data.loc[self.ticker]
            splice_data = self._splice_data()
            batch_data = self._batch_data(splice_data)
            self.parallel_processing(batch_data)
