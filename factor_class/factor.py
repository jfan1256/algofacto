from typing import List

from functions.utils.func import *
from functions.utils.system import *
from itertools import chain

from typing import Union, Optional, List

import warnings

warnings.filterwarnings('ignore')


class Factor:
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
        self.factor_data = None
        self.live = live
        self.file_name = file_name
        self.skip = skip
        self.stock = stock
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.splice_size = splice_size
        self.group = group
        self.join = join
        self.general = general
        self.window = window

        assert group == 'permno' or group == 'ticker' or group == 'date' or group is None, ValueError('group parameter must be either ''permno'', ''ticker'', or ''date''')
        if self.group == 'date' and self.join == None:
            raise ValueError('if group parameter is set to date then specify join parameter to either ''permno'' or ''ticker''')

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

    # Convert general factor to multiindex stock/date pair
    def _create_multi_index(self, stock):
        factor_values = pd.concat([self.factor_data] * len(stock), ignore_index=True).values
        multi_index = pd.MultiIndex.from_product([stock, self.factor_data.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=self.factor_data.columns, index=multi_index)
        multi_index_factor.index = multi_index_factor.index.set_names([self.group, 'date'])
        return multi_index_factor

    # Splice data into dataframes with splice_size number of stocks or dates
    def _splice_data(self):
        data_spliced = {}
        splice = 1

        # If self.general, create multiindex
        if self.general:
            if 'ticker' in self.factor_data.index.names or 'permno' in self.factor_data.index.names:
                raise TypeError('if general parameter is set to True then there cannot be ''ticker'' or ''permno'' in the index')
            self.factor_data = self._create_multi_index(self.stock)

        # If self.group is ticker or permno
        if self.group == 'ticker' or self.group == 'permno':
            count = 0
            splice_all = []

            # Iterate through groups
            for _, df in self.factor_data.groupby(self.group, group_keys=False):
                # Append data to collection
                splice_all.append(df)
                count += 1
                # If number of dataframes in collection is equal to splice_size, then add to dictionary and reset
                if count == self.splice_size:
                    name = f'splice{splice}'
                    data_spliced[name] = pd.concat(splice_all, axis=0)
                    splice_all = []
                    splice += 1
                    count = 0
            # Append any remaining data
            if splice_all:
                name = f'splice{splice}'
                data_spliced[name] = pd.concat(splice_all, axis=0)
            return data_spliced

        elif self.group == 'date':
            if 'ticker' in self.factor_data.index.names:
                raise ValueError('if group parameter is set to ''date'' then ''ticker'' or ''permno'' cannot be in the index. Must only have ''date'' in index')

            # Get the rolling window groups, append to collection, and add to dictionary once size of collection is equal to splice_size
            window_data = self._rolling_window()
            for i in range(0, len(window_data), self.splice_size):
                name = f'splice{splice}'
                data_spliced[name] = window_data[i:i + self.splice_size]
                splice += 1
            return data_spliced

    # Create batches of spliced data with size batch_size
    def _batch_data(self, splice_data):
        batch = []
        factor_batch = {}
        batch_num = 1
        count = 1

        # Iterate through splice_data
        for i, item in enumerate(splice_data):
            # Append items to batch
            batch.append(splice_data[item])

            # Once size of batch is equal to batch_size, add to dictionary and reset
            if count == self.batch_size:
                name = f'batch{batch_num}'
                if self.group == 'ticker' or self.group == 'permno':
                    factor_batch[name] = batch
                elif self.group == 'date':
                    factor_batch[name] = list(chain.from_iterable(batch))
                batch_num = batch_num + 1
                count = 0
                batch = []
            count = count + 1

        # Append any remaining data
        name = f'batch{batch_num}'
        if self.group == 'ticker' or self.group == 'permno':
            factor_batch[name] = batch
        elif self.group == 'date':
            factor_batch[name] = list(chain.from_iterable(batch))
        return factor_batch

    # Feed in batch data and (function) will execute on all spliced data within the batch at the same time
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Executing creation', animation=spinner_animation)
    def _execute_creation(self, operation, batch_data):
        # Execute parallel processing
        results = ray.get([operation.remote(self, splice_data) for splice_data in batch_data])
        return results

    # Executes (function) across all batches and exports the created factor
    def _parallel_processing(self, batch_data):
        print(f"Creating {self.file_name}...")
        start_time = time.time()
        nested_data_all = []

        # Iterate through each batch and run the parallel processing function
        for i, item in enumerate(batch_data):
            i += 1
            data = (self._execute_creation(self.function, batch_data[item]))
            nested_data_all.append(data)
            print(f"Completed batch: {str(i)}")

        # Make the data iterable so that it can be exported (problem when self.group='date')
        flattened_data_all = list(chain.from_iterable(nested_data_all))
        factor_data = pd.concat(flattened_data_all, axis=0)

        # Set and sort index
        if self.group == 'date':
            factor_data = factor_data.reset_index([self.join, 'date'])
            factor_data = factor_data.set_index([self.join, 'date'])
            factor_data = factor_data.sort_index(level=[self.join, 'date'])
        else:
            factor_data = factor_data.reset_index([self.group, 'date'])
            factor_data = factor_data.set_index([self.group, 'date'])
            factor_data = factor_data.sort_index(level=[self.group, 'date'])

        # Export factor data
        print(f"Exporting {self.file_name}...")
        factor_data.to_parquet(get_factor_dir(self.live) / f'{self.file_name}.parquet.brotli', compression='brotli')
        elapsed_time = time.time() - start_time
        print(f"Time to create {self.file_name}: {round(elapsed_time)} seconds")
        print("-" * 60)

    # Inherited function that each factor class will edit and specify the required transformations to create the factor
    @ray.remote
    def function(self, splice_data):
        return splice_data

    # Create factor
    def create_factor(self):
        # Set timeframe
        self.factor_data = set_timeframe(self.factor_data, self.start, self.end)

        # Skip splicing and batching if self.skip is True
        if self.skip:
            print('Skipping splice and batch...')
            print(f"Exporting {self.file_name}...")

            # Get dataframe with list of stocks or not
            if self.stock != 'all':
                self.factor_data = get_stocks_data(self.factor_data, self.stock)
            self.factor_data.to_parquet(get_factor_dir(self.live) / f'{self.file_name}.parquet.brotli', compression='brotli')
            print("-" * 60)
        else:
            # Get dataframe with list of stocks or not
            if not self.general and self.stock != 'all':
                if self.group == 'date':
                    try:
                        # If multiindex column (unstacking multiple columns)
                        self.factor_data.loc[:, (slice(None), self.stock)]
                    except:
                        # If multiindex column (unstacking one column)
                        self.factor_data = self.factor_data[self.stock]
                else:
                    self.factor_data = get_stocks_data(self.factor_data, self.stock)

            # Create spliced data
            splice_data = self._splice_data()
            # Create batches
            batch_data = self._batch_data(splice_data)
            # Execute parallel processing factor creation
            self._parallel_processing(batch_data)