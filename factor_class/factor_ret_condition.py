from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorRetCondition(Factor):
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
                 window: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')

    @ray.remote
    def function(self, splice_data):
        T = [5, 10, 40, 60]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        # # Mean Reversion
        # splice_data['mean_reversion'] = np.where((splice_data['RET_05'] < 0) & (splice_data['RET_60'] > 0), 1, 0)

        # # Volatility Clustering
        # splice_data['high_volatility'] = np.where(splice_data['RET_05'].rolling(window=10).std() > splice_data['RET_05'].rolling(window=60).std(), 1, 0)

        # # Price above Moving Average
        # splice_data['MA_20'] = splice_data['RET_05'].rolling(window=20).mean()
        # splice_data['price_above_ma'] = np.where(splice_data['RET_05'] > splice_data['MA_20'], 1, 0)

        # # Relative Strength Index (RSI)
        # delta = splice_data['RET_05'].diff()
        # gain = (delta.where(delta > 0, 0)).fillna(0)
        # loss = (-delta.where(delta < 0, 0)).fillna(0)
        # avg_gain = gain.rolling(window=14).mean()
        # avg_loss = loss.rolling(window=14).mean()
        # rs = avg_gain / avg_loss
        # splice_data['RSI'] = 100 - (100 / (1 + rs))
        # splice_data['rsi_oversold'] = np.where(splice_data['RSI'] < 30, 1, 0)

        # Consecutive Gains
        splice_data['consecutive_gains'] = np.where((splice_data['RET_05'] > 0) & (splice_data['RET_10'] > 0) & (splice_data['RET_40'] > 0), 1, 0)

        for t in T:
            splice_data = splice_data.drop([f'RET_{t:02}'], axis=1)

        return splice_data
