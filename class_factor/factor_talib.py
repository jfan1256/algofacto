import talib

from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor


class FactorTalib(Factor):
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
        self.mAT = [5, 21, 63]

    # -----------------------------------------Moving Averages----------------------------------------------------
    # Simple Moving Average
    def _SMA(self, splice_data):
        for t in self.mAT:
            splice_data[f'sma_{t}'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.SMA(x.Close, timeperiod=t)))

    # Exponential Moving Average
    def _EMA(self, splice_data):
        for t in self.mAT:
            splice_data[f'ema_{t}'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.EMA(x.Close, timeperiod=t)))

    # ----------------------------------------Price and Volatility Trends----------------------------------------------------
    # Hilbert Transform
    def _HT(self, splice_data):
        splice_data['ht'] = (splice_data.groupby(self.group, group_keys=False).Close.apply(talib.HT_TRENDLINE).div(splice_data.Close).sub(1))

    # -------------------------------------------Momentum Indicators----------------------------------------------------
    # Plus/Minus Directional Index
    def _PMDI(self, splice_data):
        splice_data['plus_di'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.PLUS_DI(x.High, x.Low, x.Close, timeperiod=14)))
        splice_data['minus_di'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.MINUS_DI(x.High, x.Low, x.Close, timeperiod=14)))

    # Average Directional Movement Index Rating
    def _ADXR(self, splice_data):
        splice_data['adxr'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.ADXR(x.High, x.Low, x.Close, timeperiod=14)))

    # Percentage Price Oscillator
    def _PPO(self, splice_data):
        splice_data['ppo'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.PPO(x.Close, fastperiod=12, slowperiod=26, matype=0)))

    # Aroon Oscillator
    def _AROONOSC(self, splice_data):
        splice_data['aroonosc'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.AROONOSC(high=x.High, low=x.Low, timeperiod=14)))

    # Balance of Power
    def _BOP(self, splice_data):
        splice_data['bop'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.BOP(x.Open, x.High, x.Low, x.Close)))

    # Commodity Channel Index
    def _CCI(self, splice_data):
        splice_data['cci'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.CCI(x.High, x.Low, x.Close, timeperiod=14)))

    # Moving Average Convergence/Divergence
    def _MACD(self, splice_data):
        def compute_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
            return pd.DataFrame({'macd': macd, 'macd_signal': macdsignal, 'macd_hist': macdhist}, index=close.index)

        splice_data = (splice_data.join(splice_data.groupby(self.group, group_keys=False).Close.apply(compute_macd)))

    # Money Flow Index
    def _MFI(self, splice_data):
        splice_data['mfi'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.MFI(x.High, x.Low, x.Close, x.Volume, timeperiod=14)))

    # Relative Strength Index
    def _RSI(self, splice_data):
        splice_data['rsi'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.RSI(x.Close, timeperiod=14)))

    # Ultimate Oscillator
    def _ULTOSC(self, splice_data):
        splice_data['ultosc'] = (splice_data.groupby(self.group, group_keys=False).apply(
            lambda x: talib.ULTOSC(x.High, x.Low, x.Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)))

    # Williams Percent Range
    def _WILLR(self, splice_data):
        splice_data['willr'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.WILLR(x.High, x.Low, x.Close, timeperiod=14)))

    # -------------------------------------------Volume Indicators----------------------------------------------------

    # Chaikin A/D Line
    def _AD(self, splice_data):
        splice_data['ad'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.AD(x.High, x.Low, x.Close, x.Volume) / x.Volume.mean()))

    # On Balance Volume
    def _OBV(self, splice_data):
        splice_data['obv'] = (splice_data.groupby(self.group, group_keys=False).apply(lambda x: talib.OBV(x.Close, x.Volume) / x.expanding().Volume.mean()))

    @ray.remote
    def function(self, splice_data):
        self._SMA(splice_data)
        # self._EMA(splice_data)
        self._HT(splice_data)
        # self._PMDI(splice_data)
        self._ADXR(splice_data)
        self._PPO(splice_data)
        # self._AROONOSC(splice_data)
        self._BOP(splice_data)
        self._CCI(splice_data)
        self._MACD(splice_data)
        # self._MFI(splice_data)
        self._RSI(splice_data)
        self._ULTOSC(splice_data)
        self._WILLR(splice_data)
        self._AD(splice_data)
        self._OBV(splice_data)
        return splice_data
