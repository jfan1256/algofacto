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
        def _SMA(factor_data):
            for t in self.mAT:
                factor_data[f'sma_{t}'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.SMA(x.Close, timeperiod=t)))
    
        # Exponential Moving Average
        def _EMA(factor_data):
            for t in self.mAT:
                factor_data[f'ema_{t}'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.EMA(x.Close, timeperiod=t)))
    
        # ----------------------------------------Price and Volatility Trends----------------------------------------------------
        # Hilbert Transform
        def _HT(factor_data):
            factor_data['ht'] = (factor_data.groupby(self.group, group_keys=False).Close.apply(talib.HT_TRENDLINE).div(factor_data.Close).sub(1))
    
        # -------------------------------------------Momentum Indicators----------------------------------------------------
        # Plus/Minus Directional Index
        def _PMDI(factor_data):
            factor_data['plus_di'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.PLUS_DI(x.High, x.Low, x.Close, timeperiod=14)))
            factor_data['minus_di'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.MINUS_DI(x.High, x.Low, x.Close, timeperiod=14)))
    
        # Average Directional Movement Index Rating
        def _ADXR(factor_data):
            factor_data['adxr'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.ADXR(x.High, x.Low, x.Close, timeperiod=14)))
    
        # Percentage Price Oscillator
        def _PPO(factor_data):
            factor_data['ppo'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.PPO(x.Close, fastperiod=12, slowperiod=26, matype=0)))
    
        # Aroon Oscillator
        def _AROONOSC(factor_data):
            factor_data['aroonosc'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.AROONOSC(high=x.High, low=x.Low, timeperiod=14)))
    
        # Balance of Power
        def _BOP(factor_data):
            factor_data['bop'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.BOP(x.Open, x.High, x.Low, x.Close)))
    
        # Commodity Channel Index
        def _CCI(factor_data):
            factor_data['cci'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.CCI(x.High, x.Low, x.Close, timeperiod=14)))
    
        # Moving Average Convergence/Divergence
        def _MACD(factor_data):
            def compute_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
                macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                return pd.DataFrame({'macd': macd, 'macd_signal': macdsignal, 'macd_hist': macdhist}, index=close.index)
    
            factor_data = (factor_data.join(factor_data.groupby(self.group, group_keys=False).Close.apply(compute_macd)))
    
        # Money Flow Index
        def _MFI(factor_data):
            factor_data['mfi'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.MFI(x.High, x.Low, x.Close, x.Volume, timeperiod=14)))
    
        # Relative Strength Index
        def _RSI(factor_data):
            factor_data['rsi'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.RSI(x.Close, timeperiod=14)))
    
        # Ultimate Oscillator
        def _ULTOSC(factor_data):
            factor_data['ultosc'] = (factor_data.groupby(self.group, group_keys=False).apply(
                lambda x: talib.ULTOSC(x.High, x.Low, x.Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)))
    
        # Williams Percent Range
        def _WILLR(factor_data):
            factor_data['willr'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.WILLR(x.High, x.Low, x.Close, timeperiod=14)))
    
        # -------------------------------------------Volume Indicators----------------------------------------------------
    
        # Chaikin A/D Line
        def _AD(factor_data):
            factor_data['ad'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.AD(x.High, x.Low, x.Close, x.Volume) / x.Volume.mean()))
    
        # On Balance Volume
        def _OBV(factor_data):
            factor_data['obv'] = (factor_data.groupby(self.group, group_keys=False).apply(lambda x: talib.OBV(x.Close, x.Volume) / x.expanding().Volume.mean()))
    
        _SMA(self.factor_data)
        # _EMA(factor_data)
        _HT(self.factor_data)
        # _PMDI(factor_data)
        _ADXR(self.factor_data)
        _PPO(self.factor_data)
        # _AROONOSC(factor_data)
        _BOP(self.factor_data)
        _CCI(self.factor_data)
        _MACD(self.factor_data)
        # _MFI(factor_data)
        _RSI(self.factor_data)
        _ULTOSC(self.factor_data)
        _WILLR(self.factor_data)
        _AD(self.factor_data)
        _OBV(self.factor_data)