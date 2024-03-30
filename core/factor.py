import talib

from core.operation import *

# Handle Data
def handle_data(data):
    factor_data = data.copy(deep=True)
    factor_data = factor_data.drop(['ticker', 'Open', 'Close', 'Low', 'High', 'Volume'], axis=1, errors='ignore')
    factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
    factor_data = factor_data.loc[~factor_data.index.duplicated(keep='first')]
    return factor_data

# Factor Ret
def factor_ret(data, window):
    factor_data = data.copy(deep=True)
    factor_data = create_return(factor_data, window)
    return handle_data(factor_data)

# Factor Ret Comp
def factor_ret_comp(data, window):
    factor_data = data.copy(deep=True)
    factor_data = create_return(factor_data, [1])
    for t in window:
        factor_data[f'ret_comp_{t:01}'] = factor_data.groupby('permno')['RET_01'].rolling(window=t).apply(lambda x: (1 + x).prod() - 1, raw=True).reset_index(level=0, drop=True)

    factor_data = factor_data.drop(f'RET_01', axis=1)
    return handle_data(factor_data)

# Factor Cycle
def factor_cycle(data):
    factor_data = data.copy(deep=True)
    factor_data['month'] = factor_data.index.get_level_values('date').month
    factor_data['weekday'] = factor_data.index.get_level_values('date').weekday
    factor_data['is_halloween'] = factor_data.index.get_level_values('date').map(lambda x: 1 if 5 <= x.month <= 10 else 0)
    factor_data['is_january'] = (factor_data.index.get_level_values('date').month == 1).astype(int)
    factor_data['is_friday'] = (factor_data.index.get_level_values('date').dayofweek == 4).astype(int)
    factor_data['last_day'] = factor_data.index.get_level_values('date').to_period('M').to_timestamp(how='end')
    factor_data['begin_last_week'] = factor_data['last_day'] - pd.Timedelta(days=5)
    factor_data['in_last_week'] = (factor_data.index.get_level_values('date') >= factor_data['begin_last_week']) & (factor_data.index.get_level_values('date') <= factor_data['last_day'])
    factor_data['is_quarter_end_week'] = (factor_data['in_last_week'] & factor_data.index.get_level_values('date').month.isin([3, 6, 9, 12])).astype(int)
    factor_data['is_year_end_week'] = (factor_data['in_last_week'] & (factor_data.index.get_level_values('date').month == 12)).astype(int)
    factor_data = factor_data.drop(columns=['last_day', 'begin_last_week', 'in_last_week'], axis=1)
    return handle_data(factor_data)

# Factor TALIB
def factor_talib_window(data):
    factor_data = data.copy(deep=True)
    mAT = [5, 21, 63]

    # Simple Moving Average
    def _SMA(factor_data):
        for t in mAT:
            factor_data[f'sma_{t}'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.SMA(x.Close, timeperiod=t)))

    # Exponential Moving Average
    def _EMA(factor_data):
        for t in mAT:
            factor_data[f'ema_{t}'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.EMA(x.Close, timeperiod=t)))

    # Hilbert Transform
    def _HT(factor_data):
        factor_data['ht'] = (factor_data.groupby('permno', group_keys=False).Close.apply(talib.HT_TRENDLINE).div(factor_data.Close).sub(1))

    # Plus/Minus Directional Index
    def _PMDI(factor_data):
        factor_data['plus_di'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.PLUS_DI(x.High, x.Low, x.Close, timeperiod=14)))
        factor_data['minus_di'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.MINUS_DI(x.High, x.Low, x.Close, timeperiod=14)))

    # Average Directional Movement Index Rating
    def _ADXR(factor_data):
        factor_data['adxr'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.ADXR(x.High, x.Low, x.Close, timeperiod=14)))

    # Percentage Price Oscillator
    def _PPO(factor_data):
        factor_data['ppo'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.PPO(x.Close, fastperiod=12, slowperiod=26, matype=0)))

    # Aroon Oscillator
    def _AROONOSC(factor_data):
        factor_data['aroonosc'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.AROONOSC(high=x.High, low=x.Low, timeperiod=14)))

    # Balance of Power
    def _BOP(factor_data):
        factor_data['bop'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.BOP(x.Open, x.High, x.Low, x.Close)))

    # Commodity Channel Index
    def _CCI(factor_data):
        factor_data['cci'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.CCI(x.High, x.Low, x.Close, timeperiod=14)))

    # Moving Average Convergence/Divergence
    def _MACD(factor_data):
        def compute_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
            return pd.DataFrame({'macd': macd, 'macd_signal': macdsignal, 'macd_hist': macdhist}, index=close.index)

        factor_data = (factor_data.join(factor_data.groupby('permno', group_keys=False).Close.apply(compute_macd)))

    # Money Flow Index
    def _MFI(factor_data):
        factor_data['mfi'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.MFI(x.High, x.Low, x.Close, x.Volume, timeperiod=14)))

    # Relative Strength Index
    def _RSI(factor_data):
        factor_data['rsi'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.RSI(x.Close, timeperiod=14)))

    # Ultimate Oscillator
    def _ULTOSC(factor_data):
        factor_data['ultosc'] = (factor_data.groupby('permno', group_keys=False).apply(
            lambda x: talib.ULTOSC(x.High, x.Low, x.Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)))

    # Williams Percent Range
    def _WILLR(factor_data):
        factor_data['willr'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.WILLR(x.High, x.Low, x.Close, timeperiod=14)))

    _SMA(factor_data)
    _HT(factor_data)
    _ADXR(factor_data)
    _PPO(factor_data)
    _BOP(factor_data)
    _CCI(factor_data)
    _MACD(factor_data)
    _RSI(factor_data)
    _ULTOSC(factor_data)
    _WILLR(factor_data)
    _EMA(factor_data)
    _PMDI(factor_data)
    _AROONOSC(factor_data)
    _MFI(factor_data)
    return handle_data(factor_data)


# Factor TALIB
def factor_talib_expand(data, current_date):
    factor_data = data.copy(deep=True)

    # Chaikin A/D Line
    def _AD(factor_data):
        factor_data['ad'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.AD(x.High, x.Low, x.Close, x.Volume) / x.expanding().Volume.mean()))

    # On Balance Volume
    def _OBV(factor_data):
        factor_data['obv'] = (factor_data.groupby('permno', group_keys=False).apply(lambda x: talib.OBV(x.Close, x.Volume) / x.expanding().Volume.mean()))

    _OBV(factor_data)
    _AD(factor_data)
    factor_data = window_data(factor_data, current_date, 63*2)
    return handle_data(factor_data)

# Factor Volume
def factor_volume(data, window):
    factor_data = data.copy(deep=True)
    factor_data = create_volume(factor_data, window)
    return handle_data(factor_data)

# Factor Load Ret
def factor_load_ret(data, num_component, window):
    factor_data = data.copy(deep=True)
    component = num_component
    
    # Create returns and convert stock index to columns
    factor_data = create_return(factor_data, windows=[1])
    factor_data = factor_data[[f'RET_01']]
    factor_data = factor_data.dropna()
    factor_data = factor_data['RET_01'].unstack('permno')

    # Remove last row of nan if there (i.e., 14937 has no data on Friday 2022-11-25 - causing a row of nan at the end except for permno 14937 when stacking)
    if factor_data.tail(1).isna().sum(axis=1).values[0] > (factor_data.shape[1] / 2):
        factor_data = factor_data.iloc[:-1]

    # Get last window
    factor_data = factor_data.tail(window)
    
    # Normalize data
    factor_data = (factor_data - factor_data.mean()) / factor_data.std()

    # Drop columns that have more than half of missing data
    factor_data = factor_data.drop(columns=factor_data.columns[factor_data.isna().sum() > len(factor_data) / 2])
    factor_data = factor_data.fillna(0)

    # Get loadings
    pca = PCA(n_components=component, random_state=42)
    pca.fit_transform(factor_data)
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create a dataframe that matches loadings to stock
    cols = factor_data.columns
    date = factor_data.index[-1]
    factor_data = pd.DataFrame(loading, columns=[f'load_ret_{i + 1}' for i in range(component)], index=[[date] * len(cols), cols])
    factor_data.index.names = ['date', 'permno']
    factor_data = factor_data.swaplevel()
    return handle_data(factor_data)

# Factor Ind
def factor_ind(data, live):
    factor_data = data.copy(deep=True)
    ind = pd.read_parquet(get_parquet(live) / 'data_ind.parquet.brotli')
    factor_data = pd.merge(factor_data, ind, left_index=True, right_index=True, how='left')
    factor_data = factor_data[['Industry']]
    factor_data = factor_data.groupby('permno').ffill()
    return handle_data(factor_data)

# Factor Ind Mom
def factor_ind_mom(data, live, window):
    factor_data = data.copy(deep=True)
    ind = pd.read_parquet(get_parquet(live) / 'data_ind.parquet.brotli')
    factor_data = pd.merge(factor_data, ind, left_index=True, right_index=True, how='left')
    factor_data['Industry'] = factor_data.groupby('permno')['Industry'].ffill()

    ret = create_return(factor_data, windows=window)
    collect = []

    for t in window:
        ret[f'ind_mom_{t:02}'] = ret.groupby(['Industry', 'date'])[f'RET_{t:02}'].transform('mean')
        ind_mom = ret[[f'ind_mom_{t:02}']]
        collect.append(ind_mom)

    factor_data = pd.concat(collect, axis=1)
    return handle_data(factor_data)

# Factor Clust Ret
def factor_clust_ret(data, cluster, window):
    factor_data = data.copy(deep=True)

    # Create returns and convert stock index to columns
    factor_data = create_return(factor_data, windows=[1])
    factor_data = factor_data[[f'RET_01']]
    factor_data = factor_data.dropna()
    factor_data = factor_data['RET_01'].unstack('permno')

    # Remove last row of nan if there (i.e., 14937 has no data on Friday 2022-11-25 - causing a row of nan at the end except for permno 14937 when stacking)
    if factor_data.tail(1).isna().sum(axis=1).values[0] > (factor_data.shape[1] / 2):
        factor_data = factor_data.iloc[:-1]

    # Get last window
    factor_data = factor_data.tail(window)

    # Normalize data
    factor_data = (factor_data - factor_data.mean()) / factor_data.std()

    # Drop columns that have more than half of missing data
    factor_data = factor_data.drop(columns=factor_data.columns[factor_data.isna().sum() > len(factor_data) / 2])
    factor_data = factor_data.fillna(0)

    # Run kmeans
    kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state=0, n_init=10)
    cluster_fit = kmeans.fit_predict(factor_data.T)

    # Create a dataframe that matches cluster to stock
    cols = factor_data.columns
    date = factor_data.index[-1]
    factor_data = pd.DataFrame(cluster_fit, columns=[f'ret_01_cluster'], index=[[date] * len(cols), cols])
    factor_data.index.names = ['date', 'permno']
    factor_data = factor_data.swaplevel()
    return handle_data(factor_data)