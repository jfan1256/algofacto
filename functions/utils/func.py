from functions.utils.system import *
from sklearn.cluster import KMeans
from statsmodels.regression.rolling import RollingOLS
from sklearn.decomposition import PCA
from timebudget import timebudget
from statsmodels.regression.linear_model import OLS


import pandas as pd
import matplotlib.pyplot as plt
import re
import csv
import pickle
import statsmodels.api as sm
import numpy as np
import ray
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')


# Returns a single stock dataframe
def get_stock_data(data, stock):
    idx = pd.IndexSlice
    desired_data = data.loc[idx[stock, :], :]
    return desired_data

# Returns a multiple stock dataframe
def get_stocks_data(data, stock):
    mask = data.index.get_level_values(data.index.names[0]).isin(stock)
    data = data.loc[mask]
    return data

# Remove dates that have less than 10 stocks (error in WRDS data?)
def remove_date(df, threshold):
    dates_with_counts = df.groupby(level='date').size()
    valid_dates = dates_with_counts[dates_with_counts >= threshold].index.tolist()
    return df[df.index.get_level_values('date').isin(valid_dates)]

# Returns a list of all stocks in a dataframe
def get_stock_idx(data):
    return [stock for stock, df in data.groupby(data.index.names[0], group_keys=False)]


# Splices a large data file once
def splice_data_once(data, splice_size):
    data_spliced = {}
    splice = 1
    count = 0
    splice_data = []
    for _, df in data.groupby(data.index.names[0], group_keys=False):
        splice_data.append(df)
        count += 1
        if count == splice_size:
            name = f'splice{splice}'
            data_spliced[name] = pd.concat(splice_data, axis=0)
            splice_data = []
            splice += 1
            count = 0
            break
    return data_spliced


# Gets the time frame length of a time-series dataframe
def get_timeframe_length(data):
    return len(sorted(data.index.get_level_values('date').unique()))


# Formats a time string 'HH:MM:SS' based on a numeric time() value
def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'


# Extracts the number from the end of a string
def extract_number(s):
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None


# Extracts the name from a .parquet.brotli file
def extract_first_string(s):
    return s.split('.')[0]


# Export stock list into .csv
def export_stock(data, file_name):
    my_list = get_stock_idx(data)
    my_list = [[item] for item in my_list]
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(my_list)


# Read stock list in
def read_stock(file_name):
    return pd.read_csv(file_name, header=None).iloc[:, 0].tolist()


# Find common stocks between stocks_to_train and data
def common_stocks(stocks, data):
    stocks = set(stocks)
    data_stocks = set(data.index.get_level_values(data.index.names[0]))  # Adjust the level if needed
    common = sorted(list(stocks & data_stocks))
    return common


# Get candidates and set keys to beginning of year
def get_candidate():
    with open(get_load_data_large_dir() / 'sp500_candidates.pkl', 'rb') as f:
        candidates = pickle.load(f)
    beginning_year = [date for date in candidates.keys() if date.month == 1]
    candidates = {date.year: candidates[date] for date in beginning_year if date in candidates}
    return candidates


# Set time frame of a data frame
def set_timeframe(data, start_date, end_date):
    data = data.loc[data.index.get_level_values('date') >= start_date]
    data = data.loc[data.index.get_level_values('date') <= end_date]
    return data


# Removes all data with less than year length data
def set_length(data, year):
    counts = data.groupby(data.index.names[0]).size()
    to_exclude = counts[counts < (year * 252)].index
    mask = ~data.index.get_level_values(data.index.names[0]).isin(to_exclude)
    data = data[mask]
    return data


# Rolling Kmeans cluster parallel processing
@timebudget
def rolling_kmean(data, window_size, n_clusters, name):
    # Make sure you copy the data before feeding it into rollingKmean or else it will take longer
    @ray.remote
    def exec_kmean(i, data, windowSize, n_clusters):
        # Get window data
        window_data = data.iloc[i:i + windowSize]
        window_data = window_data.drop(columns=window_data.columns[window_data.isna().sum() > len(window_data) / 2])
        window_data = window_data.fillna(0)

        # Run kmeans
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, n_init=10)
        result_clusters = kmeans.fit_predict(window_data.T)

        # Create a dataframe that matches loadings to stock
        df_cluster = pd.DataFrame(result_clusters, columns=[f'kCluster_{n_clusters}'], index=window_data.columns)
        return df_cluster

    # Parallel Processing
    ray.init(num_cpus=16, ignore_reinit_error=True)
    clusters_list = ray.get([exec_kmean.remote(i, data, window_size, n_clusters) for i in range(0, len(data) - window_size + 1)])
    ray.shutdown()

    # Concat all the window loadings
    results_clusters_combined = pd.concat(clusters_list, keys=data.index[window_size - 1:]).swaplevel()

    # Rearrange data to groupby stock
    results_clusters_combined.sort_index(level=[data.index.names[0], 'date'], inplace=True)
    results_clusters_combined.columns = [f'kMean{name}_{i}' for i in range(1, len(results_clusters_combined.columns) + 1)]

    return results_clusters_combined


# Rolling Kmeans cluster with pca through parallel processing
@timebudget
def rolling_kmean_pca(data, window_size, n_clusters, name):
    # Make sure you copy the data before feeding it into rollingKmean or else it will take longer
    @ray.remote
    def exec_kmean(i, data, windowSize, n_clusters):
        # Get window data
        window_pca = data.iloc[i:i + windowSize]
        window_pca = window_pca.fillna(0)

        inputs = []
        for value in window_pca.columns.unique(level=0):
            result = window_pca[value].to_numpy()
            pca = PCA(n_components=5)
            result = pca.fit_transform(result)
            inputs.append(result[:, 0])

        inputs = np.asarray(inputs)

        # Run kmeans
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, n_init=10)
        result_clusters = kmeans.fit_predict(inputs)

        # Create a dataframe that matches loadings to stock
        df_cluster = pd.DataFrame(result_clusters, columns=[f'kCluster_{n_clusters}'], index=window_pca.columns.get_level_values(0).unique())
        return df_cluster

    # Parallel Processing
    ray.init(num_cpus=16, ignore_reinit_error=True)
    clusters_list = ray.get([exec_kmean.remote(i, data, window_size, n_clusters) for i in range(0, len(data) - window_size + 1)])
    ray.shutdown()

    # Concat all the window loadings
    results_clusters_combined = pd.concat(clusters_list, keys=data.index[window_size - 1:]).swaplevel()

    # Rearrange data to groupby stock
    results_clusters_combined.sort_index(level=[data.index.names[0], 'date'], inplace=True)
    results_clusters_combined.columns = [f'kMean{name}_{i}' for i in range(1, len(results_clusters_combined.columns) + 1)]

    return results_clusters_combined


# Rolling LR to calculate beta coefficients + predictions + alpha
def rolling_ols_sb(price, factor_data, factor_col, window, name, ret):
    betas = []
    for stock, df in price.groupby(price.index.names[0], group_keys=False):
        model_data = df[[ret]].merge(factor_data, on='date').dropna()
        model_data[ret] -= model_data.RF
        rolling_ols = RollingOLS(endog=model_data[ret],
                                 exog=sm.add_constant(model_data[factor_col]), window=window)
        factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

        # Compute predictions of stock's return
        alpha = factor_model['ALPHA']
        beta_coef = factor_model[factor_col]
        factor_ret = model_data[factor_col]

        predictions = []
        for index, row in factor_ret.iterrows():
            predictions.append(row @ beta_coef.loc[index] + alpha.loc[index])

        result = factor_model.assign(**{price.index.names[0]: stock}).set_index(price.index.names[0], append=True).swaplevel()
        result['PRED'] = predictions
        betas.append(result)

    return pd.concat(betas).rename(columns=lambda x: f'{x}_{name}_{window:02}')

# Rolling LR to calculate beta coefficients + predictions + alpha
def rolling_ols_sb_test(price, factor_data, factor_col, window, name, ret):
    betas = []
    for stock, df in price.groupby(price.index.names[0], group_keys=False):
        model_data = df[[ret]].merge(factor_data, on='date').dropna()
        rolling_ols = RollingOLS(endog=model_data[ret],
                                 exog=sm.add_constant(model_data[factor_col]), window=window)
        factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

        # Compute predictions of stock's return
        alpha = factor_model['ALPHA']
        beta_coef = factor_model[factor_col]
        factor_ret = model_data[factor_col]

        predictions = []
        for index, row in factor_ret.iterrows():
            predictions.append(row @ beta_coef.loc[index] + alpha.loc[index])

        result = factor_model.assign(**{price.index.names[0]: stock}).set_index(price.index.names[0], append=True).swaplevel()
        result['PRED'] = predictions
        betas.append(result)

    return pd.concat(betas).rename(columns=lambda x: f'{x}_{name}_{window:02}')


# Rolling LR to calculate beta coefficients + predictions + alpha + epilson
def rolling_ols_residual(price, factor_data, factor_col, window, name, ret):
    betas = []
    for stock, df in price.groupby(price.index.names[0], group_keys=False):
        model_data = df[[ret]].merge(factor_data, on='date').dropna()
        model_data[ret] -= model_data.RF
        rolling_ols = RollingOLS(endog=model_data[ret],
                                 exog=sm.add_constant(model_data[factor_col]), window=window)
        factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

        # Compute predictions of stock's return
        alpha = factor_model['ALPHA']
        beta_coef = factor_model[factor_col]
        factor_ret = model_data[factor_col]
        stock_ret = df.reset_index(price.index.names[0]).drop(columns=price.index.names[0], axis=1)[ret]

        predictions = []
        epsilons = []
        for index, row in factor_ret.iterrows():
            prediction = row @ beta_coef.loc[index] + alpha.loc[index]
            epsilon = stock_ret.loc[index] - prediction
            predictions.append(prediction)
            epsilons.append(epsilon)

        result = factor_model.assign(**{price.index.names[0]: stock}).set_index(price.index.names[0], append=True).swaplevel()
        result['PRED'] = predictions
        result['EPSIL'] = epsilons
        result['EPSIL'] = result['EPSIL'].rolling(window=window).sum() / result['EPSIL'].rolling(window=window).std()
        betas.append(result)

    return pd.concat(betas).rename(columns=lambda x: f'{x}_{name}_{window:02}')

# Rolling LR to calculate beta coefficients + predictions + alpha + epilson + idiosyncratic risk
def rolling_ols_res_syn(price, factor_data, factor_col, window, name, ret):
    betas = []
    for stock, df in price.groupby(price.index.names[0], group_keys=False):
        model_data = df[[ret]].merge(factor_data, on='date').dropna()
        model_data[ret] -= model_data.RF
        rolling_ols = RollingOLS(endog=model_data[ret],
                                 exog=sm.add_constant(model_data[factor_col]), window=window)
        factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

        # Compute predictions of stock's return
        alpha = factor_model['ALPHA']
        beta_coef = factor_model[factor_col]
        factor_ret = model_data[factor_col]
        stock_ret = df.reset_index(price.index.names[0]).drop(columns=price.index.names[0], axis=1)[ret]

        predictions = []
        epsilons = []
        for index, row in factor_ret.iterrows():
            prediction = row @ beta_coef.loc[index] + alpha.loc[index]
            epsilon = stock_ret.loc[index] - prediction
            predictions.append(prediction)
            epsilons.append(epsilon)

        result = factor_model.assign(**{price.index.names[0]: stock}).set_index(price.index.names[0], append=True).swaplevel()
        result['PRED'] = predictions
        result['EPSIL'] = epsilons
        result['EPSIL_Z'] = result['EPSIL'].rolling(window=window).sum() / result['EPSIL'].rolling(window=window).std()
        result['IDIO_VOL'] = result['EPSIL'].rolling(window=window).std()
        betas.append(result)

    return pd.concat(betas).rename(columns=lambda x: f'{x}_{name}_{window:02}')

# Rolling PCA
def rolling_pca(data, window_size, num_components, name):
    principal_components = []
    for i in range(0, len(data) - window_size + 1):
        df = data.iloc[i:i + window_size]
        df = (df - df.mean()) / df.std()
        df.drop(columns=df.columns[df.isna().sum() > len(df) / 2], inplace=True)
        df.fillna(0, inplace=True)
        pca = PCA(n_components=num_components, random_state=42)
        results = pca.fit_transform(df)
        principal_components.append(results[-1])

    pca_df = pd.DataFrame(data=principal_components, index=data.index[window_size - 1:],
                        columns=[f'PCA_{name}_{i + 1}' for i in range(num_components)])
    pca_df.sort_index(level=['date'], inplace=True)
    return pca_df


# Rolling PCA Loadings
def rolling_pca_loading(data, window_size, num_components, name):
    loadings_list = []

    for i in range(0, len(data) - window_size + 1):
        # Get window data
        window_data = data.iloc[i:i + window_size]
        window_data.drop(columns=window_data.columns[window_data.isna().sum() > len(window_data) / 2], inplace=True)
        window_data.fillna(0, inplace=True)

        # Run pcaReturn and get loadings
        pca = PCA(n_components=num_components, random_state=42)
        pca.fit_transform(window_data)
        results_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Create a dataframe that matches loadings to stock
        df_loadings = pd.DataFrame(results_loadings, columns=[f'pcaLoading{name}_{i + 1}' for i in range(num_components)],
                                   index=window_data.columns)
        loadings_list.append(df_loadings)

    # Concat all the window loadings
    results_loadings_combined = pd.concat(loadings_list, keys=data.index[window_size - 1:]).swaplevel()
    results_loadings_combined.index.set_names([data.index.names[0], 'date'], inplace=True)
    # Rearrange data to groupby stock
    results_loadings_combined = pd.concat([df for data.index.names[0], df in results_loadings_combined.groupby(level=data.index.names[0])], axis=0)
    return results_loadings_combined


# Create historical returns
def create_return(df, windows):
    by_stock = df.groupby(level=df.index.names[0])
    for t in windows:
        df[f'RET_{t:02}'] = by_stock.Close.pct_change(t)
    return df


# Create percentage change for volume
def create_volume(df, windows):
    by_stock = df.groupby(level=df.index.names[0])
    for t in windows:
        df[f'VOL_{t:02}'] = by_stock.Volume.pct_change(t)
    return df

# Create percentage change for high
def create_high(df, windows):
    by_stock = df.groupby(level=df.index.names[0])
    for t in windows:
        df[f'HIGH_{t:02}'] = by_stock.High.pct_change(t)
    return df


# Create percentage change for low
def create_low(df, windows):
    by_stock = df.groupby(level=df.index.names[0])
    for t in windows:
        df[f'LOW_{t:02}'] = by_stock.Low.pct_change(t)
    return df

# Create percentage change for outstanding shares
def create_out(df, windows):
    by_stock = df.groupby(level=df.index.names[0])
    for t in windows:
        df[f'LOW_{t:02}'] = by_stock.Low.pct_change(t)
    return df

# Create smoothed historical returns
def create_smooth_return(df, windows, window_size):
    by_stock = df.groupby(level=df.index.names[0])
    for t in windows:
        df[f'RET_{t:02}'] = by_stock.Close.pct_change(t).rolling(window=window_size).mean()
    return df

# Plot Histogram
def plot_histogram(df):
    df.hist(bins='auto')
    plt.title('Distribution of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# For class functions that need to concat data with stocks to price data
def add_data_with_stock(data, factor_data):
    """collect = []
    for ticker, df in data.groupby(level='ticker'):
        factorData_df = get_ticker_data(factor_data, ticker)
        df = pd.concat([df, factorData_df], axis=1).drop(factorData_df.index.difference(df.index))
        collect.append(df)
    data = pd.concat(collect, axis=0)
    return data"""
    collect = []
    for stock, df in data.groupby(level=data.index.names[0]):
        factorData_df = get_stock_data(factor_data, stock)
        df = df.merge(factorData_df, left_index=True, right_index=True, how='left')
        df = df.loc[~df.index.duplicated(keep='first')]
        collect.append(df)

    data = pd.concat(collect, axis=0)
    return data


# For class functions that need to concat data with no stocks to price data
def add_data_without_stock(data, factor_data):
    collect = []
    for stock, df in data.groupby(data.index.names[0], group_keys=False):
        df = df.reset_index(data.index.names[0])
        merged = df.merge(factor_data, on='date', how='left').reset_index('date').set_index([data.index.names[0], 'date'])
        collect.append(merged)
        data = pd.concat(collect, axis=0)
    return data


# Gets SPY returns from Yahoo Finance
def get_spy(start_date, end_date):
    spy = yf.download(['SPY'], start=start_date, end=end_date)
    spy['spyRet'] = spy['Close'].pct_change()
    spy = spy.dropna()
    benchmark = spy[['spyRet']]
    benchmark.index = benchmark.index.tz_localize(None)
    return benchmark

