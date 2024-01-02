from core.system import *

from sklearn.cluster import KMeans
from statsmodels.regression.rolling import RollingOLS
from sklearn.decomposition import PCA
from timebudget import timebudget
from joblib import Parallel, delayed
from tqdm import tqdm
from plotly.subplots import make_subplots
from datetime import date
from pandas.tseries.offsets import BDay

import time
import plotly.offline as py
import plotly.graph_objs as go
import json
import pandas as pd
import requests
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

with open(get_config() / 'api_key.json') as f:
  config = json.load(f)
  api_key = config['fmp_key']

# Gets dataframe for a specified stock
def get_stock_data(data, stock):
    idx = pd.IndexSlice
    desired_data = data.loc[idx[stock, :], :]
    return desired_data

# Gets dataframe with a specified list of stocks
def get_stocks_data(data, stock):
    mask = data.index.get_level_values(data.index.names[0]).isin(stock)
    data = data.loc[mask]
    return data

# Remove dates that have less than threshold stocks (error in WRDS data?)
def remove_date(df, threshold):
    dates_with_counts = df.groupby(level='date').size()
    valid_dates = dates_with_counts[dates_with_counts >= threshold].index.tolist()
    return df[df.index.get_level_values('date').isin(valid_dates)]

# Foward fill for a certain amount of days
def ffill_max_days(df, max_days):
    # Create a new dataframe to store the filled values
    df_filled = df.copy()
    # Loop through all columns in the dataframe
    for col in df.columns:
        last_valid_idx = None
        for idx in df_filled.index:
            # If the current row contains a NaN value
            if pd.isna(df_filled.at[idx, col]):
                # If we've seen a valid index before and the gap is within max_days
                if (last_valid_idx is not None and
                        (idx - last_valid_idx).days <= max_days):
                    df_filled.at[idx, col] = df_filled.at[last_valid_idx, col]
            else:
                last_valid_idx = idx
    return df_filled

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


# Extracts the name from a file ending with ".parquet.brotli"
def extract_first_string(s):
    return s.split('.')[0]


# Export list of stocks into a csv file
def export_stock(data, file_name):
    my_list = get_stock_idx(data)
    my_list = [[item] for item in my_list]
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(my_list)


# Read list of stocks in
def read_stock(file_name):
    return pd.read_csv(file_name, header=None).iloc[:, 0].tolist()


# Find common stocks between stocks_to_train and data
def common_stocks(stocks, data):
    stocks = set(stocks)
    data_stocks = set(data.index.get_level_values(data.index.names[0]))  # Adjust the level if needed
    common = sorted(list(stocks & data_stocks))
    return common


# Get SP500 candidates and set keys to the given year
def get_candidate(live):
    with open(get_large(live) / 'sp500_candidates.pkl', 'rb') as f:
        candidates = pickle.load(f)
    beginning_year = [date for date in candidates.keys() if date.month == 1]
    candidates = {date.year: candidates[date] for date in beginning_year if date in candidates}
    return candidates


# Set timeframe
def set_timeframe(data, start_date, end_date):
    data = data.loc[data.index.get_level_values('date') >= start_date]
    data = data.loc[data.index.get_level_values('date') <= end_date]
    return data


# Removes all data with that have less than year(param) worth of data
def set_length(data, year):
    counts = data.groupby(data.index.names[0]).size()
    to_exclude = counts[counts < (year * 252)].index
    mask = ~data.index.get_level_values(data.index.names[0]).isin(to_exclude)
    data = data[mask]
    return data


# Rolling Kmean
@timebudget
def rolling_kmean(data, window_size, n_clusters, name):
    # Make sure you copy the data before feeding it into rollingKmean or else it will take longer
    @ray.remote
    def exec_kmean(i, data, windowSize, n_clusters):
        # Get window data
        window_data = data.iloc[i:i + windowSize]
        window_data = window_data.drop(columns=window_data.columns[window_data.isna().sum() > len(window_data) / 2])
        window_data = window_data.fillna(0)

        # Run KMean
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, n_init=10)
        result_clusters = kmeans.fit_predict(window_data.T)

        # Create a dataframe that matches loadings to stock
        df_cluster = pd.DataFrame(result_clusters, columns=[f'kCluster_{n_clusters}'], index=window_data.columns)
        return df_cluster

    # Execute parallel processing
    ray.init(num_cpus=16, ignore_reinit_error=True)
    clusters_list = ray.get([exec_kmean.remote(i, data, window_size, n_clusters) for i in range(0, len(data) - window_size + 1)])
    ray.shutdown()
    # Concat all the window loadings
    results_clusters_combined = pd.concat(clusters_list, keys=data.index[window_size - 1:]).swaplevel()
    # Rearrange data to groupby stock
    results_clusters_combined.sort_index(level=[data.index.names[0], 'date'], inplace=True)
    results_clusters_combined.columns = [f'kMean{name}_{i}' for i in range(1, len(results_clusters_combined.columns) + 1)]
    return results_clusters_combined


# Rolling PCA KMean
@timebudget
def rolling_kmean_pca(data, window_size, n_clusters, name):
    # Make sure you copy the data before feeding it into here or else it will take longer
    @ray.remote
    def exec_kmean(i, data, windowSize, n_clusters):
        # Get window data
        window_pca = data.iloc[i:i + windowSize]
        window_pca = window_pca.fillna(0)

        # Execute PCA
        inputs = []
        for value in window_pca.columns.unique(level=0):
            result = window_pca[value].to_numpy()
            pca = PCA(n_components=5)
            result = pca.fit_transform(result)
            inputs.append(result[:, 0])

        inputs = np.asarray(inputs)

        # Run KMean
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, n_init=10)
        result_clusters = kmeans.fit_predict(inputs)

        # Create a dataframe that matches loadings to stock
        df_cluster = pd.DataFrame(result_clusters, columns=[f'kCluster_{n_clusters}'], index=window_pca.columns.get_level_values(0).unique())
        return df_cluster

    # Execute parallel processing
    ray.init(num_cpus=16, ignore_reinit_error=True)
    clusters_list = ray.get([exec_kmean.remote(i, data, window_size, n_clusters) for i in range(0, len(data) - window_size + 1)])
    ray.shutdown()
    # Concat all the window loadings
    results_clusters_combined = pd.concat(clusters_list, keys=data.index[window_size - 1:]).swaplevel()
    # Rearrange data to groupby stock
    results_clusters_combined.sort_index(level=[data.index.names[0], 'date'], inplace=True)
    results_clusters_combined.columns = [f'kMean{name}_{i}' for i in range(1, len(results_clusters_combined.columns) + 1)]
    return results_clusters_combined

# Rolling Linear Regression
def rolling_ols_beta_res_syn(price, factor_data, factor_col, window, name, ret):
    betas = []
    # Iterate through each stock
    for stock, df in price.groupby(price.index.names[0], group_keys=False):
        model_data = df[[ret]].merge(factor_data, on='date').dropna()

        # Subtract returns by risk free weight to get risk premium
        # model_data[ret] -= model_data.RF

        rolling_ols = RollingOLS(endog=model_data[ret],
                                 exog=sm.add_constant(model_data[factor_col]), window=window)
        factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

        # Retrieve alpha, beta, and actual returns
        alpha = factor_model['ALPHA']
        beta_coef = factor_model[factor_col]
        factor_ret = model_data[factor_col]
        stock_ret = df.reset_index(price.index.names[0]).drop(columns=price.index.names[0], axis=1)[ret]

        # Calculate predictions and epsilon
        predictions = []
        epsilons = []
        for index, row in factor_ret.iterrows():
            prediction = row @ beta_coef.loc[index] + alpha.loc[index]
            epsilon = stock_ret.loc[index] - prediction
            predictions.append(prediction)
            epsilons.append(epsilon)

        # Create dataframe and calculate residual momentum and idiosyncratic volatility
        result = factor_model.assign(**{price.index.names[0]: stock}).set_index(price.index.names[0], append=True).swaplevel()
        result['pred'] = predictions
        result['epsil'] = epsilons
        result['resid_mom_21'] = result['epsil'].rolling(window=21).sum() / result['epsil'].rolling(window=21).std()
        result['resid_mom_126'] = result['epsil'].rolling(window=126).sum() / result['epsil'].rolling(window=126).std()
        result['idio_vol_21'] = result['epsil'].rolling(window=21).std()
        result['idio_vol_126'] = result['epsil'].rolling(window=126).std()
        betas.append(result)

    return pd.concat(betas).rename(columns=lambda x: f'{x}_{name}_{window:02}')


# Rolling Linear Regression (Parallelized)
def rolling_ols_parallel(data, ret, factor_data, factor_cols, window, name):
    def process_stock(stock_data, ret, factor_data, factor_cols, window, stock_name, index_name):
        model_data = stock_data[[ret]].merge(factor_data, on='date').dropna()
        model_data = model_data[[ret] + factor_cols]

        exog = sm.add_constant(model_data[factor_cols])

        rolling_ols = RollingOLS(endog=model_data[ret], exog=exog, window=window)
        factor_model_params = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

        # Calculate predicted values
        predicted = (exog * factor_model_params).sum(axis=1)
        predicted = predicted.rename('pred')

        # Compute residuals (epsilon)
        epsilon = model_data[ret] - predicted
        epsilon = epsilon.rename('epsil')

        result = factor_model_params.assign(epsil=epsilon, pred=predicted)
        result = result.assign(**{index_name: stock_name}).set_index(index_name, append=True).swaplevel()

        # Additional calculations
        result['resid_mom_21'] = result['epsil'].rolling(window=21).sum() / result['epsil'].rolling(window=21).std()
        result['resid_mom_126'] = result['epsil'].rolling(window=126).sum() / result['epsil'].rolling(window=126).std()
        result['idio_vol_21'] = result['epsil'].rolling(window=21).std()
        result['idio_vol_126'] = result['epsil'].rolling(window=126).std()

        return result

    tasks = [(group, ret, factor_data, factor_cols, window, stock, data.index.names[0]) for stock, group in data.groupby(data.index.names[0])]
    results = Parallel(n_jobs=-1)(delayed(process_stock)(*task) for task in tasks)

    return pd.concat(results).rename(columns=lambda x: f'{x}_{name}_{window:02}')

# Rolling PCA
def rolling_pca(data, window_size, num_components, name):
    principal_components = []

    # Iterate through windows
    for i in range(0, len(data) - window_size + 1):
        # Get window data
        df = data.iloc[i:i + window_size]
        # Standardize the data
        df = (df - df.mean()) / df.std()
        # Drop columns with if they have NAN values for more than half the column
        df = df.drop(columns=df.columns[df.isna().sum() > len(df) / 2])
        df = df.fillna(0)
        # Execute PCA
        pca = PCA(n_components=num_components, random_state=42)
        results = pca.fit_transform(df)
        principal_components.append(results[-1])

    # Create dataframe of general PCA components
    pca_df = pd.DataFrame(data=principal_components, index=data.index[window_size - 1:], columns=[f'PCA_{name}_{i + 1}' for i in range(num_components)])
    pca_df = pca_df.sort_index(level=['date'])
    return pca_df


# Rolling PCA Loadings
def rolling_pca_loading(data, window_size, num_components, name):
    loadings_list = []

    # Iterative through windows
    for i in range(0, len(data) - window_size + 1):
        # Get window data
        window_data = data.iloc[i:i + window_size]
        window_data = window_data.drop(columns=window_data.columns[window_data.isna().sum() > len(window_data) / 2])
        window_data = window_data.fillna(0)

        # Run PCA and get loadings
        pca = PCA(n_components=num_components, random_state=42)
        pca.fit_transform(window_data)
        results_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Create a dataframe that matches loadings to its respective stock
        df_loadings = pd.DataFrame(results_loadings, columns=[f'pcaLoading{name}_{i + 1}' for i in range(num_components)],
                                   index=window_data.columns)
        loadings_list.append(df_loadings)

    # Concat all the window loadings
    results_loadings_combined = pd.concat(loadings_list, keys=data.index[window_size - 1:]).swaplevel()
    results_loadings_combined.index.set_names([data.index.names[0], 'date'], inplace=True)
    # Rearrange data to groupby stock
    results_loadings_combined = pd.concat([df for data.index.names[0], df in results_loadings_combined.groupby(level=data.index.names[0])], axis=0)
    return results_loadings_combined


# Create returns
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
        df[f'OUTSTANDING_{t:02}'] = by_stock.Low.pct_change(t)
    return df

# Create smoothed returns
def create_smooth_return(df, windows, window_size):
    by_stock = df.groupby(level=df.index.names[0])
    for t in windows:
        df[f'RET_{t:02}'] = by_stock.Close.pct_change(t).rolling(window=window_size).mean()
    return df

# Plot histogram
def plot_histogram(df):
    df.hist(bins='auto')
    plt.title('Distribution of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# Gets SPY returns from yfinance
def get_spy(start_date, end_date):
    spy = yf.download(['SPY'], start=start_date, end=end_date)
    spy['spyRet'] = spy['Close'].pct_change()
    spy = spy.dropna()
    benchmark = spy[['spyRet']]
    benchmark.index = benchmark.index.tz_localize(None)
    return benchmark

# Remove the last date of NAN in permno groups that are delisted prior to end date
def remove_nan_before_end(data, column):
    mask = data.index.get_level_values('date') != data.index.get_level_values('date').unique().max()
    data = data[~(data[column].isna() & mask)]
    return data

# Get stock price data from FMP from start to current_date
def get_data_fmp(ticker_list, start, current_date):
    frames = []

    for ticker in tqdm(ticker_list, desc="Fetching data", unit="ticker"):
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start}&to={current_date}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()

        # Check if there's historical data in the response
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df['ticker'] = ticker
            frames.append(df)
        else:
            print(f"Skipped {ticker} - No data available")

    if frames:  # Only proceed if there are frames to concatenate
        price = pd.concat(frames)
        price['date'] = pd.to_datetime(price['date'], errors='coerce')
        price = price.set_index(['ticker', 'date'])
        cols_to_convert = ['open', 'high', 'low', 'close', 'adjClose', 'volume']
        price[cols_to_convert] = price[cols_to_convert].astype(float)
        price = price.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'adjClose': 'Adj Close',
            'close': 'Close',
            'volume': 'Volume'
        })
        return price.sort_index(level=['ticker', 'date'])
    else:
        print("No data available for any ticker.")
        return None

# Get all stock news per ticker from fmp
def get_news_fmp(tickers):
    base_url = "https://financialmodelingprep.com/api/v3/stock_news"
    limit = 1000
    all_data = []
    for ticker in tqdm(tickers):
        page = 0
        while True:
            url = f"{base_url}?tickers={ticker}&page={page}&limit={limit}&apikey={api_key}"
            response = requests.get(url)
            data = response.json()
            if not data or 'error' in data:
                break
            all_data.extend(data)
            page += 1

    df = pd.DataFrame(all_data).rename(columns={'symbol': 'ticker', 'publishedDate': 'date'}).drop('image', axis=1)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['ticker', 'date']).sort_index(level=['ticker', 'date'])
    return df

# Get adjustment factor for close price for date from FMP
def get_adj_factor_fmp(ticker_list, date):
    frames = []

    for ticker in tqdm(ticker_list, desc="Fetching data", unit="ticker"):
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={date}&to={date}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()

        # Check if there's historical data in the response
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df['ticker'] = ticker
            frames.append(df)
        else:
            print(f"Skipped {ticker} - No data available")

    if frames:  # Only proceed if there are frames to concatenate
        price = pd.concat(frames)
        price['date'] = pd.to_datetime(price['date'], errors='coerce')
        price = price.set_index(['ticker', 'date'])
        cols_to_convert = ['open', 'high', 'low', 'close', 'adjClose', 'volume']
        price[cols_to_convert] = price[cols_to_convert].astype(float)
        price = price.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'adjClose': 'Adj Close',
            'close': 'Close',
            'volume': 'Volume'
        })
        price = price.sort_index(level=['ticker', 'date'])
        price['adj_factor'] = price['Adj Close'] / price['Close']
        return price[['adj_factor']]
    else:
        print("No data available for any ticker.")
        return None

# Get dividend data per ticker from start to current_date
def get_dividend_fmp(ticker_list, start, current_date):
    frames = []

    for ticker in tqdm(ticker_list, desc="Fetching data", unit="ticker"):
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?from={start}&to={current_date}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()

        # Check if there's dividend data in the response
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df['ticker'] = ticker
            frames.append(df)
        else:
            print(f"Skipped {ticker} - No data available")

    if frames:  # Only proceed if there are frames to concatenate
        dividend = pd.concat(frames)
        dividend['date'] = pd.to_datetime(dividend['date'], errors='coerce')
        dividend = dividend.set_index(['ticker', 'date'])
        cols_to_convert = ['dividend']
        dividend[cols_to_convert] = dividend[cols_to_convert].astype(float)
        return dividend.sort_index(level=['ticker', 'date'])
    else:
        print("No dividend data available for any ticker.")
        return None

# Get sp500 constituents from FMP
def get_sp500_fmp():
    url = f"https://financialmodelingprep.com/api/v3/historical/sp500_constituent?apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        sp500_constituents = response.json()
        # Get sp500 list
        sp500_ticker = [item['symbol'] for item in sp500_constituents]
        return sp500_constituents, sp500_ticker
    else:
        print("Failed to retrieve data:", response.status_code, response.text)
        return None

# Outputs window data from a specified date
def window_data(data, date, window):
    target_date = pd.to_datetime(date)
    def process_group(group):
        group_filtered = group.tail(window)
        if group_filtered.index.get_level_values('date').max() >= target_date and len(group_filtered) >= window:
            return group_filtered
        return None
    return pd.concat([process_group(group) for _, group in data.groupby(level='permno') if process_group(group) is not None])

