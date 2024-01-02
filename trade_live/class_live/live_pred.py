import os
import base64
import math

from io import BytesIO
from IPython.display import Image

from core.operation import *
from core.system import *

import warnings

warnings.filterwarnings('ignore')
class LivePred:
    def __init__(self,
                 live: bool = None,
                 model_name: str = None,
                 num_stocks: int = None,
                 leverage: float = None,
                 port_opt: str = None,
                 current_date: str = None,
                 dir_path: Path = None):

        '''
        live (bool): Get live data or historical data
        model_name (str): Model name
        num_stocks (int): Number of stocks to long/short
        leverage (int): Leverage value for long/short (i.e., 0.5 means 0.5 * long + 0.5 short)
        port_opt (str): Type of portfolio optimization to use
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for backtest)
        dir_path (Path): Directory path to export backtest result
        '''

        self.live = live
        self.model_name = model_name
        self.num_stocks = num_stocks
        self.leverage = leverage
        self.port_opt = port_opt
        self.current_date = current_date
        self.dir_path = dir_path

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------FILE HANDLER----------------------------------------------------------------------------------
    # Reads in all the results in the "modelName" folder
    def read_result(self, result_name):
        data = []
        result_data_dir = get_ml_result_model(self.live, self.model_name)
        for i, folder_name in enumerate(os.listdir(result_data_dir)):
            try:
                if folder_name.startswith("params"):
                    folder_path = os.path.join(result_data_dir, folder_name)
                    file_path = os.path.join(folder_path, f"{result_name}.parquet.brotli")
                    print(os.path.basename(folder_path))
                    data.append(pd.read_parquet(file_path))
            except:
                continue
        return pd.concat(data, axis=0).reset_index(drop=True)

    # Get the maximum overall daily_metric in each result to find the best performing model
    @staticmethod
    def get_max_metric(data):
        collection = {}
        for index, row in data.iterrows():
            collection[max(row.loc[(row.index.str.startswith("daily_metric"))])] = index
        max_ic_idx = collection[max(list(collection.keys()))]
        return data.iloc[max_ic_idx]

    # Gets the files of the best performing model
    def get_max_metric_file(self, data):
        files = {}
        time_index = data.to_frame().index.get_loc('time')
        param_vals = data.iloc[:time_index].values
        key = [f'{float(p)}' for p in (param_vals)]
        key = '_'.join(key)

        result_data_dir = get_ml_result_model(self.live, self.model_name) / f'params_{key}'
        for file in os.listdir(result_data_dir):
            if file.endswith(".parquet.brotli"):
                files[extract_first_string(file)] = pd.read_parquet(os.path.join(result_data_dir, file))
            elif file.endswith(".png"):
                img = os.path.join(result_data_dir, file)
                files[extract_first_string(file)] = Image(img)
        return files

    # Get all files under a specific folder
    def get_all(self):
        return self.get_max_metric_file(self.get_max_metric(self.read_result('metrics')))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------PORTFOLIO OPTIMIZATION----------------------------------------------------------------------------------
    # Equal Weight
    def ewp(self, returns):
        weight = self.leverage / len(returns.columns)
        return np.full(len(returns.columns), weight)

    def exec_port_opt(self, data):
        if self.port_opt == 'equal_weight':
            # Get long returns
            long_returns = pd.DataFrame(data['longRet'].tolist())
            long_weights = self.ewp(long_returns)
            long_weights = np.tile(long_weights, (len(long_returns), 1))

            # Get short returns
            short_returns = -1 * pd.DataFrame(data['shortRet'].tolist())
            short_weights = self.ewp(short_returns)
            short_weights = np.tile(short_weights, (len(short_returns), 1))

            # Calculate equal-weights and sum up to get total strategy return
            total_ret = np.sum(long_returns.values * long_weights, axis=1) + np.sum(short_returns.values * short_weights, axis=1)
            data['totalRet'] = total_ret
        return data, long_weights, short_weights

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------PROCESSING PERIODS (BACKTEST)---------------------------------------------------------------------------------
    # Calculate sign accuracy
    @staticmethod
    def sign_accuracy(predictions, actual, target_sign, pred):
        accuracies = []  # To store accuracies for each ticker

        # Iterate through each ticker and calculate accuracy
        for ticker in predictions.index.get_level_values(0).unique():
            ticker_group = predictions.loc[ticker]
            actual_group = actual.loc[ticker]

            # Determine if each pair has the same sign
            if pred == 'price':
                correct_signs = (np.sign(ticker_group) == np.sign(actual_group))
            elif pred == 'sign':
                correct_signs = (ticker_group == np.sign(actual_group))

            # Filter by target sign if specified
            if target_sign == 'positive':
                mask = (np.sign(actual_group) == 1)
                correct_signs = correct_signs[mask]
            elif target_sign == 'negative':
                mask = (np.sign(actual_group) == -1)
                correct_signs = correct_signs[mask]

            # Calculate the accuracy and store it
            accuracy = np.mean(correct_signs) * 100  # Convert to percentage
            accuracies.append(accuracy)

        # Calculate and return the mean accuracy across all tickers
        mean_accuracy = np.nanmean(accuracies)
        return mean_accuracy

    # Performs ranking across stocks on a daily basis
    def process_period(self, period, period_returns, candidates, threshold):
        # Find sp500 candidates for the given year and assign it to data
        period_year = period.index.get_level_values('date')[0].year
        sp500 = candidates[period_year]
        tickers = common_stocks(sp500, period)
        sp500_period = get_stocks_data(period, tickers)

        # Filter the DataFrame to only include rows with market cap over the threshold
        filtered_period = period[period['market_cap'] > threshold]
        print(f'{period_year} --> Num of stocks to select from: ' + str(len(get_stock_idx(filtered_period))))

        # Group by date and compute long and short stocks and their returns
        for date, stocks in filtered_period.groupby('date'):
            sorted_stocks = stocks.sort_values(by='predictions')
            long_stocks = sorted_stocks.index.get_level_values('ticker')[-self.num_stocks:]
            short_stocks = sorted_stocks.index.get_level_values('ticker')[:self.num_stocks]

            # Get ticker, exchange tuples
            df_reset = sorted_stocks.reset_index()
            filtered_rows = df_reset[df_reset['ticker'].isin(long_stocks)]
            long_stocks_tuples = list(zip(filtered_rows['ticker'], filtered_rows['exchange']))

            filtered_rows = df_reset[df_reset['ticker'].isin(short_stocks)]
            short_stocks_tuples = list(zip(filtered_rows['ticker'], filtered_rows['exchange']))

            # Store results in period_returns DataFrame
            period_returns.loc[date] = [long_stocks_tuples, sorted_stocks.iloc[-self.num_stocks:].returns.values,
                                        short_stocks_tuples, sorted_stocks.iloc[:self.num_stocks].returns.values]

    # Creates periods (determined from window period in the multindex) and construct final dataframe
    def backtest(self, data, threshold):
        # Set portfolio weights and other tracking variables
        period_returns = pd.DataFrame(columns=['longStocks', 'longRet', 'shortStocks', 'shortRet'])

        # Get candidates
        candidates = get_candidate(self.live)

        # Loop over each group in tic.groupby('window')
        for _, df in data.groupby('window'):
            df = df.reset_index().set_index(['ticker', 'date']).drop('window', axis=1)
            self.process_period(df, period_returns, candidates, threshold)

        return period_returns

    def price(self, best_model_params, dir_path, iteration, plot):
        # Gets the predictions of the highest overall daily_metric in the boosted round cases
        if iteration == False:
            best_prediction = best_model_params['predictions'][
                [str(extract_number(best_model_params['metrics'].loc[:, best_model_params['metrics'].columns.str.startswith("daily_metric")].idxmax(axis=1)[0])), 'i']]
        else:
            best_prediction = best_model_params['predictions'][[str(iteration), 'i']]
        actual_return = best_model_params['returns']
        # Merge actual returns and prediction returns
        merged = pd.merge(best_prediction, actual_return, left_index=True, right_index=True, how='left')
        merged.columns = ['predictions', 'window', 'returns']
        merged.window = merged.window.astype(int)
        # Shift actual returns 1 day back
        merged['returns'] = merged.groupby('permno')['returns'].shift(-1)
        merged = remove_nan_before_end(merged, 'returns')

        if plot == False:
            return merged

        print('Best num_iterations: ' + str(best_prediction.columns[0]))
        print(f"Neutral Accuracy: {round(self.sign_accuracy(merged.predictions, merged.returns, None, 'price'), 2)}%")
        print(f"Positive Accuracy: {round(self.sign_accuracy(merged.predictions, merged.returns, 'positive', 'price'), 2)}%")
        print(f"Negative Accuracy: {round(self.sign_accuracy(merged.predictions, merged.returns, 'negative', 'price'), 2)}%")
        metrics = best_model_params['metrics']
        column_widths = [max(len(str(val)) for val in metrics[col]) for col in metrics.columns]
        header_widths = [len(header) for header in metrics.columns]
        max_widths = [max(col_width, header_width) for col_width, header_width in zip(column_widths, header_widths)]
        headers = " | ".join([header.ljust(width) for header, width in zip(metrics.columns, max_widths)])
        values = " | ".join([str(val).ljust(width) for val, width in zip(metrics.iloc[0], max_widths)])
        formatted_metrics = headers + "\n" + values
        print(formatted_metrics)

        # Convert to HTML
        df_html = best_model_params['metrics'].to_html(classes='my-table')

        # Prepare the plot
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        best_model_params['daily_metric'][best_prediction.columns[0]].rolling(window=42).mean().plot(
            ax=ax, linewidth=0.5, color='blue', linestyle='-', title='Daily Metric Plot'
        )
        ax.set(xlabel='Date', ylabel='Daily Metric')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Convert plot to SVG
        img_stream = BytesIO()
        fig.savefig(img_stream, format='png')
        img_base64 = base64.b64encode(img_stream.getvalue()).decode()

        with open(dir_path / 'metric.html', 'w') as f:
            f.write('''<!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>Metrics Report</title>
            <style>
                body {
                    font-family: 'Times New Roman', serif;
                    margin: 0;
                    padding: 0;
                    background-color: #ffffff;
                    text-align: center;
                    color: #000000;
                }
                header {
                    background: #d3d3d3;
                    color: #000000;
                    text-align: center;
                    padding-top: 30px;
                    min-height: 70px;
                    margin: 0 auto;
                }
                .container {
                    width: 70%;
                    margin: auto;
                }
                .main-content {
                    padding: 30px;
                }
                .my-table {
                    width: 100%; 
                    border-collapse: collapse;
                    font-size: 12px;
                    table-layout: fixed; 
                }
                th, td {
                    border: 2px solid #e3e3e3;
                    padding: 10px;
                    text-align: left;
                    word-wrap: break-word;
                }
            </style>
            </head>
            <body>
                <header>
                    <h1>Metrics Report</h1>
                </header>
                <div class="container">
                    <div class="main-content">
                        <p>Best num_iterations: ''' + str(best_prediction.columns[0]) + '''</p>
                        <p>Neutral Accuracy: ''' + f"{round(self.sign_accuracy(merged.predictions, merged.returns, None, 'price'), 2)}%" + '''</p>
                        <p>Positive Accuracy: ''' + f"{round(self.sign_accuracy(merged.predictions, merged.returns, 'positive', 'price'), 2)}%" + '''</p>
                        <p>Negative Accuracy: ''' + f"{round(self.sign_accuracy(merged.predictions, merged.returns, 'negative', 'price'), 2)}%" + '''</p>
                        ''' + df_html + '''
                        <img src="data:image/png;base64,''' + img_base64 + '''" alt="plot" />
                    </div>
                </div>
            </body>
            </html>''')
        return merged

    def plot_ensemble(self, merged, ic_by_day):
        print(f'Daily Metric Mean: {round(ic_by_day.mean()[0], 5)}')
        # Prepare the plot
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        ic_by_day.rolling(window=42).mean().plot(ax=ax, linewidth=0.5, color='blue', linestyle='-', title='Daily Metric Plot')
        ax.set(xlabel='Date', ylabel='Daily Metric')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Convert plot to SVG
        img_stream = BytesIO()
        fig.savefig(img_stream, format='png')
        img_base64 = base64.b64encode(img_stream.getvalue()).decode()

        with open(self.dir_path / 'metric.html', 'w') as f:
            f.write('''<!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>Metrics Report</title>
            <style>
                body {
                    font-family: 'Times New Roman', serif;
                    margin: 0;
                    padding: 0;
                    background-color: #ffffff;
                    text-align: center;
                    color: #000000;
                }
                header {
                    background: #d3d3d3;
                    color: #000000;
                    text-align: center;
                    padding-top: 30px;
                    min-height: 70px;
                    margin: 0 auto;
                }
                .container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }
                .main-content {
                    padding: 30px;
                }
                .my-table {
                    width: 100%; 
                    border-collapse: collapse;
                    font-size: 12px;
                    table-layout: fixed; 
                }
                th, td {
                    border: 2px solid #e3e3e3;
                    padding: 10px;
                    text-align: left;
                    word-wrap: break-word;
                }
            </style>
            </head>
            <body>
                <header>
                    <h1>Metrics Report</h1>
                </header>
                <div class="container">
                    <div class="main-content">
                        <p>Neutral Accuracy: ''' + f"{round(self.sign_accuracy(merged.predictions, merged.returns, None, 'price'), 2)}%" + '''</p>
                        <p>Positive Accuracy: ''' + f"{round(self.sign_accuracy(merged.predictions, merged.returns, 'positive', 'price'), 2)}%" + '''</p>
                        <p>Negative Accuracy: ''' + f"{round(self.sign_accuracy(merged.predictions, merged.returns, 'negative', 'price'), 2)}%" + '''</p>
                        <img src="data:image/png;base64,''' + img_base64 + '''" alt="plot" />
                    </div>
                </div>
            </body>
            </html>''')

    # Print stocks to long and short in right format
    @staticmethod
    def display_stock(stocks, title):
        n = len(stocks)
        cols = int(math.sqrt(2 * n))
        max_length = max([len(item[0]) for item in stocks])

        text_content = f"{title}\n"
        border_line = '+' + '-' * (max_length + 3) * cols + '+\n'

        text_content += border_line
        for i in range(n):
            text_content += f"| {stocks[i][0].center(max_length)} "
            if (i + 1) % cols == 0:
                text_content += "|\n"
                text_content += border_line
        return text_content

    # Gets the best model predictions (determined from daily_metric) used for SHARPE calculation
    def sharpe_ret(self, best_model_params, iteration):
        # Gets the predictions of the highest overall daily_metric in the boosted round cases
        if iteration == False:
            best_prediction = best_model_params['predictions'][
                [str(extract_number(best_model_params['metrics'].loc[:, best_model_params['metrics'].columns.str.startswith("daily_metric")].idxmax(axis=1)[0])), 'i']]
        else:
            best_prediction = best_model_params['predictions'][[str(iteration), 'i']]
        actual_return = best_model_params['returns']
        # Merge actual returns and prediction returns
        merged = pd.merge(best_prediction, actual_return, left_index=True, right_index=True, how='left')
        merged.columns = ['predictions', 'window', 'returns']
        merged.window = merged.window.astype(int)
        # Shift actual returns 1 day back
        merged['returns'] = merged.groupby('permno')['returns'].shift(-1)
        merged = remove_nan_before_end(merged, 'returns')
        return merged

    # Calculates SHARPE for each period
    def sharpe_process_period(self, period, period_returns, candidates, threshold):
        # Find sp500 candidates for the given year and assign it to data
        period_year = period.index.get_level_values('date')[0].year
        sp500 = candidates[period_year]
        tickers = common_stocks(sp500, period)
        sp500_period = get_stocks_data(period, tickers)

        # Filter the DataFrame to only include rows with market cap over the threshold
        filtered_period = period[period['market_cap'] > threshold]

        # Group by date and compute long and short stocks and their returns
        for date, stocks in filtered_period.groupby('date'):
            sorted_stocks = stocks.sort_values(by='predictions')
            long_stocks = sorted_stocks.index.get_level_values('ticker')[-self.num_stocks:]
            short_stocks = sorted_stocks.index.get_level_values('ticker')[:self.num_stocks]

            # Store results in period_returns DataFrame
            period_returns.loc[date] = [long_stocks.tolist(), sorted_stocks.iloc[-self.num_stocks:].returns.values,
                                        short_stocks.tolist(), sorted_stocks.iloc[:self.num_stocks].returns.values]

    # Calculates SHARPE for each file
    def sharpe_backtest(self, data, threshold):
        # Set portfolio weights and other tracking variables
        period_returns = pd.DataFrame(columns=['longStocks', 'longRet', 'shortStocks', 'shortRet'])

        # Get candidates
        candidates = get_candidate(self.live)

        # Loop over each group in tic.groupby('window')
        for _, df in data.groupby('window'):
            df = df.reset_index().set_index(['ticker', 'date']).drop('window', axis=1)
            self.sharpe_process_period(df, period_returns, candidates, threshold)

        return period_returns
