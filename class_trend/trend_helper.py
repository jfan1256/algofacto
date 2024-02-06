from core.operation import *

class TrendHelper:
    def __init__(self,
                 current_date,
                 start_date,
                 num_stocks):

        '''
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for backtest period)
        start_date (str: YYYY-MM-DD): Start date for backtest period
        num_stocks (int): Number of stocks to long
        '''

        self.current_date = current_date
        self.start_date = start_date
        self.num_stocks = num_stocks

    # Get return data for a list of tickers
    def _get_ret(self, ticker_list):
        data = get_data_fmp(ticker_list=ticker_list, start=self.start_date, current_date=self.current_date)
        data = data[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
        data = data.rename(columns={'Adj Close': 'Close'})
        data = create_return(data, [1])
        data = data.drop(['High', 'Low', 'Open', 'Volume'], axis=1)
        data = data.loc[~data.index.duplicated(keep='first')]
        data = data.fillna(0)
        return data

    # Calculate Trend + Bond/Commodity Portfolio
    @staticmethod
    def _calc_total_port(row, col1, col2):
        if row[col1] == 0:
            return 0 * row[col1] * 1.0 * row[col2]
        elif row['macro_buy']:
            return 0.50 * row[col1] + 0.50 * row[col2]
        else:
            return 0.25 * row[col1] + 0.75 * row[col2]

    # Retrieves the top self.num_stocks stocks with the greatest inverse volatility weight
    def _top_inv_vol(self, df):
        filtered_df = df[df['signal'].abs() == 1]
        return filtered_df.nlargest(self.num_stocks, 'inv_vol')