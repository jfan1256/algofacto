import pandas as pd
import quantstats as qs

class PortFactor:
    def __init__(self,
                 data=None,
                 window=None,
                 num_stocks=None,
                 factors=None,
                 threshold=None,
                 backtest=None,
                 dir_path=None):

        '''
        data (pd.DataFrame): A multiindex dataframe (permno, date) with columns of factors and 'RET_01'
        window (int): Rolling window size to calculate inverse volatility
        num_stocks (int): Number of stocks to long/short
        factors (list): A list of strings that are references to the factor columns to be used for ranking
        threshold (int): Market cap threshold to determine if a stock is buyable/shortable
        backtest (bool): Plot quantstat plot or not
        dir_path (Path): Directory path to export quantstats report
        '''

        self.data = data
        self.window = window
        self.num_stocks = num_stocks
        self.factors = factors
        self.threshold = threshold
        self.backtest = backtest
        self.dir_path = dir_path

    # Select top and bottom stocks
    def _select_long_short_stocks(self, group):
        top_stocks = group.nlargest(self.num_stocks, 'adj_weight')
        bottom_stocks = group.nsmallest(self.num_stocks, 'adj_weight')
        top_stocks['final_weight'] = top_stocks['adj_weight'] * 1
        bottom_stocks['final_weight'] = bottom_stocks['adj_weight'] * -1
        return pd.concat([top_stocks, bottom_stocks])

    # Create Factor-Based Portfolio
    def create_factor_port(self):
        df = self.data.copy(deep=True)
        # Filtering by Market Capitalization
        df = df[df['market_cap'] >= self.threshold]
        # Create ranks for each factor
        print("-" * 60)
        print("Creating Factor Ranks...")
        for factor_name in self.factors:
            print(f'Factor: {factor_name}')
            df[f'{factor_name}_Rank'] = df.groupby('date')[factor_name].rank(ascending=False)
        # Calculating average rank
        df['avg_rank'] = df[[f'{f}_Rank' for f in self.factors]].mean(axis=1)
        # Calculating rank weights
        df['rank_weight'] = (1 / len(self.factors)) * df['avg_rank']
        # Calculating inverse volatility (exclude the current date)
        df['vol'] = df.groupby('permno')['RET_01'].transform(lambda x: x.rolling(self.window).std().shift(1))
        df['inv_vol_weight'] = 1 / df['vol']
        # Find adjusted weight that accounts for rank and inverse volatility
        df['adj_weight'] = df['rank_weight'] * df['inv_vol_weight']
        # Selecting Top and Bottom Stocks
        print("-" * 60)
        print("Creating Long/Short portfolio...")
        top_bottom_stocks = df.groupby('date').apply(self._select_long_short_stocks).reset_index(level=0, drop=True)
        # Normalizing Weights
        top_bottom_stocks['final_weight'] /= top_bottom_stocks.groupby('date')['final_weight'].transform(lambda x: x.abs().sum())
        # Shift returns
        top_bottom_stocks['RET_01'] = top_bottom_stocks.groupby('permno')['RET_01'].shift(-1)
        top_bottom_stocks['total_ret'] = top_bottom_stocks['RET_01'] * top_bottom_stocks['final_weight']
        total_ret = top_bottom_stocks.groupby('date').total_ret.sum()
        if self.backtest:
            qs.reports.html(total_ret, 'SPY', output=self.dir_path)
        return top_bottom_stocks
