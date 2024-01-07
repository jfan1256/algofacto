from core.operation import *
class MrevSDEpsil:
    def __init__(self,
                 name=None,
                 threshold=None,
                 sbo=None,
                 sso=None,
                 sbc=None,
                 ssc=None):

        '''
        name (str): Name of beta columns
        threshold (int): Market cap threshold to determine if a stock is buyable/shortable
        sbo (float): Threshold to determine buy signal
        sso (float): Threshold to determine sell signal
        sbc (float): Threshold to determine close buy signal
        ssc (float): Threshold to determine close sell signal
        '''

        self.name = name
        self.threshold = threshold
        self.sbo = sbo
        self.sso = sso
        self.sbc = sbc
        self.ssc = ssc

    # Creates a multiindex of (permno, date) for a dataframe with only a date index
    @staticmethod
    def _create_multi_index(factor_data, stock):
        factor_values = pd.concat([factor_data] * len(stock), ignore_index=True).values
        multi_index = pd.MultiIndex.from_product([stock, factor_data.index])
        multi_index_factor = pd.DataFrame(factor_values, columns=factor_data.columns, index=multi_index)
        multi_index_factor.index = multi_index_factor.index.set_names(['permno', 'date'])
        return multi_index_factor

    # Create signals
    def _create_signal(self, data):
        def apply_rules(group):
            # Initialize signals and positions
            signals = [None] * len(group)
            positions = [None] * len(group)
            # Create masks for conditions
            open_long_condition = (group['s_score'] < -self.sbo) & (group['market_cap'] > self.threshold)
            open_short_condition = (group['s_score'] > self.sso) & (group['market_cap'] > self.threshold)
            close_long_condition = group['s_score'] > -self.ssc
            close_short_condition = group['s_score'] < self.sbc
            # Flag to check if any position is open
            position_open = False
            current_position = None

            for i in range(len(group)):
                if position_open:
                    if positions[i - 1] == 'long' and close_long_condition.iloc[i]:
                        signals[i] = 'close long'
                        positions[i] = None
                        position_open = False
                        current_position = None
                    elif positions[i - 1] == 'short' and close_short_condition.iloc[i]:
                        signals[i] = 'close short'
                        positions[i] = None
                        position_open = False
                        current_position = None
                    else:
                        signals[i] = 'hold'
                        positions[i] = current_position
                else:
                    if open_long_condition.iloc[i]:
                        positions[i] = 'long'
                        signals[i] = 'buy to open'
                        current_position = 'long'
                        position_open = True
                    elif open_short_condition.iloc[i]:
                        positions[i] = 'short'
                        signals[i] = 'sell to open'
                        position_open = True
                        current_position = 'short'

            return pd.DataFrame({'signal': signals, 'position': positions}, index=group.index)

        # Sort data
        data = data.sort_index(level=['permno', 'date'])
        # Group by permno and apply the rules for each group
        results = data.groupby('permno').apply(apply_rules).reset_index(level=0, drop=True)
        # Flatten the results and assign back to the data
        data = data.join(results)
        return data

    # Calculate weights and total portfolio return
    def calc_total_ret(self, df, hedge_ret):
        print("Get hedge weights...")
        mask_long = df['position'] == 'long'
        mask_short = df['position'] == 'short'
        df['hedge_weight'] = np.where(mask_long, -1, np.where(mask_short, 1, 0))

        # Get net hedge betas
        print("Get net hedge betas...")
        beta_columns = [col for col in df.columns if f"_{self.name}_" in col]
        weighted_betas = df[beta_columns].multiply(df['hedge_weight'], axis=0)
        net_hedge_betas = weighted_betas.groupby('date').sum()

        # Combine and normalize weights
        print("Normalize weights...")
        df['stock_weight'] = np.where(mask_long, 1, np.where(mask_short, -1, 0))

        # Normalize net hedge betas and stock weights combined
        df['abs_stock_weight'] = df['stock_weight'].abs()
        combined_weights = df.groupby('date')['abs_stock_weight'].sum() + net_hedge_betas.abs().sum(axis=1)
        df['normalized_weight'] = df['stock_weight'].div(combined_weights, axis=0)
        normalized_net_hedge_betas = net_hedge_betas.div(combined_weights, axis=0)

        # Get net hedge return
        print("Get net hedge returns...")
        net_hedge_returns = pd.DataFrame(index=normalized_net_hedge_betas.index)
        for beta in beta_columns:
            hedge_return_column = beta.split(f"_{self.name}_")[0]
            if hedge_return_column in hedge_ret.columns:
                net_hedge_returns[beta] = normalized_net_hedge_betas[beta] * hedge_ret[hedge_return_column]

        # Get total hedge return
        print("Get total hedge return...")
        net_hedge_return_total = net_hedge_returns.sum(axis=1)

        print("Get daily returns...")
        daily_returns = (df['RET_01'] * df['normalized_weight']).groupby('date').sum()

        print("Get total returns...")
        total_returns = daily_returns + net_hedge_return_total

        if 'ticker' in df.columns:
            return total_returns, normalized_net_hedge_betas, df[['normalized_weight', 'ticker']]
        else:
            return total_returns, normalized_net_hedge_betas, df[['normalized_weight']]