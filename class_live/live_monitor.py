from core.operation import *

import quantstats as qs

class LiveMonitor:
    def __init__(self,
                 capital=None,
                 strat_name=None,
                 strat_file=None,
                 allocate=None,
                 alpha_windows=None,
                 output_path=None):

        '''
        capital (int): Total capital for portfolio
        strat_name (str): Name of strategy (use class name)
        strat_csv (Path): CSV path to strategy's stock
        allocate (float): Percentage of capital allocated to this strategy
        windows (list): List of rolling window sizes for alpha report
        output_path (Path): Output path of results
        '''

        self.capital = capital
        self.strat_name = strat_name
        self.strat_file = strat_file
        self.allocate = allocate
        self.alpha_windows = alpha_windows
        self.output_path = output_path

    # Execute alpha report
    @staticmethod
    def _rolling_full_alpha(strat_ret, windows, name, path):
        # Read in risk-free rate
        risk_free = pd.read_parquet(get_parquet(True) / 'data_rf.parquet.brotli')
        strat_ret.columns = ['strat_ret']
        strat_ret = strat_ret.merge(risk_free, left_index=True, right_index=True, how='left')
        strat_ret['strat_ret'] -= strat_ret['RF']

        # Read in SPY data and adjust it with Risk-free rate
        spy = get_spy(start_date=strat_ret.index.min().strftime('%Y-%m-%d'), end_date=strat_ret.index.max().strftime('%Y-%m-%d'))
        spy.columns = ['spy_ret']
        strat_ret = strat_ret.merge(spy, left_index=True, right_index=True, how='left')
        strat_ret['spy_ret'] -= strat_ret['RF']

        # Perform full OLS regression
        strat_ret['const'] = 1
        full_ols_model = sm.OLS(strat_ret['strat_ret'], strat_ret[['const', 'spy_ret']]).fit()
        full_alpha = full_ols_model.params['const']
        full_beta = full_ols_model.params['spy_ret']
        full_p_value_alpha = full_ols_model.pvalues['const']

        # Perform rolling OLS
        for window in windows:
            rolling_results = RollingOLS(endog=strat_ret['strat_ret'], exog=strat_ret[['const', 'spy_ret']], window=window).fit()
            strat_ret[f'alpha_{window}'] = rolling_results.params['const']
            strat_ret[f'beta_{window}'] = rolling_results.params['spy_ret']
            strat_ret[f'p_value_alpha_{window}'] = rolling_results.pvalues['const']

        # Create plots
        total_rows = 3 * (len(windows) + 1)  # Extra rows for full OLS results
        fig = make_subplots(rows=total_rows, cols=1,
                            subplot_titles=[f"{metric} (Window: {window})" for window in windows for metric in ['Alpha', 'Beta', 'P-Value of Alpha']] + ['Full Alpha', 'Full Beta', 'Full P-Value of Alpha'])

        # Create rolling window plots
        current_row = 1
        for window in windows:
            fig.add_trace(go.Scatter(x=strat_ret.index, y=strat_ret[f'alpha_{window}'], name=f'Alpha (Window: {window})'), row=current_row, col=1)
            current_row += 1
            fig.add_trace(go.Scatter(x=strat_ret.index, y=strat_ret[f'beta_{window}'], name=f'Beta (Window: {window})'), row=current_row, col=1)
            current_row += 1
            fig.add_trace(go.Scatter(x=strat_ret.index, y=strat_ret[f'p_value_alpha_{window}'], name=f'P-Value of Alpha (Window: {window})'), row=current_row, col=1)
            current_row += 1

        # Create full ols plot
        fig.add_trace(go.Scatter(x=strat_ret.index, y=[full_alpha] * len(strat_ret), name='Full Alpha'), row=current_row, col=1)
        current_row += 1
        fig.add_trace(go.Scatter(x=strat_ret.index, y=[full_beta] * len(strat_ret), name='Full Beta'), row=current_row, col=1)
        current_row += 1
        fig.add_trace(go.Scatter(x=strat_ret.index, y=[full_p_value_alpha] * len(strat_ret), name='Full P-Value of Alpha'), row=current_row, col=1)

        # Update Layout
        fig.update_layout(height=300 * total_rows, width=1200, title_text="Rolling and Full Alpha, Beta, and P-Value Analysis", showlegend=True)

        # Save the figure
        filename = path / f'{name}.html'
        fig.write_html(str(filename), auto_open=False)

    # Monitor Strategy
    def exec_monitor_strat(self):
        print(f"--------------------------------------------------------------MONITOR {self.strat_name}--------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------MONITOR--------------------------------------------------------------------------------------------
        # Load in data
        strat_weight = pd.read_parquet(get_live() / self.strat_file)
        # Reset index to just be ('date', 'ticker')
        strat_weight = strat_weight.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])

        if self.strat_name in ['StratMLRet', 'StratPortIV', 'StratPortID', 'StratPortIVM']:
            strat_price = pd.read_parquet(get_live() / 'data_permno_store.parquet.brotli')
            strat_price = strat_price.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])
        elif self.strat_name in ['StratMrevETF']:
            strat_price = pd.read_parquet(get_live() / 'data_permno_store.parquet.brotli')
            strat_price = strat_price.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])
            hedge_price = pd.read_parquet(get_live() / 'data_mrev_etf_hedge_store.parquet.brotli')
            hedge_price = hedge_price.swaplevel()
            strat_price = pd.concat([strat_price, hedge_price], axis=0).sort_index(level=['date', 'ticker'])
        elif self.strat_name in ['StratMrevMkt']:
            strat_price = pd.read_parquet(get_live() / 'data_permno_store.parquet.brotli')
            strat_price = strat_price.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])
            hedge_price = pd.read_parquet(get_live() / 'data_mrev_mkt_hedge_store.parquet.brotli')
            hedge_price = hedge_price.swaplevel()
            strat_price = pd.concat([strat_price, hedge_price], axis=0).sort_index(level=['date', 'ticker'])
        elif self.strat_name in ['StratTrendMLS']:
            strat_price = pd.read_parquet(get_live() / 'data_permno_store.parquet.brotli')
            strat_price = strat_price.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])
            bond_price = pd.read_parquet(get_live() / 'data_trend_mls_bond_store.parquet.brotli')
            bond_price = bond_price.swaplevel()
            com_price = pd.read_parquet(get_live() / 'data_trend_mls_com_store.parquet.brotli')
            com_price = com_price.swaplevel()
            strat_price = pd.concat([strat_price, bond_price, com_price], axis=0).sort_index(level=['date', 'ticker'])
        elif self.strat_name in ['StratMLTrend']:
            strat_price = pd.read_parquet(get_live() / 'data_permno_store.parquet.brotli')
            strat_price = strat_price.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])
            bond_price = pd.read_parquet(get_live() / 'data_ml_trend_bond_store.parquet.brotli')
            bond_price = bond_price.swaplevel()
            re_price = pd.read_parquet(get_live() / 'data_ml_trend_re_store.parquet.brotli')
            re_price = re_price.swaplevel()
            strat_price = pd.concat([strat_price, bond_price, re_price], axis=0).sort_index(level=['date', 'ticker'])

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------PREPARE DATA---------------------------------------------------------------------------------------
        # Merge to dataframes
        strat_data = pd.merge(strat_price, strat_weight, left_index=True, right_index=True, how='left')

        # Get monetary value per stock
        strat_data['monetary_value'] = strat_data['weight'] * self.capital

        # Calculate int shares
        strat_data['share'] = (strat_data['monetary_value'] / strat_data['Close']).apply(np.floor)
        # Replace any 0 values with 1
        strat_data['shares'] = np.where(strat_data['shares'] == 0, 1, strat_data['shares'])
        del strat_data['monetary_value']

        # Calculate actual capital deployed (after int shares shares)
        strat_data['capital'] = strat_data['share'] * strat_data['Close']

        # Recalculate weights based on actual capital deployed
        strat_data['weight_share'] = strat_data.groupby('date')['capital'].transform(lambda x: x / x.sum())
        del strat_data['capital']

        # Export dataframe
        strat_data.to_parquet(self.output_path / 'data_strat.parquet.brotli', compression='brotli')

        # Adjust strat_data weights to be allocation of 1
        strat_data['weight_share'] = strat_data['weight_share'] / self.allocate
        strat_data['weight'] = strat_data['weight'] / self.allocate

        # Create returns and shift it by -1 for alignment of weights
        strat_data = create_return(strat_data, [1])
        strat_data['RET_01'] = strat_data.groupby('ticker')['RET_01'].shift(-1).reset_index(level=0, drop=True)

        # Change short weights to negative
        strat_data.loc[strat_data['type'] == 'short', 'weight_share'] = strat_data.loc[strat_data['type'] == 'short', 'weight_share'] * -1
        strat_data.loc[strat_data['type'] == 'short', 'weight'] = strat_data.loc[strat_data['type'] == 'short', 'weight'] * -1

        # Get daily strategy returns
        daily_strat_share_ret = strat_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight_share']).sum())
        daily_strat_ret = strat_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight']).sum())

        # Export qs report
        qs.reports.html(daily_strat_share_ret, 'SPY', output=self.output_path / 'qs_share_report.html')
        qs.reports.html(daily_strat_ret, 'SPY', output=self.output_path / 'qs_report.html')

        # Export CAPM statistical test
        daily_strat_share_ret_df = daily_strat_share_ret.to_frame()
        daily_strat_df = daily_strat_ret.to_frame()
        self._rolling_full_alpha(strat_ret=daily_strat_share_ret_df, windows=self.alpha_windows, name='alpha_share_report', path=self.output_path)
        self._rolling_full_alpha(strat_ret=daily_strat_df, windows=self.alpha_windows, name='alpha_report', path=self.output_path)

    # Monitor Total Portfolio (make sure to run this after you have monitored all strategies first)
    def exec_monitor_all(self):
        print(f"------------------------------------------------------------------MONITOR ALL------------------------------------------------------------------------------------------")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------MONITOR ALL------------------------------------------------------------------------------------------
        # Load in data
        strat_ml_ret = pd.read_parquet(get_live_monitor() / 'strat_ml_ret' / 'data_strat.parquet.brotli')
        strat_ml_trend = pd.read_parquet(get_live_monitor() / 'strat_ml_trend' / 'data_strat.parquet.brotli')
        strat_mrev_etf = pd.read_parquet(get_live_monitor() / 'strat_mrev_etf' / 'data_strat.parquet.brotli')
        strat_mrev_mkt = pd.read_parquet(get_live_monitor() / 'strat_mrev_mkt' / 'data_strat.parquet.brotli')
        strat_port_iv = pd.read_parquet(get_live_monitor() / 'strat_port_iv' / 'data_strat.parquet.brotli')
        strat_port_id = pd.read_parquet(get_live_monitor() / 'strat_port_id' / 'data_strat.parquet.brotli')
        strat_port_ivm = pd.read_parquet(get_live_monitor() / 'strat_port_ivm' / 'data_strat.parquet.brotli')
        strat_trend_mls = pd.read_parquet(get_live_monitor() / 'strat_trend_mls' / 'data_strat.parquet.brotli')

        # Merge all data
        total_strat_data = pd.concat([strat_ml_ret, strat_ml_trend, strat_mrev_etf, strat_mrev_mkt, strat_port_iv, strat_port_id, strat_port_ivm, strat_trend_mls], axis=0)

        # Extract ticker and price (get a unique price for each unique {date, ticker} index pair)
        strat_price = total_strat_data[['Close']].copy(deep=True)
        strat_price = strat_price.loc[~strat_price.index.duplicated(keep='last')]

        # Merge data by 'date', 'ticker', 'type' to calculate total weight per type per stock
        total_strat_data = total_strat_data.reset_index().set_index(['date', 'ticker', 'type'])
        strat_share_data = total_strat_data[['weight_share']]
        strat_data = total_strat_data[['weight']]
        strat_share_data = strat_share_data.groupby(level=['date', 'ticker', 'type']).sum()
        strat_data = strat_data.groupby(level=['date', 'ticker', 'type']).sum()

        # Convert 'type' into positive and negative weights (long and short)
        strat_share_data['signed_weight_share'] = np.where(strat_share_data.index.get_level_values('type') == 'long', strat_share_data['weight_share'], -strat_share_data['weight_share'])
        strat_data['signed_weight'] = np.where(strat_data.index.get_level_values('type') == 'long', strat_data['weight'], -strat_data['weight'])
        # Sum signed weights by 'date' and 'ticker'
        net_weights_share = strat_share_data.groupby(['date', 'ticker'])['signed_weight_share'].sum()
        net_weights = strat_data.groupby(['date', 'ticker'])['signed_weight'].sum()
        # Determine the 'type' based on the sign of the net weight
        net_weights_share = net_weights_share.reset_index()
        net_weights_share['type'] = np.where(net_weights_share['signed_weight_share'] > 0, 'long', 'short')
        net_weights = net_weights.reset_index()
        net_weights['type'] = np.where(net_weights['signed_weight'] > 0, 'long', 'short')
        # Assign absolute values to the new weight column
        net_weights_share['weight_share'] = net_weights_share['signed_weight_share'].abs()
        del net_weights_share['signed_weight_share']
        net_weights['weight'] = net_weights['signed_weight'].abs()
        del net_weights['signed_weight']
        # Create final strat_data and strat_share_data dataframe with net weights across tickers (ensures that no stock enters both a long and short position)
        strat_share_data = net_weights_share.set_index(['date', 'ticker', 'type'])
        strat_data = net_weights.set_index(['date', 'ticker', 'type'])

        # Reset strat_data and strat_share_data index to be ('date', 'ticker')
        strat_share_data = strat_share_data.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])
        strat_data = strat_data.reset_index().set_index(['date', 'ticker']).sort_index(level=['date', 'ticker'])

        # Add price back to strat_data and strat_share_data
        strat_share_data = strat_share_data.merge(strat_price, left_index=True, right_index=True, how='left')
        strat_data = strat_data.merge(strat_price, left_index=True, right_index=True, how='left')

        # Export Data
        strat_share_data.to_parquet(self.output_path / 'data_strat_share.parquet.brotli', compression='brotli')
        strat_data.to_parquet(self.output_path / 'data_strat.parquet.brotli', compression='brotli')

        # Create returns and shift it by -1 for alignment of weights
        strat_share_data = create_return(strat_share_data, [1])
        strat_share_data['RET_01'] = strat_share_data.groupby('ticker')['RET_01'].shift(-1).reset_index(level=0, drop=True)
        strat_data = create_return(strat_data, [1])
        strat_data['RET_01'] = strat_data.groupby('ticker')['RET_01'].shift(-1).reset_index(level=0, drop=True)

        # Change short weights to negative
        strat_share_data.loc[strat_share_data['type'] == 'short', 'weight_share'] = strat_share_data.loc[strat_share_data['type'] == 'short', 'weight_share'] * -1
        strat_data.loc[strat_data['type'] == 'short', 'weight'] = strat_data.loc[strat_data['type'] == 'short', 'weight'] * -1

        # Get daily strategy returns
        daily_strat_ret_share = strat_share_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight_share']).sum())
        daily_strat_ret = strat_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight']).sum())

        # Export qs report
        qs.reports.html(daily_strat_ret_share, 'SPY', output=self.output_path / 'qs_share_report.html')
        qs.reports.html(daily_strat_ret, 'SPY', output=self.output_path / 'qs_report.html')

        # Export CAPM statistical test
        daily_strat_share_df = daily_strat_ret_share.to_frame()
        daily_strat_df = daily_strat_ret.to_frame()
        self._rolling_full_alpha(strat_ret=daily_strat_share_df, windows=self.alpha_windows, name='alpha_share_report', path=self.output_path)
        self._rolling_full_alpha(strat_ret=daily_strat_df, windows=self.alpha_windows, name='alpha_share', path=self.output_path)





