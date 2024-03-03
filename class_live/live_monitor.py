import os
import quantstats as qs

from bs4 import BeautifulSoup

from core.operation import *

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
    def _rolling_full_alpha(strat_ret, spy, windows, name, path):
        # Read in risk-free rate
        risk_free = pd.read_parquet(get_parquet(True) / 'data_rf.parquet.brotli')
        strat_ret.columns = ['strat_ret']
        strat_ret = strat_ret.merge(risk_free, left_index=True, right_index=True, how='left')
        strat_ret['RF'] = strat_ret['RF'].ffill()
        strat_ret['strat_ret'] -= strat_ret['RF']
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
        total_rows = 3 * (len(windows) + 1)
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

        if self.strat_name in ['StratMLRet', 'StratPortIV', 'StratPortID', 'StratPortIM']:
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
        strat_price.index = strat_price.index.map(lambda x: (pd.to_datetime(x[0]), x[1]))
        strat_weight.index = strat_weight.index.map(lambda x: (pd.to_datetime(x[0]), x[1]))
        strat_data = pd.merge(strat_price, strat_weight, left_index=True, right_index=True, how='left')

        # Get monetary value per stock
        strat_data['monetary_value'] = strat_data['weight'] * self.capital

        # Calculate int shares
        strat_data['share'] = (strat_data['monetary_value'] / strat_data['Close']).apply(np.floor)
        # Replace any 0 values with 1
        strat_data['share'] = np.where(strat_data['share'] == 0, 1, strat_data['share'])
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
        strat_data['RET_01'] = strat_data.groupby('ticker').Close.pct_change(1)
        strat_data['RET_01'] = strat_data.groupby('ticker')['RET_01'].shift(-1)

        # Change short weights to negative
        strat_data.loc[strat_data['type'] == 'short', 'weight_share'] = strat_data.loc[strat_data['type'] == 'short', 'weight_share'] * -1
        strat_data.loc[strat_data['type'] == 'short', 'weight'] = strat_data.loc[strat_data['type'] == 'short', 'weight'] * -1

        # Get daily strategy returns
        daily_strat_share_ret = strat_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight_share']).sum())
        daily_strat_ret = strat_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight']).sum())

        # Export qs report
        start_date = (daily_strat_ret.index.min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (daily_strat_ret.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        spy = get_spy(start_date=start_date, end_date=end_date)['spyRet']
        day_zero = daily_strat_share_ret.index[0] - pd.Timedelta(days=1)
        daily_strat_share_ret = pd.concat([pd.Series([1e-4], index=[day_zero]), daily_strat_share_ret])
        daily_strat_ret = pd.concat([pd.Series([1e-4], index=[day_zero]), daily_strat_ret])
        spy = pd.concat([pd.Series([1e-4], index=[day_zero]), spy])
        spy = spy.to_frame('spy_ret')
        qs.reports.html(daily_strat_share_ret, spy, output=self.output_path / 'qs_share_report.html')
        qs.reports.html(daily_strat_ret, spy, output=self.output_path / 'qs_report.html')

        # Export CAPM statistical test
        daily_strat_share_ret_df = daily_strat_share_ret.to_frame()
        daily_strat_df = daily_strat_ret.to_frame()
        self._rolling_full_alpha(strat_ret=daily_strat_share_ret_df, spy=spy, windows=self.alpha_windows, name='alpha_share_report', path=self.output_path)
        self._rolling_full_alpha(strat_ret=daily_strat_df, spy=spy, windows=self.alpha_windows, name='alpha_report', path=self.output_path)

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
        strat_port_im = pd.read_parquet(get_live_monitor() / 'strat_port_im' / 'data_strat.parquet.brotli')
        strat_trend_mls = pd.read_parquet(get_live_monitor() / 'strat_trend_mls' / 'data_strat.parquet.brotli')

        # Merge all data
        total_strat_data = pd.concat([strat_ml_ret, strat_ml_trend, strat_mrev_etf, strat_mrev_mkt, strat_port_iv, strat_port_id, strat_port_im, strat_trend_mls], axis=0)

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
        strat_share_data['RET_01'] = strat_share_data.groupby('ticker').Close.pct_change(1)
        strat_share_data['RET_01'] = strat_share_data.groupby('ticker')['RET_01'].shift(-1)
        strat_data['RET_01'] = strat_data.groupby('ticker').Close.pct_change(1)
        strat_data['RET_01'] = strat_data.groupby('ticker')['RET_01'].shift(-1)

        # Change short weights to negative
        strat_share_data.loc[strat_share_data['type'] == 'short', 'weight_share'] = strat_share_data.loc[strat_share_data['type'] == 'short', 'weight_share'] * -1
        strat_data.loc[strat_data['type'] == 'short', 'weight'] = strat_data.loc[strat_data['type'] == 'short', 'weight'] * -1

        # Get daily strategy returns
        daily_strat_share_ret = strat_share_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight_share']).sum())
        daily_strat_ret = strat_data.groupby('date').apply(lambda x: (x['RET_01'] * x['weight']).sum())

        # Export qs report
        start_date = (daily_strat_ret.index.min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (daily_strat_ret.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        spy = get_spy(start_date=start_date, end_date=end_date)['spyRet']
        day_zero = daily_strat_share_ret.index[0] - pd.Timedelta(days=1)
        daily_strat_share_ret = pd.concat([pd.Series([1e-4], index=[day_zero]), daily_strat_share_ret])
        daily_strat_ret = pd.concat([pd.Series([1e-4], index=[day_zero]), daily_strat_ret])
        spy = pd.concat([pd.Series([1e-4], index=[day_zero]), spy])
        spy = spy.to_frame('spy_ret')
        qs.reports.html(daily_strat_share_ret, spy, output=self.output_path / 'qs_share_report.html')
        qs.reports.html(daily_strat_ret, spy, output=self.output_path / 'qs_report.html')

        # Export CAPM statistical test
        daily_strat_share_ret_df = daily_strat_share_ret.to_frame()
        daily_strat_df = daily_strat_ret.to_frame()
        self._rolling_full_alpha(strat_ret=daily_strat_share_ret_df, spy=spy, windows=self.alpha_windows, name='alpha_share_report', path=self.output_path)
        self._rolling_full_alpha(strat_ret=daily_strat_df, spy=spy, windows=self.alpha_windows, name='alpha_share', path=self.output_path)

        # Updated CSS for flexbox layout
        html_header = """
        <html>
        <head>
            <style>
                .report-section {
                    border: 3px solid #000;
                    margin: 50px 50px;
                }
                .report-title {
                    background-color: #d4ebf2;
                    padding: 10px;
                    font-size: 20px;
                    font-weight: bold;
                    border-bottom: 2px solid #000;
                }
                .content-container { 
                    display: flex;
                }
                #left, #right {
                    margin: 10px;
                }
                #left { 
                    flex: 2;
                    max-height: 1000px; 
                    overflow: auto; 
                }
                #right { 
                    flex: 1.5; 
                    width: 100%;
                    max-height: 1000px; 
                    overflow: auto;
                    display: flex;
                    flex-direction: column;
                }
            </style>
        </head>
        <body>
        """

        subdirectories = ['strat_all', 'strat_ml_ret', 'strat_ml_trend', 'strat_mrev_etf',
                          'strat_mrev_mkt', 'strat_port_id', 'strat_port_iv', 'strat_port_im',
                          'strat_trend_mls']

        combined_report_content = ""
        combined_share_report_content = ""

        for subdir in subdirectories:
            # Handle qs_report.html
            report_path = os.path.join(get_live_monitor(), subdir, "qs_report.html")
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    # Wrap the existing content inside a div with class 'content-container'
                    left_content = soup.find(id='left')
                    right_content = soup.find(id='right')
                    content = f'<div class="content-container">{str(left_content)}{str(right_content)}</div>'

                    combined_report_content += f'<div class="report-section">'
                    combined_report_content += f'<div class="report-title">{"Strategy" + subdir.replace("strat_", " ").replace("_", " ").title()}</div>\n'
                    combined_report_content += content
                    combined_report_content += '</div>\n'

            # Handle qs_share_report.html
            share_report_path = os.path.join(get_live_monitor(), subdir, "qs_share_report.html")
            if os.path.exists(share_report_path):
                with open(share_report_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    # Wrap the existing content inside a div with class 'content-container'
                    left_content = soup.find(id='left')
                    right_content = soup.find(id='right')
                    content = f'<div class="content-container">{str(left_content)}{str(right_content)}</div>'

                    combined_share_report_content += f'<div class="report-section">'
                    combined_share_report_content += f'<div class="report-title">{"Strategy" + subdir.replace("strat_", " ").replace("_", " ").title()}</div>\n'
                    combined_share_report_content += content
                    combined_share_report_content += '</div>\n'

        combined_report_final = html_header + combined_report_content + "</body></html>"
        combined_share_report_final = html_header + combined_share_report_content + "</body></html>"

        # Write the combined reports to separate HTML files
        combined_report_path = os.path.join(self.output_path, 'all_report.html')
        with open(combined_report_path, 'w', encoding='utf-8') as combined_file:
            combined_file.write(combined_report_final)

        combined_share_report_path = os.path.join(self.output_path, 'all_share_report.html')
        with open(combined_share_report_path, 'w', encoding='utf-8') as combined_share_file:
            combined_share_file.write(combined_share_report_final)