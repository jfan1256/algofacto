import shutil

import pandas as pd
import quantstats as qs
import os

from scipy.stats import spearmanr
from live.strategy_ml.live_test import LiveTest
from functions.utils.func import *

def exec_pred(num_stocks, leverage, port_opt, use_model):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------PARAMS------------------------------------------------------------------------------------
    live = True
    current_date = date.today().strftime('%Y-%m-%d')
    model_name = f"lightgbm_{date.today().strftime('%Y%m%d')}"
    dir_path = Path(get_report(live) / model_name)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------INITIATE LIVE TEST------------------------------------------------------------------------------
    print("--------------------------------------------------------------------------INITIATE LIVE TEST------------------------------------------------------------------------------")
    live_test = LiveTest(live=live, num_stocks=num_stocks, leverage=leverage, port_opt=port_opt, model_name=model_name, current_date=current_date, dir_path=dir_path)
    files = live_test.read_result('metrics')

    # Create directory for backtest report
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------CALCULATE SHARPE PER TRAIL--------------------------------------------------------------------------
    print("----------------------------------------------------------------------CALCULATE SHARPE PER TRAIL--------------------------------------------------------------------------")
    # Dictionary to keep track of SHARPE
    keep = {}
    ticker = pd.read_parquet(get_parquet_dir(live) / 'data_ticker.parquet.brotli')

    # Iterate through each trial
    for i, row in files.iterrows():
        # Read file in
        read_file = live_test.get_max_ic_file(row)
        # Execute ranking of stocks
        returns = live_test.sharpe_ret(read_file, iteration=False)
        # Convert Permno to Ticker
        tic = returns.merge(ticker, left_index=True, right_index=True, how='left')
        tic = tic.reset_index().set_index(['window', 'ticker', 'date'])
        tic = tic.drop('permno', axis=1)
        # Calculate SHARPE with EWP
        pred = live_test.sharpe_backtest(tic)
        equal_weight = live_test.exec_port_opt(data=pred, option='both')
        stock = equal_weight['totalRet']
        sharpe = qs.stats.sharpe(stock)
        # Display metrics
        print('-' * 60)
        print(f'Row: {i}')
        metrics = read_file['metrics']
        column_widths = [max(len(str(val)) for val in metrics[col]) for col in metrics.columns]
        header_widths = [len(header) for header in metrics.columns]
        max_widths = [max(col_width, header_width) for col_width, header_width in zip(column_widths, header_widths)]
        headers = " | ".join([header.ljust(width) for header, width in zip(metrics.columns, max_widths)])
        values = " | ".join([str(val).ljust(width) for val, width in zip(metrics.iloc[0], max_widths)])
        formatted_metrics = headers + "\n" + values
        print(formatted_metrics)
        print(f'SHARPE Ratio: {sharpe}')
        # Save SHARPE to dictionary
        keep[i] = sharpe

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------PERFORM ENSEMBLE PREDICTION AND SAVE METRIC/IC---------------------------------------------------------------
    print("-------------------------------------------------------------PERFORM ENSEMBLE PREDICTION AND SAVE METRIC/IC---------------------------------------------------------------")
    # Retrieves the indices from the top 5 best performing SHARPE
    max_sharpe_idxs = sorted(keep, key=keep.get, reverse=True)[:use_model]
    collect = []
    # Append the individual trial predictions to a dataframe
    for idx in max_sharpe_idxs:
        print(f'Best Sharpe Idx: {idx}')
        best_model_params = live_test.get_max_ic_file(files.iloc[idx])
        merged = live_test.price(best_model_params, dir_path, iteration=False, plot=False)
        collect.append(merged['predictions'])

    # Concat and calculate the mean of the predictions
    total = pd.concat(collect, axis=1)
    total['mean_predictions'] = total.mean(axis=1)
    merged['predictions'] = total['mean_predictions']
    # Calculate IC
    by_day = merged.groupby(level='date')
    ic_by_day = by_day.apply(lambda x: spearmanr(x.predictions, x.returns)[0]).to_frame('combined')
    # Save Plot
    live_test.plot_ensemble(merged, ic_by_day)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------CONVERT FROM PERMNO TO TICKER/EXCHANGE-------------------------------------------------------------------
    print("-----------------------------------------------------------------CONVERT FROM PERMNO TO TICKER/EXCHANGE-------------------------------------------------------------------")
    ticker = pd.read_parquet(get_parquet_dir(live) / 'data_ticker.parquet.brotli')
    tic = merged.merge(ticker, left_index=True, right_index=True, how='left')
    tic = tic.reset_index().set_index(['window', 'ticker', 'date'])
    exchange = pd.read_parquet(get_parquet_dir(live) / 'data_exchange.parquet.brotli')
    tic_reset = tic.reset_index()
    exchange_df_reset = exchange.reset_index()
    combined = pd.merge(tic_reset, exchange_df_reset, on=['ticker', 'permno'], how='left')
    combined = combined.set_index(['window', 'ticker', 'date'])

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------EXECUTE LIVETEST------------------------------------------------------------------------------
    print("----------------------------------------------------------------------------EXECUTE LIVETEST------------------------------------------------------------------------------")
    # Create the desired dataframe with structure longRet, longStocks, shortRet, shortStocks
    pred_return = live_test.backtest(combined)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------RETRIEVE LONG/SHORT----------------------------------------------------------------------------
    print("---------------------------------------------------------------------------RETRIEVE LONG/SHORT----------------------------------------------------------------------------")
    data = pred_return.copy(deep=True)
    pred_return_opt = live_test.exec_port_opt(data=data, option='both')
    stock = pred_return_opt['totalRet']
    # Save plot to "report" directory
    qs.reports.html(stock, 'SPY', output=dir_path / 'report.html')

    # Retrieve stocks to long/short tomorrow
    long = pred_return.iloc[-1]['longStocks']
    short = pred_return.iloc[-1]['shortStocks']
    # Display stocks to long and short tomorrow
    content = live_test.display_stock(long, "Stocks to Long Tomorrow:")
    content += '\n\n' + live_test.display_stock(short, "Stocks to Short Tomorrow:")
    print(content)

    # Append long/short stocks to dataframe and export
    all_columns = ['date'] + [f'Long_{i:02}' for i in range(1, len(long) + 1)] + [f'Short_{i:02}' for i in range(1, len(short) + 1)]
    combined_data = [current_date] + long + short
    df_combined = pd.DataFrame([combined_data], columns=all_columns)
    filename = Path(get_strategy_ml() / f'trade_stock_{num_stocks}.csv')

    # Check if file exists
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        # Check if the current_date already exists in the existing_df
        if current_date in existing_df['date'].values:
            existing_df = existing_df[existing_df['date'] != current_date]
        updated_df = pd.concat([existing_df, df_combined], ignore_index=True)
        updated_df.to_csv(filename, index=False)
    else:
        df_combined.to_csv(filename, index=False)

# exec_pred(num_stocks=50, leverage=0.5, port_opt='equal_weight', use_model=5)
