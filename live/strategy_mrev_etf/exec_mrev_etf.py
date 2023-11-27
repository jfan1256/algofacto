import asyncio
import quantstats as qs
import math
import os

from live.callback import OrderCounter
from ib_insync import *
from functions.utils.func import *

# Create multi index for dataframe with only date index
def create_multi_index(factor_data, stock):
    factor_values = pd.concat([factor_data] * len(stock), ignore_index=True).values
    multi_index = pd.MultiIndex.from_product([stock, factor_data.index])
    multi_index_factor = pd.DataFrame(factor_values, columns=factor_data.columns, index=multi_index)
    multi_index_factor.index = multi_index_factor.index.set_names(['permno', 'date'])
    return multi_index_factor

# Create signals for past data
def create_signal_past(data, sbo, sso, sbc, ssc, threshold):
    def apply_rules(group):
        # Initialize signals and positions
        signals = [None] * len(group)
        positions = [None] * len(group)
        # Create masks for conditions
        open_long_condition = (group['s_score'] < -sbo) & (group['market_cap'] > threshold)
        open_short_condition = (group['s_score'] > sso) & (group['market_cap'] > threshold)
        close_long_condition = group['s_score'] > -ssc
        close_short_condition = group['s_score'] < sbc
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


# Create signals for current date data
def create_signal_live(data, sbo, sso, sbc, ssc, threshold, current_date):
    def apply_rules(group):
        if current_date in group.index.get_level_values('date'):
            # Retrieve the current date's s_score and market_cap
            current_s_score = group.iloc[-1]['s_score']
            market_cap = group.iloc[-1]['market_cap']
            # Retrieve the previous date's position
            prev_date_position = group.iloc[-2]['position']
            # Set conditions for signals and positions
            if prev_date_position == 'long':
                if current_s_score > -ssc:
                    signal = 'close long'
                    position = None
                else:
                    signal = 'hold'
                    position = 'long'
            elif prev_date_position == 'short':
                if current_s_score < sbc:
                    signal = 'close short'
                    position = None
                else:
                    signal = 'hold'
                    position = 'short'
            elif prev_date_position == None or prev_date_position == 0:
                if current_s_score < -sbo and market_cap > threshold:
                    signal = 'buy to open'
                    position = 'long'
                elif current_s_score > sso and market_cap > threshold:
                    signal = 'sell to open'
                    position = 'short'
                else:
                    signal = None
                    position = None
            return pd.Series({'signal': signal, 'position': position})

    # Apply the rules to each permno group and get signals and positions for the current date
    signal_positions = data.groupby(level='permno').apply(apply_rules)

    # Assign the new signals and positions to the data for the current date
    for permno in signal_positions.index:
        if pd.notna(signal_positions.loc[permno, 'signal']):
            data.loc[(permno, current_date), 'signal'] = signal_positions.loc[permno, 'signal']
            data.loc[(permno, current_date), 'position'] = signal_positions.loc[permno, 'position']
    return data

# Calculate strategy return across backtest period
def calc_total_ret(df, etf_returns):
    print("Get hedge weights...")
    mask_long = df['position'] == 'long'
    mask_short = df['position'] == 'short'
    df['hedge_weight'] = np.where(mask_long, -1, np.where(mask_short, 1, 0))

    # Get net hedge betas
    print("Get net hedge betas...")
    beta_columns = [col for col in df.columns if '_sector_' in col]
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
        etf_return_column = beta.split('_sector_')[0]
        if etf_return_column in etf_returns.columns:
            net_hedge_returns[beta] = normalized_net_hedge_betas[beta] * etf_returns[etf_return_column]

    # Get total hedge return
    print("Get total hedge return...")
    net_hedge_return_total = net_hedge_returns.sum(axis=1)

    print("Get daily returns...")
    daily_returns = (df['RET_01'] * df['normalized_weight']).groupby('date').sum()

    print("Get total returns...")
    total_returns = daily_returns + net_hedge_return_total

    return total_returns, daily_returns, net_hedge_return_total, normalized_net_hedge_betas, df[['normalized_weight']]

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------EXECUTE PRICES--------------------------------------------------------------------------------
# Execute get last price functionality
async def exec_price(ib, current_date):
    # Get first valid contract
    async def get_contract(symbol):
        contract = Stock(symbol, 'SMART', 'USD')
        contracts = await ib.reqContractDetailsAsync(contract)
        if contracts:
            qualified_contract = contracts[0].contract
            print(f"Obtained qualified contract for {symbol}: {qualified_contract}")
            return qualified_contract
        else:
            print(f"No qualified contract found for {symbol}")
            return None

    # Get the last closing price of a stock
    async def get_market_data(stock):
        print("-" * 60)
        MAX_RETRIES = 10
        SLEEP_DURATION = 5

        for _ in range(MAX_RETRIES):
            market_data = ib.reqMktData(stock, '', False, False)
            await asyncio.sleep(SLEEP_DURATION)
            if market_data.last:
                print(f"Obtained {stock.symbol} last price")
                print("-" * 60)
                ib.cancelMktData(stock)  # Cancel market data request
                return market_data

        print(f"Failed to get market data for {stock.symbol} after {MAX_RETRIES} consecutive calls.")
        print("-" * 60)
        ib.cancelMktData(stock)  # Cancel market data request
        return None

    async def get_last_price(symbol):
        contract = await get_contract(symbol)
        if not contract:
            return symbol, None
        market_data = await get_market_data(contract)
        if market_data and market_data.last:
            return symbol, market_data.last
        return symbol, None

    # Split the all_stocks list into chunks of batch_size
    async def get_prices_in_batches(all_stocks, batch_size):
        batches = [all_stocks[i:i + batch_size] for i in range(0, len(all_stocks), batch_size)]

        symbol_price_tuples = []
        for batch in batches:
            tasks = [get_last_price(stock_symbol) for stock_symbol in batch]
            batch_results = await asyncio.gather(*tasks)
            # Filter and extend the main list
            symbol_price_tuples.extend([t for t in batch_results if t[1] is not None])
            await asyncio.sleep(1)  # Sleep to avoid hitting rate limits, if necessary
        return symbol_price_tuples


    # Get stocks
    live = True
    past_data = pd.read_parquet(get_parquet_dir(live) / 'data_price.parquet.brotli', columns=['Close'])
    ticker = pd.read_parquet(get_parquet_dir(live) / 'data_ticker.parquet.brotli')
    # Most up to "date" piece of data
    etf_ticker = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']
    past_date = past_data.index.get_level_values('date').max().strftime('%Y-%m-%d')
    ticker_price = past_data.loc[past_data.index.get_level_values('date') == past_date]
    ticker_price = ticker_price.merge(ticker, left_index=True, right_index=True, how='left')
    trade_ticker = ticker_price.ticker.tolist()
    all_stocks = trade_ticker + etf_ticker

    # Get stock prices in batches (or else it will hit ticker request rate limit
    batch_size = 100
    symbol_price_tuples = await get_prices_in_batches(all_stocks, batch_size)

    # Create DataFrame
    price_all = pd.DataFrame(symbol_price_tuples, columns=['ticker', 'Close'])
    price_all['date'] = pd.to_datetime(current_date)

    # Separate price data into ETF and Stocks to trade
    etf_data = price_all[price_all['ticker'].isin(etf_ticker)].set_index(['ticker', 'date'])
    trade_data = price_all[price_all['ticker'].isin(trade_ticker)]
    price_data = trade_data.copy(deep=True).set_index(['ticker', 'date'])


    # Add permno to trade_data and set_index
    permno_to_ticker_dict = ticker_price.reset_index(level='date')['ticker'].to_dict()
    ticker_to_permno_dict = {v: k for k, v in permno_to_ticker_dict.items()}
    trade_data['permno'] = trade_data['ticker'].map(ticker_to_permno_dict)
    trade_data = trade_data.set_index(['permno', 'date']).drop('ticker', axis=1)

    # Get Adjustment Factor and adjust live price data
    adj_factor_trade = get_adj_factor(trade_ticker, past_date)
    trade_data['adj_factor'] = adj_factor_trade['adj_factor_trade']
    trade_data['Close'] = trade_data['Close'] / trade_data['adj_factor']

    # Add live trade price to the end of the historical price dataset and calculate returns
    T = [1]
    live_data = pd.concat([past_data, trade_data], axis=0).sort_index(level=['permno', 'date'])
    live_data = create_return(live_data, T)
    live_data = live_data.fillna(0)
    live_data = live_data.drop(['Close'], axis=1)

    # Add live etf price to the end of the historical price dataset
    past_etf = pd.read_parquet(get_parquet_dir(live) / 'data_etf.parquet.brotli', columns=['Close'])

    # Get Adjustment Factor and adjust live price data
    adj_factor_etf = get_adj_factor(etf_ticker, past_date)
    etf_data['adj_factor'] = adj_factor_etf['adj_factor_trade']
    etf_data['Close'] = etf_data['Close'] / etf_data['adj_factor']

    # Add live trade price to the end of the historical price dataset
    live_etf = pd.concat([past_etf, etf_data], axis=0).sort_index(level=['ticker', 'date'])
    price_etf = live_etf.copy(deep=True)[['Close']]

    # Create returns and unstack dataframe to only have 'date' index and 'ETF ticker' columns
    live_etf = create_return(live_etf, T)
    live_etf = live_etf.drop(['Close'], axis=1)
    live_etf = live_etf.unstack('ticker').swaplevel(axis=1)
    live_etf.columns = ['_'.join(col).strip() for col in live_etf.columns.values]
    live_etf = live_etf.fillna(0)

    return live_data, live_etf, price_data, price_etf

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------EXECUTE GET DATA-----------------------------------------------------------------------------
# Execute get data
def exec_mrev_etf_data(window, threshold):
    # Params
    live = True
    etf_list = ['QQQ', 'XLE', 'XLV', 'VNQ', 'XLB', 'XLF', 'XLY', 'XLI', 'XLI', 'XLU', 'XLP']
    current_date = date.today().strftime('%Y-%m-%d')
    start = '2005-01-01'
    sbo = 0.85
    sso = 0.85
    sbc = 0.25
    ssc = 0.25

    # Read in ETF data from FMP
    T = [1]
    sector_ret_past = get_data_fmp(ticker_list=etf_list, start=start, current_date=current_date)
    sector_ret_past = sector_ret_past[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
    sector_ret_past = sector_ret_past.rename(columns={'Adj Close': 'Close'})
    sector_ret_past = sector_ret_past.loc[~sector_ret_past.index.duplicated(keep='first')]
    # Create returns and unstack dataframe to only have 'date' index and 'ETF ticker' columns
    sector_ret_past = create_return(sector_ret_past, T)
    sector_ret_past = sector_ret_past.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
    sector_ret_past = sector_ret_past.unstack('ticker').swaplevel(axis=1)
    sector_ret_past.columns = ['_'.join(col).strip() for col in sector_ret_past.columns.values]
    sector_ret_past = sector_ret_past.fillna(0)

    # Read in price data and set up params for Rolling LR
    T = [1]
    ret = f'RET_01'
    past_data = pd.read_parquet(get_parquet_dir(live) / 'data_price.parquet.brotli')
    factor_col_past = sector_ret_past.columns
    past_data = create_return(past_data, T)
    past_data = past_data.fillna(0)

    # Execute Rolling LR
    beta_data_past = rolling_ols_parallel(data=past_data, ret=ret, factor_data=sector_ret_past, factor_cols=factor_col_past.tolist(), window=window, name=f'sector_01')

    # Retrieve Needed Data
    beta_data_past = beta_data_past[beta_data_past.columns[1:14]]
    beta_data_past = beta_data_past.fillna(0)

    # Calculate rolling mean and standard deviation
    rolling_mean = beta_data_past[f'epsil_sector_01_{window:02}'].rolling(window=window).mean()
    rolling_std = beta_data_past[f'epsil_sector_01_{window:02}'].rolling(window=window).std()
    # Calculate the rolling Z-score
    beta_data_past['s_score'] = (beta_data_past[f'epsil_sector_01_{window:02}'] - rolling_mean) / rolling_std

    # Export data
    beta_data_past.to_parquet(get_strategy_mrev_etf_data() / 'data_beta_etf.parquet.brotli', compression='brotli')

    # Convert ETF Dataframe to multi-index
    stock = read_stock(get_large_dir(live) / 'permno_live.csv')
    sector_multi_past = create_multi_index(sector_ret_past, stock)
    # Merge the necessary columns together into one dataframe
    combined_past = beta_data_past.merge(sector_multi_past, left_index=True, right_index=True, how='left')
    combined_past = combined_past.merge(past_data[['RET_01']], left_index=True, right_index=True, how='left')
    combined_past = combined_past.fillna(0)

    # Retrieve the needed columns
    ret_columns = [col for col in combined_past.columns if "RET_01" in col]
    combined_past = combined_past[['s_score'] + ret_columns]

    # Create signals
    copy = combined_past.copy(deep=True)
    misc = pd.read_parquet(get_parquet_dir(live) / 'data_misc.parquet.brotli', columns=['market_cap'])
    copy = copy.merge(misc, left_index=True, right_index=True, how='left')
    result_past = create_signal_past(copy, sbo, sso, sbc, ssc, threshold)

    result_past_copy = result_past.copy(deep=True)
    # Shift returns back for accurate backtest
    result_past_copy['RET_01'] = result_past_copy.groupby('permno')['RET_01'].shift(-1)
    # Calculate strategy's return
    ewp_ret, daily_ret, hedge_ret, beta_weight, stock_weight = calc_total_ret(result_past_copy, sector_ret_past.shift(-1))

    # Save backtest plot
    spy = get_spy(start_date='2005-01-01', end_date=current_date)
    format_date =  current_date.replace('-', '')
    qs.reports.html(ewp_ret, spy, output=get_strategy_mrev_etf() / 'report' / f'mrev_etf_{format_date}.html')

    # Export data
    result_past_copy[['signal', 'position', 'market_cap']].to_parquet(get_strategy_mrev_etf_data() / 'data_signal_etf.parquet.brotli', compression='brotli')
    stock_weight.to_parquet(get_strategy_mrev_etf_data() / 'data_stock_etf.parquet.brotli', compression='brotli')
    beta_weight.to_parquet(get_strategy_mrev_etf_data() / 'data_hedge_etf.parquet.brotli', compression='brotli')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------EXECUTE GET LIVE WEIGHTS------------------------------------------------------------------------
# Execute get live weights
def exec_mrev(live_data, sector_ret_live, live, window, sbo, sso, sbc, ssc, threshold, stock, current_date):
    # Function that executes the RollingOLS for the last window of data (this function will be parallelized)
    def per_stock_ols_last(stock_data, ret, factor_data, factor_cols, window, stock_name, index_name):
        # Set up data
        model_data = stock_data[[ret]].merge(factor_data, on='date').dropna()
        model_data = model_data[[ret] + factor_cols]
        model_data = model_data.iloc[-window:]
        # Train model
        exog = sm.add_constant(model_data[factor_cols])
        model = OLS(model_data[ret], exog)
        results = model.fit()
        # Get betas
        factor_model_params = results.params.to_frame().T
        factor_model_params.rename(columns={'const': 'ALPHA'}, inplace=True)
        # Get predictions
        predicted_last = (exog.iloc[-1] * factor_model_params.iloc[-1]).sum()
        predicted_series = pd.Series(predicted_last, name='pred')
        # Compute residuals (epsilon) for the last observation
        epsilon_last = model_data[ret].iloc[-1] - predicted_last
        epsilon_series = pd.Series(epsilon_last, name='epsil')
        # Format data
        result = factor_model_params.assign(epsil=epsilon_series, pred=predicted_series)
        result = result.assign(**{index_name: stock_name}).set_index(index_name, append=True).swaplevel()
        permnos = result.index.get_level_values(0)
        new_index = pd.MultiIndex.from_product([permnos, [pd.to_datetime(current_date)]], names=['permno', 'date'])
        result.index = new_index
        return result

    # Function to execute the parallelization for last window of data
    def rolling_ols_last(data, ret, factor_data, factor_cols, window, name):
        valid_groups = [(name, group) for name, group in data.groupby(level='permno') if current_data(group, current_date, window)]
        print(len(valid_groups))
        tasks = [(group, ret, factor_data, factor_cols, window, permno, data.index.names[0]) for permno, group in valid_groups]
        results = Parallel(n_jobs=-1)(delayed(per_stock_ols_last)(*task) for task in tasks)
        return pd.concat(results).rename(columns=lambda x: f'{x}_{name}_{window:02}')

    # Calculate weight for stocks and etfs for current date data
    def calc_total_weight(df):
        print("Get hedge weights...")
        mask_long = df['position'] == 'long'
        mask_short = df['position'] == 'short'
        df['hedge_weight'] = np.where(mask_long, -1, np.where(mask_short, 1, 0))

        # Get net hedge betas
        print("Get net hedge betas...")
        beta_columns = [col for col in df.columns if '_sector_' in col]
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

        return normalized_net_hedge_betas, df[['normalized_weight']]

    # Display long and short stocks
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

    # Update csv data at given date index
    def update_csv_for_date(filename, df_combined, current_date):
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


    # Execute Rolling LR
    ret = f'RET_01'
    factor_col_live = sector_ret_live.columns
    beta_data_live = rolling_ols_last(data=live_data, ret=ret, factor_data=sector_ret_live, factor_cols=factor_col_live.tolist(), window=window, name=f'sector_01')
    beta_data_live = beta_data_live[beta_data_live.columns[1:14]]

    # Combine live beta with historical beta
    beta_data_past = pd.read_parquet(get_strategy_mrev_etf_data() / 'data_beta_etf.parquet.brotli')
    beta_data_live = pd.concat([beta_data_past, beta_data_live], axis=0).sort_index(level=['permno', 'date'])
    beta_data_live = beta_data_live.fillna(0)

    # Calculate rolling mean and standard deviation
    rolling_mean = beta_data_live[f'epsil_sector_01_{window:02}'].rolling(window=window).mean()
    rolling_std = beta_data_live[f'epsil_sector_01_{window:02}'].rolling(window=window).std()

    # Calculate the rolling Z-score
    beta_data_live['s_score'] = (beta_data_live[f'epsil_sector_01_{window:02}'] - rolling_mean) / rolling_std

    # Convert ETF Dataframe to multi-index
    sector_multi_live = create_multi_index(sector_ret_live, stock)
    # Merge the necessary columns together into one dataframe
    combined_live = beta_data_live.merge(sector_multi_live, left_index=True, right_index=True, how='left')
    combined_live = combined_live.merge(live_data[['RET_01']], left_index=True, right_index=True, how='left')
    signal_past = pd.read_parquet(get_strategy_mrev_etf_data() / 'data_signal_etf.parquet.brotli')
    combined_live = combined_live.merge(signal_past[['signal', 'position']], left_index=True, right_index=True, how='left')
    combined_live = combined_live.fillna(0)

    # Retrieve the needed columns
    ret_columns = [col for col in combined_live.columns if "RET_01" in col]
    combined_live = combined_live[['s_score', 'signal', 'position'] + ret_columns]

    # Create signals
    combined_live_copy = combined_live.copy(deep=True)
    misc = pd.read_parquet(get_parquet_dir(live) / 'data_misc.parquet.brotli', columns=['market_cap'])
    combined_live_copy = combined_live_copy.merge(misc, left_index=True, right_index=True, how='left')
    result_live = create_signal_live(combined_live_copy, sbo, sso, sbc, ssc, threshold)
    result_live['RET_01'] = result_live.groupby('permno')['RET_01'].shift(-1)

    result_live_curr = result_live.loc[result_live.index.get_level_values('date') == current_date]
    beta_weight, stock_weight = calc_total_weight(result_live_curr)

    # Get current date dataframe
    trade = result_live.loc[result_live.index.get_level_values('date') == current_date]
    trade = trade.fillna(0)
    trade = trade.merge(stock_weight, left_index=True, right_index=True, how='left')

    # Get weight to invest into each ETF for current date
    beta_columns = [col for col in beta_data_live.columns if '_sector_' in col]
    beta_weight.columns = [(col.split('_')[0]) for col in beta_columns]

    # Read in ticker dataframe
    ticker = pd.read_parquet(get_parquet_dir(live) / 'data_ticker.parquet.brotli')

    # Get long positions
    long = trade.loc[trade['position'].str.contains('long', na=False)]
    long = long.merge(ticker, left_index=True, right_index=True, how='left')
    long_ticker = list(zip(long['ticker'], long['normalized_weight']))

    # Get short positions
    short = trade.loc[trade['signal'].str.contains('short', na=False)]
    short = short.merge(ticker, left_index=True, right_index=True, how='left')
    short_ticker = list(zip(short['ticker'], short['normalized_weight']))

    # Display stocks to long and short tomorrow
    content = display_stock(long_ticker, "Stocks to Long Today:")
    content += '\n\n' + display_stock(short_ticker, "Stocks to Short Today:")
    print(content)

    # Append long/short stocks to dataframe and export
    all_columns = ['date'] + [f'Long_{i:02}' for i in range(1, len(long) + 1)] + [f'Short_{i:02}' for i in range(1, len(short) + 1)]
    stock_data = [current_date] + long_ticker + short_ticker
    df_combined_stock = pd.DataFrame([stock_data], columns=all_columns)
    filename_stock = Path(get_strategy_mrev_etf() / f'trade_stock_mrev_etf.csv')

    etf_ticker = beta_weight.columns.tolist()
    etf_columns = ['date'] + [f'ETF_{i:02}' for i in range(1, len(etf_ticker) + 1)]
    etf_data = [current_date] + etf_ticker
    df_combined_etf = pd.DataFrame([etf_data], columns=etf_columns)
    filename_etf = Path(get_strategy_mrev_etf() / f'trade_etf_mrev_etf.csv')

    # Assuming df_combined_stock and df_combined_etf are your dataframes for stock and ETF respectively
    # and filename_stock and filename_etf are the corresponding filenames for stock and ETF data
    update_csv_for_date(filename_stock, df_combined_stock, current_date)
    update_csv_for_date(filename_etf, df_combined_etf, current_date)

    return long_ticker, short_ticker, beta_weight


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
# Execute trades
def exec_trade(ib, long_stock, short_stock, etf_weight, price_data, price_etf, settlement, capital):
    # Execute trades
    def create_moc_order(action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MOC"
        order.totalQuantity = quantity
        order.transmit = True
        return order

    # Callback to see if order has been filled
    def order_filled(trade, fill):
        print(f"Order has been filled for {trade.contract.symbol}")
        print(trade.order)
        print(fill)

    # Function to get a specific contract
    def get_contract(symbol):
        contracts = ib.qualifyContracts(Stock(symbol, 'SMART', 'USD'))
        if contracts:
            return contracts[0]
        else:
            print(f"Could not find a unique contract for {symbol}")
            return None

    # Fetch available capital
    print("Fetching available capital...")
    account_info = ib.accountValues()
    for item in account_info:
        if item.tag == 'NetLiquidation':
            total_capital = float(item.value)
            print(f"Available capital: ${total_capital}")
            break
    else:
        print("Could not fetch available capital. Exiting...")
        ib.disconnect()
        exit()

    available_capital = total_capital * capital
    available_capital_settlement = available_capital / settlement
    etf_stock = etf_weight.columns.tolist()

    order_num = 1
    # Buy long positions
    for stock_symbol in long_stock:
        print("-" * 60)
        capital_per_stock = available_capital_settlement * stock_symbol[1]
        stock_price = price_data.loc[price_data.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares

        stock = get_contract(stock_symbol[0])
        print(f"Buying LONG position for: {stock_symbol[0]}")
        action = 'BUY'
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {stock_symbol[0]}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Sell short positions
    for stock_symbol in short_stock:
        print("-" * 60)
        capital_per_stock = available_capital_settlement * stock_symbol[1]
        stock_price = price_data.loc[price_data.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares

        stock = get_contract(stock_symbol[0])
        print(f"Selling SHORT position for: {stock_symbol[0]}")
        action = 'SELL'
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {stock_symbol[0]}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Sell short positions
    for stock_symbol in short_stock:
        print("-" * 60)
        capital_per_stock = available_capital_settlement * stock_symbol[1]
        stock_price = price_data.loc[price_data.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares

        stock = get_contract(stock_symbol[0])
        print(f"Selling SHORT position for: {stock_symbol[0]}")
        action = 'SELL'
        moc_order = create_moc_order(action, num_share)
        print(f"Placing MOC order to {action}: {num_share} of {stock_symbol[0]}")
        trade_moc = ib.placeOrder(stock, moc_order)
        trade_moc.fillEvent += order_filled
        print(f"Order Number: {order_num}")
        order_num += 1

    # Buy/Sell ETF positions
    for stock_symbol in etf_stock:
        print("-" * 60)
        weight = etf_weight[stock_symbol][0]
        capital_per_stock = available_capital_settlement * abs(weight)
        stock_price = price_etf.loc[price_etf.index.get_level_values('ticker') == stock_symbol[0]]['Close'][0]
        num_share = int(capital_per_stock / stock_price)  # This will provide the number of whole shares
        stock = get_contract(stock_symbol[0])

        if weight >= 0:
            # Buy position
            print(f"Buying LONG position for: {stock_symbol}")
            action = 'BUY'
            moc_order = create_moc_order(action, num_share)
            print(f"Placing MOC order to {action}: {num_share} of {stock_symbol}")
            trade_moc = ib.placeOrder(stock, moc_order)
            trade_moc.fillEvent += order_filled
            print(f"Order Number: {order_num}")
            order_num += 1
        else:
            # Short position
            print(f"Selling SHORT position for: {stock_symbol}")
            action = 'SELL'
            moc_order = create_moc_order(action, num_share)
            print(f"Placing MOC order to {action}: {num_share} of {stock_symbol}")
            trade_moc = ib.placeOrder(stock, moc_order)
            trade_moc.fillEvent += order_filled
            print(f"Order Number: {order_num}")
            order_num += 1


def exec_mrev_etf_trade(window, threshold, settlement, capital):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------CONNECT------------------------------------------------------------------------------------
    print("-------------------------------------------------------------------------------CONNECT------------------------------------------------------------------------------------")
    # Connect to IB
    print("Attempting to connect to IBKR TWS Workstation...")
    ib = IB()
    ib.connect(host='127.0.0.1', port=7497, clientId=1512)
    print("Connected to IBKR TWS Workstation.")

    # Params
    live = True
    stock = read_stock(get_large_dir(live) / 'permno_live.csv')
    current_date = date.today().strftime('%Y-%m-%d')
    sbo = 0.85
    sso = 0.85
    sbc = 0.25
    ssc = 0.25

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------EXECUTE PRICES--------------------------------------------------------------------------------
    print("----------------------------------------------------------------------------EXECUTE PRICES--------------------------------------------------------------------------------")
    # Create an event loop
    loop = asyncio.get_event_loop()
    # Retrieve live prices
    live_data, live_etf, price_data, price_etf = loop.run_until_complete(exec_price(ib=ib, current_date=current_date))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------EXECUTE MEAN REVERSION----------------------------------------------------------------------------
    print("------------------------------------------------------------------------EXECUTE MEAN REVERSION----------------------------------------------------------------------------")
    long_stock, short_stock, beta_weight = exec_mrev(live_data=live_data, sector_ret_live=live_etf, live=live, window=window, sbo=sbo, sso=sso, sbc=sbc, ssc=ssc, threshold=threshold, stock=stock, current_date=current_date)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------
    print("-------------------------------------------------------------------------EXECUTE TRADE ORDERS-----------------------------------------------------------------------------")
    # Subscribe the class method to the newOrderEvent
    order_counter = OrderCounter()
    ib.newOrderEvent += order_counter.new_order_event_handler
    exec_trade(ib=ib,long_stock=long_stock, short_stock=short_stock, etf_weight=beta_weight, price_data=price_data, price_etf=price_etf, settlement=settlement, capital=capital)
    print(f"----------------------------------------------------Total number of new orders placed: {order_counter.new_order_count}---------------------------------------------------")
    # Close the loop
    loop.close()
    ib.disconnect()

exec_mrev_etf_data(window=168, threshold=2_000_000_000)
