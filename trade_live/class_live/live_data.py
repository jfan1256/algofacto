from datetime import datetime
from fredapi import Fred

from core.operation import *
from core.system import *

import wrds
import json
import warnings

warnings.filterwarnings('ignore')

class LiveData:
    def __init__(self,
                 live: bool = None,
                 start_date: str = None,
                 current_date: str = None):

        '''
        live (bool): Get live data or historical data
        start_date (str: YYYY-MM-DD): Start date for data retrieval
        current_date (str: YYYY-MM-DD): Current date (this will be the last date for data retrieval)
        '''

        self.live = live
        self.start_date = start_date
        self.current_date = current_date

        with open(get_config() / 'api_key.json') as f:
            config = json.load(f)
            wrd_key = config['wrd_key']
            fred_key = config['fred_key']
            
        self.wrd_key = wrd_key
        self.fred_key = fred_key


    # Create Link Table
    def create_link_table(self):
        print("-" * 60)
        sql_link = f"""
            SELECT a.gvkey, a.conm, a.tic, a.cusip, a.cik, a.sic, a.naics, b.linkprim,
                   b.linktype, b.liid, b.lpermno, b.lpermco, b.linkdt, b.linkenddt
            FROM comp_na_daily_all.names as a
            INNER JOIN crsp.ccmxpf_lnkhist as b
            ON a.gvkey = b.gvkey
            WHERE b.linktype in ('LC', 'LU')
            AND b.linkprim in ('P', 'C')
            ORDER BY a.gvkey;
        """

        # Read in linking table
        print("Read in linking table...")
        db = wrds.Connection(wrds_username=self.wrd_key)
        link_table = db.raw_sql(sql_link)
        db.close()

        # Rename columns
        print("Rename columns...")
        link_table = link_table.rename(columns={
            'linkdt': 'timeLinkStart_d',
            'linkenddt': 'timeLinkEnd_d',
            'lpermno': 'permno',
            'tic': 'ticker'
        })

        # Convert linkdates to datetime
        print("Convert linkdates to datetime...")
        link_table['timeLinkStart_d'] = pd.to_datetime(link_table['timeLinkStart_d']).replace(pd.NaT, self.current_date)
        link_table['timeLinkEnd_d'] = pd.to_datetime(link_table['timeLinkEnd_d']).replace(pd.NaT, self.current_date)
        link_table['permno'] = link_table['permno'].astype(int)

        # Export link table
        print("Export link table...")
        link_table.to_parquet(get_parquet(self.live) / 'data_link.parquet.brotli', compression='brotli')


    # Create CRSP Price
    def create_crsp_price(self, threshold, set_length):
        print("-" * 60)
        # Read in CRSP dataset
        print('Read in CRSP dataset')
        crsp = pd.read_csv(get_large(self.live) / 'crsp_price.csv')

        # Rename Columns
        print('Rename columns...')
        crsp.columns = crsp.columns.str.lower()
        crsp = crsp.rename(columns={'prc': 'Close', 'bid': 'High', 'ask': 'Low', 'openprc': 'Open', 'shrout': 'outstanding', 'vol': 'Volume', 'cfacpr': 'adj_price'})

        # Adjust closing price
        print('Adjusting Close Price...')
        crsp['Close'] = crsp['Close'] / crsp['adj_price']

        # Set and sort index
        print('Set and sort indices...')
        crsp.date = pd.to_datetime(crsp.date)
        crsp = crsp.set_index(['permno', 'date'])
        crsp = crsp.sort_index(level=['permno', 'date'])

        # Remove duplicate indices and replace all infinity with nan
        print('Remove duplicate indices and infinity...')
        crsp = crsp[~crsp.index.duplicated(keep='first')]
        crsp = crsp.replace([np.inf, -np.inf], np.nan)

        # Remove stocks that have more than 1 NAN values in their Closing price column
        # Stocks that get delisted have 1 row of NAN values as their last row
        # Stocks that switch ticker (WM to COOP: 81593) have rows of NAN valuescap = cap.dropna(subset='Close')
        # Afterwards, drop all rows that have NAN values in Close (every delisted permno stock only has 1 NAN in Close now)
        print('Remove stocks with NAN...')
        nan_counts = crsp.groupby('permno')['Close'].apply(lambda x: x.isna().sum())
        valid_permnos = nan_counts[nan_counts <= 1].index.tolist()
        crsp = crsp[crsp.index.get_level_values('permno').isin(valid_permnos)]
        crsp = crsp.dropna(subset='Close')

        # Remove dates in stocks that have a negative closing price
        crsp = crsp[crsp['Close'] >= 0]

        # Remove stocks that do not have at least 3 years worth of year data
        print(f'Set length to {set_length} years...')
        crsp = set_length(crsp, set_length)

        # Drop permno that do not have over _B market cap
        print(f"Drop permnos that do not have over {threshold}B market cap...")
        crsp['market_cap'] = crsp['Close'] * crsp['outstanding'] * 1000
        avg_cap = crsp.groupby('permno')['market_cap'].mean()
        above_cap = avg_cap[avg_cap > threshold].index
        crsp = crsp[crsp.index.get_level_values('permno').isin(above_cap)]

        # Drop permnos that have the same ticker on the last date of the dataframe
        print('Export ticker and filter out any permnos that share tickers on the last date...')
        last_date = crsp.index.get_level_values('date').max()
        filtered_df = crsp[crsp.index.get_level_values('date') == last_date]
        duplicated_tickers = filtered_df[filtered_df['ticker'].duplicated(keep=False)]['ticker'].unique()
        permnos_to_drop = filtered_df[filtered_df['ticker'].isin(duplicated_tickers)].index.get_level_values('permno').unique()
        crsp = crsp[~crsp.index.get_level_values('permno').isin(permnos_to_drop)]
        ticker = crsp[['ticker']]
        ticker.to_parquet(get_parquet(self.live) / 'data_crsp_ticker.parquet.brotli', compression='brotli')

        # Export ohclv
        print('Export ohclv...')
        ohclv = crsp[['Open', 'High', 'Low', 'Close', 'Volume']]
        ohclv = ohclv.astype(float)
        ohclv.to_parquet(get_parquet(self.live) / 'data_crsp_price.parquet.brotli', compression='brotli')

        # Set up Exchange Mapping
        print("Set up Exchange Mapping...")
        exchange_mapping = {
            -2: "Halted NYSE/AMEX",
            -1: "Suspended NYSE/AMEX/NASDAQ",
            0: "Not Trading NYSE/AMEX/NASDAQ",
            1: "NYSE",
            2: "AMEX",
            3: "NASDAQ",
            4: "Arca",
            5: "Mutual Funds NASDAQ",
            10: "Boston Stock Exchange",
            13: "Chicago Stock Exchange",
            16: "Pacific Stock Exchange",
            17: "Philadelphia Stock Exchange",
            19: "Toronto Stock Exchange",
            20: "OTC Non-NASDAQ",
            31: "When-issued NYSE",
            32: "When-issued AMEX",
            33: "When-issued NASDAQ"
        }
        crsp['exchcd'] = crsp['exchcd'].map(exchange_mapping)
        exchange_copy = crsp.copy(deep=True)
        exchange_copy = exchange_copy.sort_values(by=['permno', 'date', 'ticker'])

        # Convert to permno/ticker multindex and export data
        print("Convert to permno/ticker multindex and export data...")
        exchange = exchange_copy.groupby(['permno', 'ticker'])['exchcd'].last()
        exchange = exchange.reset_index().rename(columns={'exchcd': 'exchange'}).set_index(['permno', 'ticker'])
        exchange.to_parquet(get_parquet(self.live) / 'data_exchange.parquet.brotli', compression='brotli')

    def create_compustat_quarterly(self):
        print("-" * 60)
        sql_compustat_quarterly = f"""
            SELECT a.gvkey, a.datadate, a.fyearq, a.fqtr, a.datacqtr, a.datafqtr, a.acoq,
            a.actq, a.ajexq, a.apq, a.atq, a.ceqq, a.cheq, a.cogsq, a.cshoq, a.cshprq,
            a.dlcq, a.dlttq, a.dpq, a.drcq, a.drltq, a.dvpsxq, a.dvpq, a.dvy, a.epspiq, a.epspxq, a.fopty,
            a.gdwlq, a.ibq, a.invtq, a.intanq, a.ivaoq, a.lcoq, a.lctq, a.loq, a.ltq, a.mibq,
            a.niq, a.oancfy, a.oiadpq, a.oibdpq, a.piq, a.ppentq, a.ppegtq, a.prstkcy, a.prccq,
            a.pstkq, a.rdq, a.req, a.rectq, a.revtq, a.saleq, a.seqq, a.sstky, a.txdiq, a.dltisy, a.mibtq,
            a.txditcq, a.txpq, a.txtq, a.xaccq, a.xintq, a.xsgaq, a.xrdq, a.capxy, a.dlcchy, a.dltry, a.dcomq
            FROM comp_na_daily_all.fundq as a
            WHERE a.consol = 'C'
            AND a.popsrc = 'D'
            AND a.datafmt = 'STD'
            AND a.curcdq = 'USD'
            AND a.indfmt = 'INDL'
            AND a.datadate BETWEEN '{self.start_date}' AND '{self.current_date}'
        """

        # Read in Compustat Quarterly
        print("Read In Compustat Quarterly...")
        db = wrds.Connection(wrds_username=self.wrd_key)
        compustat_quarterly = db.raw_sql(sql_compustat_quarterly)
        db.close()

        # Read in link table
        print("Read in link table...")
        link_table = pd.read_parquet(get_parquet(self.live) / 'data_link.parquet.brotli')
        link_table = link_table.drop(['cusip', 'conm'], axis=1)

        # Merge link table and Compustat Quarterly
        print("Merge link table and Compustat Quarterly...")
        quarterly = compustat_quarterly.merge(link_table, on='gvkey', how='left')

        # Apply the linkdate restriction
        print("Applying linkdate restriction...")
        quarterly['datadate'] = pd.to_datetime(quarterly['datadate'])
        quarterly = quarterly[(quarterly['datadate'] >= quarterly['timeLinkStart_d']) & (quarterly['datadate'] <= quarterly['timeLinkEnd_d'])]

        # Keep only the most recent data for each fiscal quarter
        print("Keep only the most recent data for each fiscal quarter...")
        quarterly = quarterly.sort_values(by=['gvkey', 'fyearq', 'fqtr', 'datadate'])
        quarterly = quarterly.groupby(['gvkey', 'fyearq', 'fqtr']).last().reset_index()

        # Convert to datetime
        print("Convert to datetime...")
        quarterly['rdq'] = pd.to_datetime(quarterly['rdq'])

        # Resample to monthly
        print("Resample to monthly...")
        quarterly['time_avail_m'] = (quarterly['datadate']).dt.to_period('M')
        quarterly.loc[(~quarterly['rdq'].isnull()) & (quarterly['rdq'].dt.to_period('M') > quarterly['time_avail_m']), 'time_avail_m'] = quarterly['rdq'].dt.to_period('M')

        # Compute month difference
        print("Compute month difference...")
        month_diff = (quarterly['rdq'] - quarterly['datadate']).dt.days // 30
        quarterly = quarterly.drop(quarterly[(month_diff > 6) & ~quarterly['rdq'].isnull()].index)
        quarterly = quarterly.sort_values(by=['gvkey', 'time_avail_m', 'datadate'])

        # Keep most recent data
        print("Keep most recent data...")
        quarterly = quarterly.groupby(['gvkey', 'time_avail_m']).last().reset_index()

        # Create extra yearly columns
        print("Create extra yearly columns...")
        for col in ['sstky', 'prstkcy', 'oancfy', 'fopty']:
            grouped = quarterly.groupby(['gvkey', 'fyearq'])[col]
            condition = quarterly['fqtr'] == 1
            new_values = np.where(condition, quarterly[col], quarterly[col] - grouped.shift(1))
            quarterly[col + 'q'] = new_values

        # Convert index from quarterly to monthly
        print("Convert index from quarterly to monthly...")
        quarterly = quarterly.loc[quarterly.index.repeat(3)]
        quarterly['tempTimeAvailM'] = quarterly['time_avail_m']
        quarterly = quarterly.sort_values(by=['gvkey', 'tempTimeAvailM'])
        quarterly['time_avail_m'] = quarterly.groupby(['gvkey', 'tempTimeAvailM']).cumcount() + quarterly['time_avail_m']

        # Sort values
        print("Sort values and keep most recent data...")
        quarterly = quarterly.sort_values(by=['gvkey', 'time_avail_m', 'datadate'])
        # Keep most recent data
        quarterly = quarterly.groupby(['gvkey', 'time_avail_m']).last().reset_index()
        quarterly = quarterly.drop(columns=['tempTimeAvailM'])
        quarterly = quarterly.rename(columns={'datadate': 'datadateq', 'time_avail_m': 'date'})

        # Convert from YY-MM to YY-MM-DD (2012-01 to 2012-01-31)
        print("Convert from YY-MM to YY-MM-DD (2012-01 to 2012-01-31)...")
        quarterly.date = quarterly.date.dt.to_timestamp("M")
        quarterly = quarterly[quarterly['date'] <= pd.Timestamp(self.current_date)]
        quarterly['permno'] = quarterly['permno'].astype(int)
        quarterly = quarterly.set_index(['permno', 'date'])
        quarterly = quarterly.sort_index(level=['permno', 'date'])
        quarterly = quarterly[~quarterly.index.duplicated(keep='last')]

        # Convert data to numerical format (exclude columns that are not numerical format)
        print("Convert data to numerical format (exclude columns that are not numerical format)...")
        numeric_cols = quarterly.select_dtypes(include=['number']).columns
        quarterly[numeric_cols] = quarterly[numeric_cols].astype(float)
        quarterly_numeric = quarterly[numeric_cols]
        quarterly_numeric = quarterly_numeric.sort_index(level=['permno', 'date'])
        quarterly_numeric = quarterly_numeric[~quarterly_numeric.index.duplicated(keep='last')]

        # Forward fill yearly data
        print("Forward fill yearly data...")
        cols_to_fill = [col for col in quarterly_numeric.columns if col.endswith('y')]
        quarterly_numeric[cols_to_fill] = quarterly_numeric[cols_to_fill].ffill()

        # Export data
        print("Export data...")
        quarterly_numeric.to_parquet(get_parquet(self.live) / 'data_fund_raw_q.parquet.brotli', compression='brotli')

    # Create Compustat Annual
    def create_compustat_annual(self):
        print("-" * 60)
        sql_compustat_annual = f"""
            SELECT a.gvkey, a.datadate, a.conm, a.fyear, a.tic, a.cusip, a.naicsh, a.sich, 
            a.aco, a.act, a.ajex, a.am, a.ao, a.ap, a.at, a.capx, a.ceq, a.ceqt, a.che, a.cogs,
            a.csho, a.cshrc, a.dcpstk, a.dcvt, a.dlc, a.dlcch, a.dltis, a.dltr,
            a.dltt, a.dm, a.dp, a.drc, a.drlt, a.dv,a.dvc,a.dvp,a.dvpa,a.dvpd,
            a.dvpsx_c, a.dvt, a.ebit, a.ebitda, a.emp, a.epspi, a.epspx, a.fatb, a.fatl,
            a.ffo, a.fincf, a.fopt, a.gdwl, a.gdwlia, a.gdwlip, a.gwo, a.ib, a.ibcom,
            a.intan, a.invt, a.ivao, a.ivncf, a.ivst, a.lco, a.lct, a.lo ,a.lt, a.mib,
            a.msa, a.ni, a.nopi, a.oancf, a.ob, a.oiadp, a.oibdp, a.pi, a.ppenb, a.ppegt,
            a.ppenls, a.ppent, a.prcc_c, a.prcc_f, a.prstkc, a.prstkcc, a.pstk, a.pstkl, a.pstkrv,
            a.re, a.rect, a.recta, a.revt, a.sale, a.scstkc, a.seq, a.spi, a.sstk,
            a.tstkp, a.txdb, a.txdi, a.txditc, a.txfo, a.txfed, a.txp, a.txt,
            a.wcap, a.wcapch, a.xacc, a.xad, a.xint, a.xrd, a.xpp, a.xsga, a.cdvc
            FROM comp_na_daily_all.funda as a
            WHERE a.consol = 'C'
            AND a.popsrc = 'D'
            AND a.datafmt = 'STD'
            AND a.curcd = 'USD'
            AND a.indfmt = 'INDL'
            AND a.datadate BETWEEN '{self.start_date}' AND '{self.current_date}'
        """

        # Read in Compustat Annual
        print("Read in Compustat Annual...")
        db = wrds.Connection(wrds_username=self.wrd_key)
        compustat_annual = db.raw_sql(sql_compustat_annual)
        db.close()

        # Read in link table
        print("Read in link table...")
        link_table = pd.read_parquet(get_parquet(self.live) / 'data_link.parquet.brotli')
        link_table = link_table.drop(['cusip', 'conm'], axis=1)

        # Merge link table and Compustat Annual
        print("Merge link table and Compustat Annual...")
        annual = compustat_annual.merge(link_table, on='gvkey', how='left')

        # Apply the linkdate restriction
        print("Applying linkdate restriction...")
        annual['datadate'] = pd.to_datetime(annual['datadate'])
        annual = annual[(annual['datadate'] >= annual['timeLinkStart_d']) & (annual['datadate'] <= annual['timeLinkEnd_d'])]

        # Keep only the most recent data for each fiscal quarter
        print("Keep only the most recent data for each fiscal quarter...")
        annual = annual.sort_values(by=['gvkey', 'fyear', 'datadate'])
        annual = annual.groupby(['gvkey', 'fyear']).last().reset_index()

        # Drop rows based on condition
        print("Drop rows based on condition...")
        annual = annual.dropna(subset=['at', 'prcc_c', 'ni'])

        # Extract 6 digits from CUSIP
        print("Extract 6 digits from CUSIP...")
        annual['cnum'] = annual['cusip'].str[:6]

        # Replacing missing values
        print("Replacing missing values...")
        annual['dr'] = annual.apply(
            lambda row: row['drc'] + row['drlt'] if pd.notna(row['drc']) and pd.notna(row['drlt']) else (row['drc'] if pd.notna(row['drc']) else (row['drlt'] if pd.notna(row['drlt']) else None)),
            axis=1)
        annual.loc[(annual['dcpstk'] > annual['pstk']) & pd.notna(annual['dcpstk']) & pd.notna(annual['pstk']) & pd.isna(annual['dcvt']), 'dc'] = annual['dcpstk'] - annual['pstk']
        annual.loc[pd.isna(annual['pstk']) & pd.notna(annual['dcpstk']) & pd.isna(annual['dcvt']), 'dc'] = annual['dcpstk']
        annual.loc[pd.isna(annual['dc']), 'dc'] = annual['dcvt']
        annual['xint0'] = annual['xint'].fillna(0)
        annual['xsga0'] = annual['xsga'].fillna(0)
        annual['xad0'] = annual.apply(lambda row: 0 if row['xad'] < 0 else row['xad'], axis=1)
        vars_list = ['nopi', 'dvt', 'ob', 'dm', 'dc', 'aco', 'ap', 'intan', 'ao', 'lco', 'lo', 'rect', 'invt', 'drc', 'spi', 'gdwl', 'che', 'dp', 'act', 'lct', 'tstkp', 'dvpa', 'scstkc', 'sstk',
                     'mib', 'ivao', 'prstkc', 'prstkcc', 'txditc', 'ivst']
        for var in vars_list:
            annual[var].fillna(0, inplace=True)

        # Resample to monthly
        print("Resample to monthly...")
        annual['date'] = annual['datadate'].dt.to_period('M')

        # Convert index from annually to monthly
        print("Convert index from annually to monthly...")
        annual = annual.reindex(annual.index.repeat(12))
        annual['tempTime'] = annual.groupby(['gvkey', 'date']).cumcount()
        annual['date'] += annual['tempTime']
        annual = annual.drop(columns=['tempTime'])

        # Convert from YY-MM to YY-MM-DD (2012-01 to 2012-01-31)
        print("Convert from YY-MM to YY-MM-DD (2012-01 to 2012-01-31)")
        annual.date = annual.date.dt.to_timestamp("M")
        annual = annual.drop('datadate', axis=1)
        annual = annual[annual['date'] <= pd.Timestamp(self.current_date)]

        # Set index and remove duplicate indices
        print("Set index and remove duplicate indices...")
        annual['permno'] = annual['permno'].astype(int)
        annual = annual.set_index(['permno', 'date'])
        annual = annual.sort_index(level=['permno', 'date'])
        annual = annual[~annual.index.duplicated(keep='last')]

        # Export data
        print("Export data...")
        annual.to_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli', compression='brotli')


    # Create Common Stock List
    def create_stock_list(self):
        print("-" * 60)
        # Read in files
        print("Read in files...")
        quarterly = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_q.parquet.brotli')
        annual = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli')
        price = pd.read_parquet(get_parquet(self.live) / 'data_crsp_price.parquet.brotli')
        quarterly_list = get_stock_idx(quarterly)
        annual_list = get_stock_idx(annual)
        price_list = get_stock_idx(price)
        set1 = set(quarterly_list)
        set2 = set(annual_list)
        set3 = set(price_list)

        # Find common stocks
        print("Find common stocks...")
        common_elements = set1.intersection(set2, set3)
        common_stock_list = list(common_elements)

        # Export stock list
        print("Export stock list...")
        print(f'Number of stocks: {len(common_stock_list)}')
        common = pd.DataFrame(index=common_stock_list)
        common.index.names = ['permno']
        export_stock(common, get_large(self.live) / 'permno_common.csv')


    # Create Live Price
    def create_live_price(self):
        print("-" * 60)
        # Get ticker list from crsp_ticker
        print("Get common ticker list...")
        crsp_ticker = pd.read_parquet(get_parquet(self.live) / 'data_crsp_ticker.parquet.brotli')
        common_stock_list = read_stock(get_large(self.live) / 'permno_common.csv')
        crsp_ticker = get_stocks_data(crsp_ticker, common_stock_list)
        # This date should be the end of the annual CRSP dataset (i.e., 2022-12-31)
        last_date = crsp_ticker.index.get_level_values('date').max()
        ticker_list = crsp_ticker.loc[crsp_ticker.index.get_level_values('date') == last_date, 'ticker'].tolist()

        # Read in trade_live market data
        print("Read in trade_live market data...")
        start_year = datetime.strptime(self.current_date, "%Y-%m-%d")
        start_year = start_year.replace(month=1, day=1)
        start_year = start_year.strftime("%Y-%m-%d")
        price = get_data_fmp(ticker_list=ticker_list, start=start_year, current_date=self.current_date)

        # Extract permno ticker pair used for mapping
        print("Extract permno ticker pair used for mapping...")
        permno_ticker_map = crsp_ticker.loc[crsp_ticker.index.get_level_values('date') == last_date][['ticker']]

        # Map permno ticker pair to trade_live market data
        print("Map permno ticker pair to trade_live market data...")
        permno_ticker_map = permno_ticker_map.reset_index()
        price_change = price.reset_index()
        price_change['permno'] = price_change['ticker'].map(permno_ticker_map.set_index('ticker').to_dict()['permno'])
        price_change = price_change.dropna(subset=['permno'])  # drop rows without a matching permno
        price_change['permno'] = price_change['permno'].astype(int)  # Convert permno to int if it's in float due to NaNs
        price_change = price_change.set_index(['permno', 'date'])

        # Read in CRSP price
        print("Read in CRSP price...")
        crsp_price = pd.read_parquet(get_parquet(self.live) / 'data_crsp_price.parquet.brotli')
        crsp_price = get_stocks_data(crsp_price, common_stock_list)

        # Change adj close to close
        print("Change adj close to close...")
        price_change = price_change.drop('Close', axis=1)
        price_change = price_change.rename(columns={'Adj Close': 'Close'})
        price_change_price = price_change[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Concat and sort trade_historical price and trade_live price
        print("Concat and sort trade_historical price and trade_live price...")
        combined_price = pd.concat([crsp_price, price_change_price], axis=0)
        combined_price = combined_price.sort_index(level=['permno', 'date'])

        ret = create_return(combined_price, [1])

        # Identify and drop permnos that have returns greater than 10 (this is due to price descrepancy between CRSP and FMP)
        print("Identify and drop permnos that have returns greater than 10 (this removes extremely volatile stocks (i.e., TPST on October 10, 2023))...")
        permnos_to_remove = ret.loc[ret.RET_01 > 10].index.get_level_values('permno').unique()
        print(f"Number of stocks to remove: {len(permnos_to_remove)}")
        combined_price = combined_price.drop(permnos_to_remove, level='permno')

        # Filter for the period of interest around start_year and then identify permnos with returns greater than 5
        print("Filter for the period of interest around start_year and then identify permnos with returns greater than 5 (this is due to price discrepancy between CRSP and FMP)...")
        print("Note: some stocks may have already been removed by the >10 condition (there may be some overlaps)")
        start_year = pd.to_datetime(start_year)
        subset_ret = ret.loc[(ret.index.get_level_values('date') >= start_year - pd.Timedelta(days=5)) & (ret.index.get_level_values('date') <= start_year + pd.Timedelta(days=5))]
        permnos_to_remove = subset_ret[subset_ret.RET_01 > 5].index.get_level_values('permno').unique()
        print(f"Number of stocks to remove: {len(permnos_to_remove)}")
        combined_price = combined_price.drop(permnos_to_remove, level='permno')
        combined_price = combined_price.drop('RET_01', axis=1)

        # Concat and sort trade_historical ticker and trade_live ticker
        print("Concat and sort trade_historical ticker and trade_live ticker...")
        price_change_ticker = price_change[['ticker']]
        combined_ticker = pd.concat([crsp_ticker, price_change_ticker], axis=0)
        combined_ticker = combined_ticker.sort_index(level=['permno', 'date'])
        combined_price_stock_list = get_stock_idx(combined_price)
        combined_ticker = get_stocks_data(combined_ticker, combined_price_stock_list)

        # Export ohclv
        print('Export Price...')
        combined_price.to_parquet(get_parquet(self.live) / 'data_price.parquet.brotli', compression='brotli')

        # Export date
        print('Export Date...')
        date = combined_price.drop(columns=combined_price.columns)
        date.to_parquet(get_parquet(self.live) / 'data_date.parquet.brotli', compression='brotli')

        # Export ticker
        print('Export Tickers...')
        combined_ticker.to_parquet(get_parquet(self.live) / 'data_ticker.parquet.brotli', compression='brotli')

        # Export permno list for trade_live trading
        print("Export permno list for trade_live trading...")
        print(f'Number of stocks: {len(get_stock_idx(combined_price))}')
        export_stock(combined_price, get_large(self.live) / 'permno_live.csv')

    # Create Misc
    def create_misc(self):
        print("-" * 60)
        print("Read in Misc...")
        quarterly = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_q.parquet.brotli', columns=['cshoq'])
        quarterly.columns = ['outstanding']
        misc = pd.read_parquet(get_parquet(self.live) / 'data_price.parquet.brotli')
        misc = misc.merge(quarterly, left_index=True, right_index=True, how='left')

        # Add outstanding and market cap
        print("Add outstanding and market cap...")
        misc['outstanding'] = misc.groupby('permno')['outstanding'].ffill()
        misc['outstanding'] = misc['outstanding'] * 1_000_000
        misc['market_cap'] = misc['Close'] * misc['outstanding']
        misc = misc[['outstanding', 'market_cap']]
        misc = misc[~misc.index.duplicated(keep='first')]

        # Get Dividend Data from FMP
        print("Get dividend data from FMP...")
        ticker = pd.read_parquet(get_parquet(self.live) / 'data_ticker.parquet.brotli')
        tickers = ticker.ticker.unique().tolist()
        dividends = get_dividend_fmp(tickers, self.start_date, self.current_date)
        tic_reset = ticker.reset_index()
        div_reset = dividends.reset_index()
        combined = pd.merge(tic_reset, div_reset, on=['ticker', 'date'], how='left')
        combined = combined.set_index(['permno', 'date']).sort_index(level=['permno', 'date'])
        combined = combined.rename(columns={'paymentDate': 'paydt'})
        combined = combined.drop('ticker', axis=1)
        combined = combined[~combined.index.duplicated(keep='first')]

        misc = misc.merge(combined, left_index=True, right_index=True, how='left')
        misc = misc[~misc.index.duplicated(keep='first')]

        # Export data
        print("Export data...")
        misc.to_parquet(get_parquet(self.live) / 'data_misc.parquet.brotli', compression='brotli')

    # Create Compustat Pension
    def create_compustat_pension(self):
        print("-" * 60)
        sql_compustat_pension = f"""
            SELECT a.gvkey, a.datadate, a.paddml, a.pbnaa, a.pbnvv, a.pbpro, 
                   a.pbpru, a.pcupsu, a.pplao, a.pplau
            FROM comp_na_daily_all.aco_pnfnda as a
            WHERE a.consol = 'C'
            AND a.popsrc = 'D'
            AND a.datafmt = 'STD'
            AND a.indfmt = 'INDL'
            AND a.datadate BETWEEN '{self.start_date}' AND '{self.current_date}'
        """

        # Read in Pension Annual
        print("Read in Pension Annual...")
        db = wrds.Connection(wrds_username=self.wrd_key)
        pension = db.raw_sql(sql_compustat_pension)
        db.close()

        # Drop duplicate indices
        print("Drop duplicate indices...")
        pension = pension.sort_values(by=['gvkey', 'datadate'])
        pension = pension.groupby(['gvkey', 'datadate']).last().reset_index()

        # Convert to datetime and set index
        print("Convert to datetime and set index...")
        pension['datadate'] = pd.to_datetime(pension['datadate'])
        pension = pension.rename(columns={'datadate': 'date', 'tic': 'ticker'})
        pension = pension.set_index('date')

        # Export data
        print("Export data...")
        pension.to_parquet(get_parquet(self.live) / 'data_pension.parquet.brotli', compression='brotli')

    # Create Industry
    def create_industry(self):
        print("-" * 60)
        sql_compustat_industry = f"""
            SELECT a.gvkey, a.gind, a.gsubind
            FROM comp_na_daily_all.namesq as a
        """

        # Read in Compustat Industry
        print("Read in Compustat Industry...")
        db = wrds.Connection(wrds_username=self.wrd_key)
        industry_compustat = db.raw_sql(sql_compustat_industry)
        db.close()

        # Read in link table
        print("Read in link table...")
        link_table = pd.read_parquet(get_parquet(self.live) / 'data_link.parquet.brotli')
        link_table = link_table.drop(['conm', 'sic'], axis=1)

        # Merge link table and Compustat Annual
        print("Merge link table and Compustat Annual...")
        industry = pd.merge(industry_compustat, link_table, on='gvkey', how='left')

        # Rename Columns
        print('Rename columns...')
        industry.columns = industry.columns.str.lower()
        industry = industry.rename(columns={'gsubind': 'Subindustry', 'gind': 'Industry'})

        # Remove duplicate permno
        print('Remove duplicate permno...')
        industry = industry.drop_duplicates(subset='permno')
        industry = industry[['permno', 'Industry', 'Subindustry']]

        # Read in Compustat Annual
        print("Read in Compustat Annual")
        annual = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_a.parquet.brotli', columns=['sich'])
        stock = read_stock(get_large(self.live) / 'permno_live.csv')
        annual = get_stocks_data(annual, stock)
        annual.columns = ['sic_comp']

        # Assign Fama industries based off given ranges
        print("Assign Fama industries based off given ranges...")

        def assign_ind(df, column_name, sic_ranges, label):
            # Sic from CRSP and Compustat
            df['sic_temp_comp'] = df['sic_comp']

            # Iterate through each range and assign industry
            for r in sic_ranges:
                if isinstance(r, tuple):
                    df.loc[(df['sic_temp_comp'] >= r[0]) & (df['sic_temp_comp'] <= r[1]), f'{column_name}_comp'] = label
                else:
                    df.loc[df['sic_temp_comp'] == r, f'{column_name}_comp'] = label

            df = df.drop(columns=['sic_temp_comp'], axis=1)
            return df

        # FF49 Industry ranges
        fama_ind = {
            'agric': [(100, 199), (200, 299), (700, 799), (910, 919), 2048],
            'food': [(2000, 2009), (2010, 2019), (2020, 2029), (2030, 2039), (2040, 2046), (2050, 2059), (2060, 2063), (2070, 2079), (2090, 2092), 2095, (2098, 2099)],
            'soda': [(2064, 2068), 2086, 2087, 2096, 2097],
            'beer': [2080, 2082, 2083, 2084, 2085],
            'smoke': [(2100, 2199)],
            'toys': [(920, 999), (3650, 3651), 3652, 3732, (3930, 3931), (3940, 3949)],
            'fun': [(7800, 7829), (7830, 7833), (7840, 7841), 7900, (7910, 7911), (7920, 7929), (7930, 7933), (7940, 7949), 7980, (7990, 7999)],
            'books': [(2700, 2709), (2710, 2719), (2720, 2729), (2730, 2739), (2740, 2749), (2770, 2771), (2780, 2789), (2790, 2799)],
            'hshld': [2047, (2391, 2392), (2510, 2519), (2590, 2599), (2840, 2843), 2844, (3160, 3161), (3170, 3171), 3172, (3190, 3199), 3229, 3260, (3262, 3263), 3269, (3230, 3231), (3630, 3639),
                      (3750, 3751), 3800, (3860, 3861), (3870, 3873), (3910, 3911), 3914, 3915, (3960, 3962), 3991, 3995],
            'clths': [(2300, 2390), (3020, 3021), (3100, 3111), (3130, 3131), (3140, 3149), (3150, 3151), (3963, 3965)],
            'hlth': [(8000, 8099)],
            'medeq': [3693, (3840, 3849), (3850, 3851)],
            'drugs': [2830, 2831, 2833, 2834, 2835, 2836],
            'chems': [(2800, 2809), (2810, 2819), (2820, 2829), (2850, 2859), (2860, 2869), (2870, 2879), (2890, 2899)],
            'rubbr': [3031, 3041, (3050, 3053), (3060, 3069), (3070, 3079), (3080, 3089), (3090, 3099)],
            'txtls': [(2200, 2269), (2270, 2279), (2280, 2284), (2290, 2295), 2297, 2298, 2299, (2393, 2395), (2397, 2399)],
            'bldmt': [(800, 899), (2400, 2439), (2450, 2459), (2490, 2499), (2660, 2661), (2950, 2952), 3200, (3210, 3211), (3240, 3241), (3250, 3259), 3261, 3264, (3270, 3275), (3280, 3281),
                      (3290, 3293), (3295, 3299), (3420, 3429), (3430, 3433), (3440, 3441), 3442, 3446, 3448, 3449, (3450, 3451), 3452, (3490, 3499), 3996],
            'cnstr': [(1500, 1511), (1520, 1529), (1530, 1539), (1540, 1549), (1600, 1699), (1700, 1799)],
            'steel': [3300, (3310, 3317), (3320, 3325), (3330, 3339), (3340, 3341), (3350, 3357), (3360, 3369), (3370, 3379), (3390, 3399)],
            'fabpr': [3400, 3443, 3444, (3460, 3469), (3470, 3479)],
            'mach': [(3510, 3519), (3520, 3529), 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3538, (3540, 3549), (3550, 3559), (3560, 3569), 3580, 3581, 3582, 3585, 3586, 3589, (3590, 3599)],
            'elceq': [3600, (3610, 3613), (3620, 3621), (3623, 3629), (3640, 3644), 3645, 3646, (3648, 3649), 3660, 3690, (3691, 3692), 3699],
            'autos': [2296, 2396, (3010, 3011), 3537, 3647, 3694, 3700, 3710, 3711, 3713, 3714, 3715, 3716, 3792, (3790, 3791), 3799],
            'aero': [3720, 3721, (3723, 3724), 3725, (3728, 3729)],
            'ships': [(3730, 3731), (3740, 3743)],
            'guns': [(3760, 3769), 3795, (3480, 3489)],
            'gold': [(1040, 1049)],
            'mines': [(1000, 1009), (1010, 1019), (1020, 1029), (1030, 1039), (1050, 1059), (1060, 1069), (1070, 1079), (1080, 1089), (1090, 1099), (1100, 1119), (1400, 1499)],
            'coal': [(1200, 1299)],
            'oil': [1300, (1310, 1319), (1320, 1329), (1330, 1339), (1370, 1379), 1380, 1381, 1382, 1389, (2900, 2912), (2990, 2999)],
            'util': [4900, (4910, 4911), (4920, 4922), 4923, (4924, 4925), (4930, 4931), 4932, 4939, (4940, 4942)],
            'telcm': [4800, (4810, 4813), (4820, 4822), (4830, 4839), (4840, 4841), 4880, 4890, 4891, 4892, 4899],
            'persv': [(7020, 7021), (7030, 7033), 7200, (7210, 7212), 7214, (7215, 7216), 7217, 7219, (7220, 7221), (7230, 7231), (7240, 7241), (7250, 7251), (7260, 7269), (7270, 7290), 7291,
                      (7292, 7299), 7395, 7500, (7520, 7529), (7530, 7539), (7540, 7549), 7600, 7620, 7622, 7623, 7629, 7630, 7640, (7690, 7699), (8100, 8199), (8200, 8299), (8300, 8399),
                      (8400, 8499), (8600, 8699), (8800, 8899), (7510, 7515)],
            'bussv': [(2750, 2759), 3993, 7218, 7300, (7310, 7319), (7320, 7329), (7330, 7339), (7340, 7342), 7349, (7350, 7351), 7352, 7353, 7359, (7360, 7369), 7374, 7376, 7377, 7378, 7379, 7380,
                      (7381, 7382), 7383, 7384, 7385, 7389, 7390, 7391, (7392, 7392), 7393, 7394, 7396, 7397, 7399, (7519, 7519), 8700, (8710, 8713), (8720, 8721), (8730, 8734), (8740, 8748),
                      (8900, 8910), 8911, (8920, 8999), (4220, 4229)],
            'hardw': [(3570, 3579), 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3695],
            'softw': [(7370, 7372), 7375, 7373],
            'chips': [3622, 3661, (3662, 3662), 3663, 3664, 3665, 3666, 3669, (3670, 3679), (3810, 3810), (3812, 3812)],
            'labeq': [3811, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3829, (3830, 3839)],
            'paper': [(2520, 2549), (2600, 2639), (2670, 2699), (2760, 2761), (3950, 3955)],
            'boxes': [(2440, 2449), (2640, 2659), (3220, 3221), (3410, 3412)],
            'trans': [(4000, 4013), (4040, 4049), 4100, (4110, 4119), (4120, 4121), (4130, 4131), (4140, 4142), (4150, 4151), (4170, 4173), (4190, 4199), 4200, (4210, 4219), (4230, 4231),
                      (4240, 4249), (4400, 4499), (4500, 4599), (4600, 4699), 4700, (4710, 4712), (4720, 4729), (4730, 4739), (4740, 4749), 4780, 4782, 4783, 4784, 4785, 4789],
            'whlsl': [5000, (5010, 5015), (5020, 5023), (5030, 5039), (5040, 5042), 5043, 5044, 5045, 5046, 5047, 5048, 5049, (5050, 5059), 5060, 5063, 5064, 5065, (5070, 5078), 5080, 5081, 5082,
                      5083, 5084, 5085, (5086, 5087), 5088, 5090, (5091, 5092), 5093, 5094, 5099, 5100, (5110, 5113), (5120, 5122), (5130, 5139), (5140, 5149), (5150, 5159), (5160, 5169),
                      (5170, 5172), (5180, 5182), (5190, 5199)],
            'rtail': [5200, (5210, 5219), (5220, 5229), (5230, 5231), (5250, 5251), (5260, 5261), (5270, 5271), 5300, 5310, 5320, (5330, 5331), 5334, (5340, 5349), (5390, 5399), 5400, (5410, 5411),
                      5412, (5420, 5429), (5430, 5439), (5440, 5449), (5450, 5459), (5460, 5469), (5490, 5499), 5500, (5510, 5529), (5530, 5539), (5540, 5549), (5550, 5559), (5560, 5569),
                      (5570, 5579), (5590, 5599), (5600, 5699), 5700, (5710, 5719), (5720, 5722), (5730, 5733), 5734, 5735, 5736, (5750, 5799), 5900, (5910, 5912), (5920, 5929), (5930, 5932), 5940,
                      5941, 5942, 5943, 5944, 5945, 5946, 5947, 5948, 5949, (5950, 5959), (5960, 5969), (5970, 5979), (5980, 5989), 5990, 5992, 5993, 5994, 5995, 5999],
            'meals': [(5800, 5819), (5820, 5829), (5890, 5899), 7000, (7010, 7019), (7040, 7049), 7213],
            'banks': [6000, (6010, 6019), 6020, 6021, 6022, 6023, 6025, 6026, 6027, (6028, 6029), (6030, 6036), (6040, 6059), (6060, 6062), (6080, 6082), (6090, 6099), 6100, (6110, 6111),
                      (6112, 6113), (6120, 6129), (6130, 6139), (6140, 6149), (6150, 6159), (6160, 6169), (6170, 6179), (6190, 6199)],
            'insur': [6300, (6310, 6319), (6320, 6329), (6330, 6331), (6350, 6351), (6360, 6361), (6370, 6379), (6390, 6399), (6400, 6411)],
            'rlest': [6500, 6510, 6512, 6513, 6514, 6515, (6517, 6519), (6520, 6529), (6530, 6531), 6532, (6540, 6541), (6550, 6553), (6590, 6599), (6610, 6611)],
            'fin': [(6200, 6299), 6700, (6710, 6719), (6720, 6722), 6723, 6724, 6725, 6726, (6730, 6733), (6740, 6779), 6790, 6791, 6792, 6793, 6794, 6795, 6798, 6799],
            'other': [(4950, 4959), (4960, 4961), (4970, 4971), (4990, 4991)]
        }

        # Iterate through each key
        print("Iterate through each key...")
        print('-' * 30)
        for name, ranges in fama_ind.items():
            print(name)
            combined = assign_ind(annual, 'IndustryFama', ranges, name)

        # Assign industry based off Compustat. If Compustat is NAN, then use CRSP
        print("Assign industry based off Compustat...")
        annual['IndustryFama'] = annual['IndustryFama_comp']
        annual['IndustryFama'], category_mapping = annual['IndustryFama'].factorize()

        # Fill NAN values for industries with -1
        print("Fill industries with -1...")
        cols_to_fill = ['Industry', 'Subindustry']
        industry[cols_to_fill] = industry[cols_to_fill].fillna(-1)
        industry[cols_to_fill] = industry[cols_to_fill].astype(int)
        annual = annual[['IndustryFama']]
        annual = annual.fillna(-1).astype(int)

        # Retrieve trade_live trade stock list
        print("Retrieve trade_live trade stock list...")
        stock = read_stock(get_large(self.live) / 'permno_live.csv')
        industry = industry[industry['permno'].isin(stock)]

        # Merge ind data with price dataset
        print("Merge ind data with price dataset...")
        date = pd.read_parquet(get_parquet(self.live) / 'data_date.parquet.brotli')
        date = date.reset_index()
        industry = pd.merge(date, industry, on=['permno'], how='left')
        annual = annual.reset_index()
        industry = pd.merge(industry, annual, on=['permno', 'date'], how='left')
        industry['IndustryFama'] = industry.groupby('permno')['IndustryFama'].ffill().bfill().astype(int)

        # Sort index
        print("Sort index...")
        industry = industry.set_index(['permno', 'date']).sort_index(level=['permno', 'date'])
        industry = industry[~industry.index.duplicated(keep='first')]
        industry = industry.replace([np.inf, -np.inf], np.nan)

        # Export data
        print("Export data...")
        industry[['Industry']].to_parquet(get_parquet(self.live) / 'data_ind.parquet.brotli', compression='brotli')
        industry[['Subindustry']].to_parquet(get_parquet(self.live) / 'data_ind_sub.parquet.brotli', compression='brotli')
        industry[['IndustryFama']].to_parquet(get_parquet(self.live) / 'data_ind_fama.parquet.brotli', compression='brotli')

    def create_ibes(self):
        print("-" * 60)
        tickers = pd.read_parquet(get_parquet(self.live) / 'data_ticker.parquet.brotli')
        tickers = tickers.ticker.unique().tolist()
        ticker_string = ', '.join(f"'{ticker}'" for ticker in tickers)

        sql_actual_adj = f"""
            SELECT b.oftic, b.prdays, b.price, b.shout, b.ticker, b.statpers
            FROM ibes.actpsum_epsus as b
            WHERE b.oftic IN ({ticker_string});
        """

        sql_statistic_adj = f"""
            SELECT a.fpi, a.oftic, a.ticker, a.statpers, a.fpedats, a.anndats_act, 
            a.meanest, a.actual, a.medest, a.stdev, a.numest
            FROM ibes.statsum_epsus as a
            WHERE a.oftic IN ({ticker_string});
        """

        # Read in IBES Actual Adj
        print("Read in IBES Actual Adj...")
        db = wrds.Connection(wrds_username=self.wrd_key)
        actual_adj = db.raw_sql(sql_actual_adj)
        db.close()

        # Read in IBES Statistic Adj
        print("Read in IBES Statistic Adj...")
        db = wrds.Connection(wrds_username=self.wrd_key)
        statistic_adj = db.raw_sql(sql_statistic_adj)
        db.close()

        print("Export Data...")
        actual_adj.to_csv(get_large(self.live) / 'summary_actual_adj_ibes.csv', index=False)
        statistic_adj.to_csv(get_large(self.live) / 'summary_statistic_adj_ibes.csv', index=False)

    # Create Macro
    def create_macro(self):
        print("-" * 60)
        # API key
        fred = Fred(api_key=self.fred_key)

        # Read in Median CPI
        print("Read in Median CPI Index...")
        median_cpi = fred.get_series("MEDCPIM158SFRBCLE").to_frame()
        median_cpi = median_cpi.reset_index()
        median_cpi.columns = ['date', 'medCPI']

        # Read in Producer Production Index
        print("Read in Producer Production Index...")
        ppi = fred.get_series("PPIACO").to_frame()
        ppi = ppi.reset_index()
        ppi.columns = ['date', 'PPI']

        # Read in Industry Production
        print("Read in Industry Production Index...")
        ind_prod = fred.get_series("INDPRO").to_frame()
        ind_prod = ind_prod.reset_index()
        ind_prod.columns = ['date', 'indProdIndex']

        # Read in Median CPI
        print("Read in 10 Year Real Interest Rate...")
        ir_rate = fred.get_series("REAINTRATREARAT10Y").to_frame()
        ir_rate = ir_rate.reset_index()
        ir_rate.columns = ['date', 'rIR']

        # Read in Inflation Rate
        print("Read in Inflation Rate...")
        if_rate = fred.get_series("T7YIEM").to_frame()
        if_rate = if_rate.reset_index()
        if_rate.columns = ['date', '5YIF']

        # Read in Unemployment Rate
        print("Read in Unemployment Rate...")
        u_rate = fred.get_series("UNRATE").to_frame()
        u_rate = u_rate.reset_index()
        u_rate.columns = ['date', 'UR']

        # Export data
        print("Export data...")
        median_cpi.to_csv(get_large(self.live) / 'macro' / 'medianCPI.csv', index=False)
        ppi.to_csv(get_large(self.live) / 'macro' / 'PPI.csv', index=False)
        ind_prod.to_csv(get_large(self.live) / 'macro' / 'indProdIndex.csv', index=False)
        ir_rate.to_csv(get_large(self.live) / 'macro' / 'realInterestRate.csv', index=False)
        if_rate.to_csv(get_large(self.live) / 'macro' / 'fiveYearIR.csv', index=False)
        u_rate.to_csv(get_large(self.live) / 'macro' / 'unemploymentRate.csv', index=False)

    # Create Risk Free Rate
    def create_risk_rate(self):
        print("-" * 60)

        # API key
        fred = Fred(api_key=self.fred_key)

        # Read in Median CPI
        print("Read in 1 Month Treasury Yield...")
        rates = fred.get_series("DGS1MO").to_frame()
        rates.columns = ['RF']
        rates.index = pd.to_datetime(rates.index)
        rates.index = rates.index + pd.DateOffset(days=1)
        rates = rates / 100
        rates.index.names = ['date']

        # Export Data
        print("Export Data...")
        rates.to_parquet(get_parquet(self.live) / 'data_rf.parquet.brotli', compression='brotli')

    # Get ETF Data for Mean Reversion Strategy
    def create_etf(self):
        print("-" * 60)
        # Get data from FMP
        print("Get data from FMP...")
        etf_data = get_data_fmp(ticker_list=['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU'], start=self.start_date, current_date=self.current_date)
        etf_data = etf_data[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
        etf_data = etf_data.rename(columns={'Adj Close': 'Close'})

        # Export Data
        print("Export Data...")
        etf_data.to_parquet(get_parquet(self.live) / 'data_etf.parquet.brotli', compression='brotli')
