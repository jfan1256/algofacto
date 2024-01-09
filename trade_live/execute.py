import asyncio
import concurrent.futures
import schedule

from ib_insync import *

from core.operation import *

from class_live.live_create import LiveCreate
from class_live.live_price import LivePrice
from class_live.live_close import LiveClose
from class_live.live_trade import LiveTrade

from trade_live.strat_ml_trend.strat_ml_trend import StratMLTrend
from trade_live.strat_ml_ret.strat_ml_ret import StratMLRet
from trade_live.strat_port_iv.strat_port_iv import StratPortIV
from trade_live.strat_port_im.strat_port_im import StratPortIM
from trade_live.strat_port_id.strat_port_id import StratPortID
from trade_live.strat_port_ivmd.strat_port_ivmd import StratPortIVMD
from trade_live.strat_trend_mls.strat_trend_mls import StratTrendMLS
from trade_live.strat_mrev_etf.strat_mrev_etf import StratMrevETF
from trade_live.strat_mrev_mkt.strat_mrev_mkt import StratMrevMkt

from class_monitor.monitor_strat import MonitorStrat

def build():
    '''
    Note: Specify strategy criteria in strat_crit.json. Each strategy's criteria should contain "allocate", "start_backtest", "num_stocks", and "threshold":
                - allocate (float): Percentage of capital to allocate for a strategy (allocation weights should remain static and sum up to 1 across all strategies)
                - start_backtest (str: YYYY-MM-DD): Start date for a strategy's backtest period
                - num_stocks (list: [int (# of long), int (# of short)]: Number of stocks to long and short for a strategy
                - threshold (int): Market Cap Threshold that must be met in order to long/short a stock for a strategy

          Specify data criteria in data_crit.json:
                - threshold (int): Market Cap Threshold that must be met in order to retrieve the data of a stock (stock's market_cap average over entire timeline > threshold)
                - age (int): Age of stock threshold that must be met in order to retrieve the data of a stock (stock's age > age)
                - annual_update (str: "True" or "False"): Update crsp_price data set or not (this should be updated annually at the start of the year: manually download CRSP Daily Price Dataset from WRDS)
                - start_date (str: YYYY-MM-DD): Start date for data retrieval

          build() should be executed at 12:01 AM EST Daily on Monday to Friday
    '''

    # Log Time
    start_time = time.time()

    # Get strategy criteria
    strat_crit = json.load(open(get_config() / 'strat_crit.json'))
    # Get data criteria
    data_crit = json.load(open(get_config() / 'data_crit.json'))
        
    # Params
    current_date = date.today().strftime('%Y-%m-%d')

    # Create Strategies
    strat_ml_ret = StratMLRet(allocate=strat_crit['ml_ret']['allocate'], current_date=current_date, start_model=strat_crit['ml_ret']['start_backtest'], threshold=strat_crit['ml_ret']['threshold'], num_stocks=strat_crit['ml_ret']['per_side'][0], leverage=0.5, port_opt='equal_weight', use_top=6)
    start_ml_trend = StratMLTrend(allocate=strat_crit['ml_trend']['allocate'], current_date=current_date, start_model=strat_crit['ml_trend']['start_backtest'], threshold=strat_crit['ml_trend']['threshold'], num_stocks=strat_crit['ml_trend']['per_side'][0], leverage=0.5, port_opt='equal_weight', use_top=1)
    strat_port_iv = StratPortIV(allocate=strat_crit['port_iv']['allocate'], current_date=current_date, start_date=strat_crit['port_iv']['start_backtest'], threshold=strat_crit['port_iv']['threshold'], num_stocks=strat_crit['port_iv']['per_side'][0], window_port=21)
    strat_port_im = StratPortIM(allocate=strat_crit['port_im']['allocate'], current_date=current_date, start_date=strat_crit['port_im']['start_backtest'], threshold=strat_crit['port_im']['threshold'], num_stocks=strat_crit['port_im']['per_side'][0], window_port=21)
    strat_port_id = StratPortID(allocate=strat_crit['port_id']['allocate'], current_date=current_date, start_date=strat_crit['port_id']['start_backtest'], threshold=strat_crit['port_id']['threshold'], num_stocks=strat_crit['port_id']['per_side'][0], window_port=21)
    strat_port_ivmd = StratPortIVMD(allocate=strat_crit['port_ivmd']['allocate'], current_date=current_date, start_date=strat_crit['port_ivmd']['start_backtest'], threshold=strat_crit['port_ivmd']['threshold'], num_stocks=strat_crit['port_ivmd']['per_side'][0], window_port=21)
    strat_trend_mls = StratTrendMLS(allocate=strat_crit['trend_mls']['allocate'], current_date=current_date, start_date=strat_crit['trend_mls']['start_backtest'], threshold=strat_crit['trend_mls']['threshold'], num_stocks=strat_crit['trend_mls']['per_side'][0], window_hedge=60, window_port=252)
    strat_mrev_etf = StratMrevETF(allocate=strat_crit['mrev_etf']['allocate'], current_date=current_date, start_date=strat_crit['mrev_etf']['start_backtest'], threshold=strat_crit['mrev_etf']['threshold'], window_epsil=168, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)
    strat_mrev_mkt = StratMrevMkt(allocate=strat_crit['mrev_mkt']['allocate'], current_date=current_date, start_date=strat_crit['mrev_mkt']['start_backtest'], threshold=strat_crit['mrev_mkt']['threshold'], window_epsil=168, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)

    # Create Live Create
    live_retrieve = LiveCreate(current_date=current_date, threshold=data_crit['threshold'], set_length=data_crit['age'], update_crsp_price=data_crit['annual_update'], start_data=data_crit['start_date'], start_factor=data_crit['start_date'])

    # Retrieve live data
    live_retrieve.exec_data()
    # Create factor data
    live_retrieve.exec_factor()
    # Get adj factor data
    live_retrieve.exec_adj_factor()

    # Execute model training and predicting for StratMLRet
    strat_ml_ret.exec_backtest()
    strat_ml_ret.exec_live()
    # Execute model training and predicting for StratMLTrend
    start_ml_trend.exec_backtest()
    start_ml_trend.exec_live()

    # Backtest StratPortIV
    strat_port_iv.exec_backtest()
    # Backtest StratPortIM
    strat_port_im.exec_backtest()
    # Backtest StratPortID
    strat_port_id.exec_backtest()
    # Backtest StratPortIVMD
    strat_port_ivmd.exec_backtest()
    # Backtest StratTrendMLS
    strat_trend_mls.exec_backtest()
    # Backtest StratMrevETF
    strat_mrev_etf.exec_backtest()
    # Backtest StratMrevMkt
    strat_mrev_mkt.exec_backtest()

    # Log Time
    print("-"*180)
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total Time to Execute Build: {int(minutes)}:{int(seconds):02}")
    print("-"*180)

def trade():
    '''
    Note: Specify ibkr criteria in ibkr_crit.json:
                - first_day (str: "True" or "False"): First day of trading or not
                - settle_period (int): IBKR Cash Settlement Period

          In ibkr_crit.json, set "first_day" to be "True" if this is the first trade() run and "settle_period" to IBKR's cash settlement period.
          For the following days after, set "first_day" to be "False" and decrement "settle_period" by 1 continuously until "settle_period" is 1.
          This is to avoid the problem of running out of capital.

          trade() should be executed at 3:40 PM EST Daily on Monday to Friday
    '''

    # Log Time
    start_time = time.time()

    # Get strategy criteria
    strat_crit = json.load(open(get_config() / 'strat_crit.json'))

    # Get IBKR criteria
    ibkr_crit = json.load(open(get_config() / 'ibkr_crit.json'))

    # Params
    current_date = date.today().strftime('%Y-%m-%d')

    # Connect to IB
    print("Attempting to connect to IBKR TWS Workstation...")
    ibkr_server = IB()
    ibkr_server.connect(host='127.0.0.1', port=7497, clientId=1512)
    print("Connected to IBKR TWS Workstation.")

    # Create Executioners
    live_price = LivePrice(ibkr_server=ibkr_server, current_date=current_date)
    live_close = LiveClose(ibkr_server=ibkr_server, current_date=current_date)
    live_trade = LiveTrade(ibkr_server=ibkr_server, current_date=current_date, settle_period=ibkr_crit['settle_period'])

    # Create Strategies
    strat_port_iv = StratPortIV(allocate=strat_crit['port_iv']['allocate'], current_date=current_date, start_date=strat_crit['port_iv']['start_backtest'], threshold=strat_crit['port_iv']['threshold'], num_stocks=strat_crit['port_iv']['per_side'][0], window_port=21)
    strat_port_im = StratPortIM(allocate=strat_crit['port_im']['allocate'], current_date=current_date, start_date=strat_crit['port_im']['start_backtest'], threshold=strat_crit['port_im']['threshold'], num_stocks=strat_crit['port_im']['per_side'][0], window_port=21)
    strat_port_id = StratPortID(allocate=strat_crit['port_id']['allocate'], current_date=current_date, start_date=strat_crit['port_id']['start_backtest'], threshold=strat_crit['port_id']['threshold'], num_stocks=strat_crit['port_id']['per_side'][0], window_port=21)
    strat_port_ivmd = StratPortIVMD(allocate=strat_crit['port_ivmd']['allocate'], current_date=current_date, start_date=strat_crit['port_ivmd']['start_backtest'], threshold=strat_crit['port_ivmd']['threshold'], num_stocks=strat_crit['port_ivmd']['per_side'][0], window_port=21)
    strat_trend_mls = StratTrendMLS(allocate=strat_crit['trend_mls']['allocate'], current_date=current_date, start_date=strat_crit['trend_mls']['start_backtest'], threshold=strat_crit['trend_mls']['threshold'], num_stocks=strat_crit['trend_mls']['per_side'][0], window_hedge=60, window_port=252)
    strat_mrev_etf = StratMrevETF(allocate=strat_crit['mrev_etf']['allocate'], current_date=current_date, start_date=strat_crit['mrev_etf']['start_backtest'], threshold=strat_crit['mrev_etf']['threshold'], window_epsil=168, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)
    strat_mrev_mkt = StratMrevMkt(allocate=strat_crit['mrev_mkt']['allocate'], current_date=current_date, start_date=strat_crit['mrev_mkt']['start_backtest'], threshold=strat_crit['mrev_mkt']['threshold'], window_epsil=168, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)

    # Retrieve live close prices
    loop = asyncio.get_event_loop()
    loop.run_until_complete(live_price.exec_live_price())
    loop.close()

    # Parallel Strategy Execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Load Strategies
        exec_strategies = [
            executor.submit(strat_port_iv.exec_live),
            executor.submit(strat_port_im.exec_live),
            executor.submit(strat_port_id.exec_live),
            executor.submit(strat_port_ivmd.exec_live),
            executor.submit(strat_trend_mls.exec_live),
            executor.submit(strat_mrev_etf.exec_live),
            executor.submit(strat_mrev_mkt.exec_live)
        ]
        # Wait for all strategies to execute
        for future in concurrent.futures.as_completed(exec_strategies):
            future.result()

    # Close trades from previous day
    if ibkr_crit['first_day'] == "False":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(live_close.exec_close())
        loop.close()

    # Execute new trades for today
    loop = asyncio.get_event_loop()
    loop.run_until_complete(live_trade.exec_trade())
    loop.close()

    # Log Time
    print("-" * 180)
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total Time to Execute Trade: {int(minutes)}:{int(seconds):02}")
    print("-" * 180)

    # Store live price and live stock data
    live_price.exec_live_store()

    # Disconnect
    ibkr_server.disconnect()

def monitor():
    '''
    Note: Specify monitor criteria in mont_crit.json:
               - rolling_window (list[int]): List of rolling windows for Dynamic Alpha Test (rolling window)
               - full_sample (str: "True" or "False"): Full Sample Alpha Test or not

          monitor() should be executed at 4:00 PM EST Weekly on Friday
    '''

    # Get strategy criteria
    strat_crit = json.load(open(get_config() / 'strat_crit.json'))
    # Get monitor criteria
    mont_crit = json.load(open(get_config() / 'mont_crit.json'))
    
    # Create Monitors
    mont_ml_ret = MonitorStrat(strat_name='StratMLRet', strat_file='data_ml_ret_store.parquet.brotli', allocate=strat_crit['ml_ret']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_ml_ret')
    mont_ml_trend = MonitorStrat(strat_name='StratMLTrend', strat_file='data_ml_trend_store.parquet.brotli', allocate=strat_crit['ml_trend']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_ml_trend')
    mont_port_iv = MonitorStrat(strat_name='StratPortIV', strat_file='data_port_iv_store.parquet.brotli', allocate=strat_crit['port_iv']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_port_iv')
    mont_port_im = MonitorStrat(strat_name='StratPortIM', strat_file='data_port_im_store.parquet.brotli', allocate=strat_crit['port_im']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_port_im')
    mont_port_id = MonitorStrat(strat_name='StratPortID', strat_file='data_port_id_store.parquet.brotli', allocate=strat_crit['port_id']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_port_id')
    mont_port_ivmd = MonitorStrat(strat_name='StratPortIVMD', strat_file='data_port_ivmd_store.parquet.brotli', allocate=strat_crit['port_ivmd']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_port_ivmd')
    mont_trend_mls = MonitorStrat(strat_name='StratTrendMLS', strat_file='data_trend_mls_store.parquet.brotli', allocate=strat_crit['trend_mls']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_trend_mls')
    mont_mrev_etf = MonitorStrat(strat_name='StratMrevETF', strat_file='data_mrev_etf_store.parquet.brotli', allocate=strat_crit['mrev_etf']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_mrev_etf')
    mont_mrev_mkt = MonitorStrat(strat_name='StratMrevMkt', strat_file='data_mrev_mkt_store.parquet.brotli', allocate=strat_crit['mrev_mkt']['allocate'], alpha_windows=mont_crit['rolling_window'], output_path=get_live_monitor() / 'strat_mrev_mkt')
    mont_all = MonitorStrat(output_path=get_live_monitor() / 'strat_all')

    # Monitor Strategies
    mont_ml_ret.monitor_strat()
    mont_ml_trend.monitor_strat()
    mont_port_iv.monitor_strat()
    mont_port_im.monitor_strat()
    mont_port_id.monitor_strat()
    mont_port_ivmd.monitor_strat()
    mont_trend_mls.monitor_strat()
    mont_mrev_etf.monitor_strat()
    mont_mrev_mkt.monitor_strat()

    # Monitor All Strategies
    mont_all.monitor_all()

# Build
schedule.every().monday.at("00:01").do(build)
schedule.every().tuesday.at("00:01").do(build)
schedule.every().wednesday.at("00:01").do(build)
schedule.every().thursday.at("00:01").do(build)
schedule.every().friday.at("00:01").do(build)

# Trade
schedule.every().monday.at("15:40").do(trade)
schedule.every().tuesday.at("15:40").do(trade)
schedule.every().wednesday.at("15:40").do(trade)
schedule.every().thursday.at("15:40").do(trade)
schedule.every().friday.at("15:40").do(trade)

# Monitor
schedule.every().friday.at("16:00").do(monitor)

# Execute
while True:
    schedule.run_pending()
    time.sleep(1)