import asyncio

from ib_insync import *

from core.operation import *

from trade_live.class_live.live_create import LiveCreate
from trade_live.class_live.live_price import LivePrice
from trade_live.class_live.live_close import LiveClose
from trade_live.class_live.live_trade import LiveTrade

from trade_live.strat_ml_trend.strat_ml_trend import StratMLTrend
from trade_live.strat_ml_ret.strat_ml_ret import StratMLRet
from trade_live.strat_port_iv.strat_port_iv import StratPortIV
from trade_live.strat_port_im.strat_port_im import StratPortIM
from trade_live.strat_port_id.strat_port_id import StratPortID
from trade_live.strat_port_ivmd.strat_port_ivmd import StratPortIVMD
from trade_live.strat_trend_mls.strat_trend_mls import StratTrendMLS
from trade_live.strat_mrev_etf.strat_mrev_etf import StratMrevETF
from trade_live.strat_mrev_mkt.strat_mrev_mkt import StratMrevMkt

from trade_live.class_monitor.monitor_strat import MonitorStrat

def build():
    # Get strategy criteria
    strat_crit = json.load(open(get_config() / 'strat_crit.json'))
    # Get data criteria
    data_crit = json.load(open(get_config() / 'data_crit.json'))
        
    # Params
    start_date = '2008-01-01'
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

    # Execute model training and predicting for StratMLRet
    strat_ml_ret.exec_ml_ret_model()
    strat_ml_ret.exec_ml_ret_pred()
    # Execute model training and predicting for StratMLTrend
    start_ml_trend.exec_ml_trend_model()
    start_ml_trend.exec_ml_trend_model()

    # Backtest StratPortIV
    strat_port_iv.backtest_port_iv()
    # Backtest StratPortIM
    strat_port_im.backtest_port_im()
    # Backtest StratPortID
    strat_port_id.backtest_port_id()
    # Backtest StratPortIVMD
    strat_port_ivmd.backtest_port_ivmd()
    # Backtest StratTrendMLS
    strat_trend_mls.backtest_trend_mls()
    # Backtest StratMrevETF
    strat_mrev_etf.backtest_mrev_etf()
    # Backtest StratMrevMkt
    strat_mrev_mkt.backtest_mrev_mkt()

def trade():
    # Get strategy criteria
    strat_crit = json.load(open(get_config() / 'strat_crit.json'))

    # Params
    start_date = '2008-01-01'
    current_date = date.today().strftime('%Y-%m-%d')

    # Connect to IB
    print("Attempting to connect to IBKR TWS Workstation...")
    ibkr_server = IB()
    ibkr_server.connect(host='127.0.0.1', port=7497, clientId=1512)
    print("Connected to IBKR TWS Workstation.")

    # Create Executioners
    live_price = LivePrice(ibkr_server=ibkr_server, current_date=current_date)
    live_close = LiveClose(ibkr_server=ibkr_server, current_date=current_date)
    live_trade = LiveTrade(ibkr_server=ibkr_server, current_date=current_date)

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

    # Execute StratPortIV
    strat_port_iv.exec_port_iv()
    # Execute StratPortIM
    strat_port_im.exec_port_im()
    # Execute StratPortID
    strat_port_id.exec_port_id()
    # Execute StratPortIVMD
    strat_port_ivmd.exec_port_ivmd()
    # Execute StratTrendMLS
    strat_trend_mls.exec_trend_mls()
    # Execute StratMrevETF
    strat_mrev_etf.exec_mrev_etf()
    # Execute StratMrevMkt
    strat_mrev_mkt.exec_mrev_mkt()

    # Execute Trades
    live_close.exec_close()
    live_trade.exec_trade()

    # Store live price data
    live_price.exec_live_store()

def monitor():
    # Params
    alpha_windows = None

    # Create Monitors
    mont_ml_ret = MonitorStrat(strat_name='StratMLRet', strat_csv='trade_stock_ml_ret.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_ml_ret')
    mont_ml_trend = MonitorStrat(strat_name='StratMLTrend', strat_csv='trade_stock_ml_trend.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_ml_trend')
    mont_port_iv = MonitorStrat(strat_name='StratPortIV', strat_csv='trade_stock_port_iv.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_port_iv')
    mont_port_im = MonitorStrat(strat_name='StratPortIM', strat_csv='trade_stock_port_im.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_port_im')
    mont_port_id = MonitorStrat(strat_name='StratPortID', strat_csv='trade_stock_port_id.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_port_id')
    mont_port_ivmd = MonitorStrat(strat_name='StratPortIVMD', strat_csv='trade_stock_port_ivmd.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_port_ivmd')
    mont_trend_mls = MonitorStrat(strat_name='StratTrendMLS', strat_csv='trade_stock_ml_ret.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_trend_mls')
    mont_mrev_etf = MonitorStrat(strat_name='StratMrevETF', strat_csv='trade_stock_mrev_etf.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_mrev_etf')
    mont_mrev_mkt = MonitorStrat(strat_name='StratMrevMkt', strat_csv='trade_stock_mrev_mkt.csv', allocate=0.5, alpha_windows=alpha_windows, output_path=get_live_monitor() / 'strat_mrev_mkt')
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









