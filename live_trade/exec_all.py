import asyncio

from ib_insync import *
from functions.utils.func import *

from live_trade.live_create import LiveCreate
from live_trade.live_price import LivePrice
from live_trade.live_close import LiveClose
from live_trade.live_trade import LiveTrade

from live_trade.strat_ml_trend.strat_ml_trend import StratMLTrend
from live_trade.strat_ml_ret.strat_ml_ret import StratMLRet
from live_trade.strat_port_iv.strat_port_iv import StratPortIV
from live_trade.strat_port_im.strat_port_im import StratPortIM
from live_trade.strat_port_id.strat_port_id import StratPortID
from live_trade.strat_port_ivmd.strat_port_ivmd import StratPortIVMD
from live_trade.strat_trend_mls.strat_trend_mls import StratTrendMLS
from live_trade.strat_mrev_etf.strat_mrev_etf import StratMrevETF
from live_trade.strat_mrev_mkt.strat_mrev_mkt import StratMrevMkt

from live_trade.exec_trade_ml import exec_ml_ret_trade
from live_trade.exec_close_ml import exec_ml_ret_close
from live_trade.strat_port_ims.exec_port_ims import exec_port_ims_data, exec_port_ims_trade
from live_trade.exec_close_ims import exec_port_ims_close
from outdated.exec_mrev_etf import exec_mrev_etf_trade, exec_mrev_etf_data
from live_trade.exec_close_mrev import exec_mrev_etf_close


def daily_train():
    # Get current date
    current_date = date.today().strftime('%Y-%m-%d')

    # Create Strategies
    strat_ml_ret = StratMLRet(allocate=0.5, current_date=current_date, start_model='2008-01-01', threshold=2_000_000_000, num_stocks=50, leverage=0.5, port_opt='equal_weight', use_top=6)
    start_ml_trend = StratMLTrend(allocate=0.5, current_date=current_date, start_model='2008-01-01', threshold=2_000_000_000, num_stocks=50, leverage=0.5, port_opt='equal_weight', use_top=1)
    strat_port_iv = StratPortIV(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_port_im = StratPortIM(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_port_id = StratPortID(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_port_ivmd = StratPortIVMD(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_trend_mls = StratTrendMLS(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=50, window_hedge=60, window_port=252)
    strat_mrev_etf = StratMrevETF(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=50, window_epsil=168, window_port=21, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)
    strat_mrev_mkt = StratMrevMkt(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=50, window_epsil=168, window_port=21, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)

    # Retrieve live data and create factor data
    live_retrieve = LiveCreate(current_date=current_date, threshold=6_000_000_000, set_length=3, update_crsp_price=False, start_data='2004-01-01', start_factor='2004-01-01')
    live_retrieve.exec_data()
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





    # Get data for Invport Strategy
    exec_port_ims_data(window=3, scale=10, start_date='2005-01-01')
    # Get data for Mrev ETF Strategy
    exec_mrev_etf_data(window=168, threshold=2_000_000_000)


def daily_trade():
    # Get current date
    current_date = date.today().strftime('%Y-%m-%d')

    # Connect to IB
    print("Attempting to connect to IBKR TWS Workstation...")
    ibkr_server = IB()
    ibkr_server.connect(host='127.0.0.1', port=7497, clientId=1512)
    print("Connected to IBKR TWS Workstation.")

    # Retrieve live close prices
    loop = asyncio.get_event_loop()
    live_price = LivePrice(ibkr_server=ibkr_server, current_date=current_date)
    loop.run_until_complete(live_price.exec_live_price())

    # Create Executioners
    live_close = LiveClose(ibkr_server=ibkr_server, current_date=current_date)
    live_trade = LiveTrade(ibkr_server=ibkr_server, current_date=current_date)

    # Create Strategies
    strat_port_iv = StratPortIV(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_port_im = StratPortIM(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_port_id = StratPortID(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_port_ivmd = StratPortIVMD(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=25, window=21)
    strat_trend_mls = StratTrendMLS(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=50, window_hedge=60, window_port=252)
    strat_mrev_etf = StratMrevETF(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=50, window_epsil=168, window_port=21, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)
    strat_mrev_mkt = StratMrevMkt(allocate=0.5, current_date=current_date, start_date='2008-01-01', threshold=2_000_000_000, num_stocks=50, window_epsil=168, window_port=21, sbo=0.85, sso=0.85, sbc=0.25, ssc=0.25)

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





    # Execute trades for ML Strategy
    exec_ml_ret_close()
    asyncio.run(exec_ml_ret_trade(num_stocks=50, settlement=3, capital=0.25))

    # Execute trades for Invport Strategy
    exec_port_ims_close()
    exec_port_ims_trade(scale=10, window=3, settlement=3, capital=0.50)
    # Execute Mrev ETF Strategy
    exec_mrev_etf_close()
    exec_mrev_etf_trade(window=168, threshold=2_000_000_000, settlement=3, capital=0.25)





