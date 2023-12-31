import schedule
import datetime
import asyncio

from ib_insync import *
from functions.utils.func import *

from live_trade.live_create import LiveCreate
from live_trade.live_price import LivePrice

from live_trade.strat_ml_trend.strat_ml_trend import StrategyMLTrend
from live_trade.strat_ml_ret.strat_ml_ret import StrategyMLRet

from live_trade.exec_trade_ml import exec_ml_ret_trade
from live_trade.exec_close_ml import exec_ml_ret_close
from live_trade.strat_port_ims.exec_port_ims import exec_port_ims_data, exec_port_ims_trade
from live_trade.exec_close_ims import exec_port_ims_close
from live_trade.strat_mrev_etf.exec_mrev_etf import exec_mrev_etf_trade, exec_mrev_etf_data
from live_trade.exec_close_mrev import exec_mrev_etf_close


# Check if current time is within the provided range
def within_time_range(start, end):
    current_time = datetime.datetime.now().time()
    return start <= current_time <= end

# Job to execute train
def daily_train():
    current_date = date.today().strftime('%Y-%m-%d')

    live_retrieve = LiveCreate(current_date=current_date, threshold=6_000_000_000, set_length=3,
                               update_crsp_price=False, start_data='2004-01-01', start_factor='2004-01-01')

    strat_ml_ret = StrategyMLRet(allocate=0.5, current_date=current_date, start_model='2008-01-01',
                                   threshold=2_000_000_000, num_stocks=50, leverage=0.5, port_opt='equal_weight', use_top=6)

    start_ml_trend = StrategyMLTrend(allocate=0.5, current_date=current_date, start_model='2008-01-01',
                                     threshold=2_000_000_000, num_stocks=50, leverage=0.5, port_opt='equal_weight', use_top=1)

    print("Running daily training at: ", datetime.datetime.now())
    # Retrieve live data and create factor data
    live_retrieve.exec_data()
    live_retrieve.exec_factor()
    # Get data for Invport Strategy
    exec_port_ims_data(window=3, scale=10, start_date='2005-01-01')
    # Get data for Mrev ETF Strategy
    exec_mrev_etf_data(window=168, threshold=2_000_000_000)
    # Execute model training and predicting for StrategyMLRet
    strat_ml_ret.exec_ml_ret_model()
    strat_ml_ret.exec_ml_ret_pred()
    # Execute model training and predicting for StrategyMLTrend
    start_ml_trend.exec_ml_trend_model()
    start_ml_trend.exec_ml_trend_model()

# Job to execute trade
def daily_trade():
    print("Running daily trade at: ", datetime.datetime.now())
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

    # Execute trades for ML Strategy
    exec_ml_ret_close()
    asyncio.run(exec_ml_ret_trade(num_stocks=50, settlement=3, capital=0.25))
    # Execute trades for Invport Strategy
    exec_port_ims_close()
    exec_port_ims_trade(scale=10, window=3, settlement=3, capital=0.50)
    # Execute Mrev ETF Strategy
    exec_mrev_etf_close()
    exec_mrev_etf_trade(window=168, threshold=2_000_000_000, settlement=3, capital=0.25)

# Schedule daily train to run every day at 12:01 AM
# Schedule daily trade to run every day at 3:40 PM
schedule.every().day.at('15:40').do(daily_trade)
while True:
    daily_train()
    schedule.run_pending()
    time.sleep(15)





