import schedule
import time
import datetime
import asyncio

from live.strategy_ml.exec_ml_model import exec_ml_model
from live.strategy_ml.exec_ml_pred import exec_ml_pred
from live.strategy_ml.exec_ml_trade import exec_ml_trade
from live.strategy_ml.exec_ml_close import exec_ml_close
from live.strategy_port.exec_invport import exec_invport_data, exec_invport_trade

# Check if current time is within the provided range
def within_time_range(start, end):
    current_time = datetime.datetime.now().time()
    return start <= current_time <= end

# Job to execute train
def daily_train():
    if within_time_range(datetime.time(0, 1), datetime.time(2, 0)):
        print("---------------------------------------------------------------------------------RUN---------------------------------------------------------------------------------------")
        print("Running daily training at: ", datetime.datetime.now())
        # Get data for Invport Strategy
        exec_invport_data(window=3, scale=10, start_date='2005-01-01')
        # Get, train, predict ML Strategy
        exec_ml_model(threshold=6_000_000_000, update_price=False, start_data='2004-01-01', start_factor='2004-01-01', start_model='2008-01-01', tune=['optuna', 30], save_prep=True)
        exec_ml_pred(threshold=2_000_000_000, num_stocks=50, leverage=0.5, port_opt='equal_weight', use_model=6)

# Job to execute trade
def daily_trade():
    print("---------------------------------------------------------------------------------RUN---------------------------------------------------------------------------------------")
    print("Running daily trade at: ", datetime.datetime.now())
    # Execute trades for ML Strategy
    exec_ml_close(num_stocks=50)
    asyncio.run(exec_ml_trade(num_stocks=50, settlement=3))
    # Execute trades for Invport Strategy
    exec_invport_trade(window=3, scale=10)

# Schedule daily train to run every day at 12:01 AM
# Schedule daily trade to run every day at 3:40 PM
schedule.every().day.at('15:40').do(daily_trade)
while True:
    daily_train()
    schedule.run_pending()
    time.sleep(15)





