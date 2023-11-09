import schedule
import time
import datetime
import asyncio

from live.strategy_ml.exec_model import exec_model
from live.strategy_ml.exec_pred import exec_pred
from live.strategy_ml.exec_trade import exec_trade
from live.strategy_ml.exec_close import exec_close

# Check if current time is within the provided range
def within_time_range(start, end):
    current_time = datetime.datetime.now().time()
    return start <= current_time <= end

# Job to execute train
def daily_train():
    if within_time_range(datetime.time(0, 1), datetime.time(2, 0)):
        print("---------------------------------------------------------------------------------RUN---------------------------------------------------------------------------------------")
        print("Running daily training at: ", datetime.datetime.now())
        exec_model(threshold=6_000_000_000, update_price=False, start_data='2004-01-01', start_factor='2004-01-01', start_model='2008-01-01', tune=['optuna', 30], save_prep=True)
        exec_pred(num_stocks=50, leverage=0.5, port_opt='equal_weight', use_model=6, threshold=2_000_000_000)
        time.sleep(30)

# Job to execute trade
def daily_trade():
    if within_time_range(datetime.time(15, 40), datetime.time(15, 45)):
        print("---------------------------------------------------------------------------------RUN---------------------------------------------------------------------------------------")
        print("Running daily trade at: ", datetime.datetime.now())
        asyncio.run(exec_trade(num_stocks=50))
        # exec_close(num_stocks=50)
        time.sleep(30)

# Schedule daily train to run every day at 12:01 AM
# Schedule daily trade to run every day at 3:40 PM
while True:
    daily_train()
    daily_trade()
    time.sleep(15)





