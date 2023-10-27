import schedule
import time
import datetime
import asyncio

from live.strategy_ml.exec_model import exec_model
from live.strategy_ml.exec_pred import exec_pred
from live.strategy_ml.exec_trade import exec_trade
from live.strategy_ml.exec_close import exec_close

# Job to execute everything
def daily_train():
    print("Running daily training at: ", datetime.datetime.now())
    exec_model(update_price=True, start_data='2004-01-01', start_factor='2004-01-01', start_model='2008-01-01')
    exec_pred(num_stocks=50, leverage=0.5, port_opt='equal_weight')
    # exec_trade(num_stocks=50)

def daily_trade():
    print("Running daily trade at: ", datetime.datetime.now())
    asyncio.run(exec_trade(num_stocks=50))
    exec_close(num_stocks=50)

# Schedule daily train to run every day at 12:01 AM
schedule.every().day.at("00:01").do(daily_train)
# Schedule daily trade to run every day at 3:40 PM
schedule.every().day.at("15:40").do(daily_trade)

while True:
    # Get the current time
    current_time = datetime.datetime.now().time()
    # Check if the current time is between 12:01 AM and 2:00 AM or between 3:40 PM and 5:00 PM
    if (datetime.time(hour=0, minute=1) <= current_time <= datetime.time(hour=2, minute=0)) or \
       (datetime.time(hour=15, minute=40) <= current_time <= datetime.time(hour=15, minute=45)):
        schedule.run_pending()
    time.sleep(60)





