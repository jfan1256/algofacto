import schedule
import time
import datetime

from live.strategy_ml.exec_model import exec_model
from live.strategy_ml.exec_pred import exec_pred
from live.strategy_ml.exec_trade import exec_trade

# Define the job to execute your functions in the desired order
def daily_execution():
    print("-"*120)
    print("Running daily execution at: ", datetime.datetime.now())
    exec_model()
    exec_pred()
    exec_trade()

# Run every day at 12:01 AM
schedule.every().day.at("00:01").do(daily_execution)

while True:
    schedule.run_pending()
    time.sleep(1)