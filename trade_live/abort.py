import os
import asyncio
import shutil

from ib_insync import *
from core.operation import *

from class_live.live_stop import LiveStop

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------STOP, RESET------------------------------------------------------------------------------
def stop():
    '''
    Note: Closes all actively open positions in portfolio

          stop() should be executed whenever the portfolio blows up
    '''

    # Connect to IB
    print("Attempting to connect to IBKR TWS Workstation...")
    ibkr_server = IB()
    ibkr_server.connect(host='127.0.0.1', port=7497, clientId=1512)
    print("Connected to IBKR TWS Workstation")

    # Create Executioners
    live_stop = LiveStop(ibkr_server=ibkr_server, )

    # Execute stop orders
    loop = asyncio.get_event_loop()
    loop.run_until_complete(live_stop.exec_stop())

    # Disconnect
    ibkr_server.disconnect()
    loop.close()

def reset():
    '''
    Note: Deletes all data, prices, and reports in trade_live directory's subdirectories

          reset() should be executed at caution
    '''

    # Reset Logic
    print("-"*60 + "\nReseting...")
    for item in os.listdir(get_live_trade()):
        item_path = os.path.join(get_live_trade(), item)

        # Skip data_large reset
        if item=='data_large' or item=='data_parquet':
            continue

        # Remove data in live_monitor
        elif item=='live_monitor':
            for item in os.listdir(item_path):
                item_path = os.path.join(item_path, item)
                if os.path.isdir(item_path) and item.startswith('strat_'):
                    for sub_item in os.listdir(item_path):
                        sub_item_path = os.path.join(item_path, sub_item)
                        for file in os.listdir(sub_item_path):
                            file_path = os.path.join(sub_item_path, file)
                            if os.path.isfile(file_path) and file.endswith('.parquet.brotli'):
                                os.remove(file_path)

        # Remove data in directories that start with 'strat_'
        elif item.startswith('strat_'):
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                # Remove data in subdirectories that are in ['report', 'result', 'data']
                if os.path.isdir(sub_item_path) and (sub_item in ['report', 'result', 'data']):
                    for file in os.listdir(sub_item_path):
                        file_path = os.path.join(sub_item_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

        # Remove data in directories that start with 'live_', or 'data_'
        elif item.startswith('live_') or item.startswith('data_'):
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                os.remove(file_path)

    print("Finished\n" + "-"*60)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------TIME TO ABORT--------------------------------------------------------------------------------
# Stop
stop()

# Reset
reset()