from live_trade.live_data import LiveData

from functions.utils.func import *

def exec_data(threshold, update_price, start_data):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------GET LIVE DATA----------------------------------------------------------------------------------
    print("---------------------------------------------------------------------------GET LIVE DATA----------------------------------------------------------------------------------")
    live = True
    current_date = (date.today()).strftime('%Y-%m-%d')
    total_time = time.time()

    start_time = time.time()
    live_data = LiveData(live=live, start_date=start_data, current_date=current_date)

    if update_price:
        live_data.create_crsp_price(threshold)
    live_data.create_compustat_quarterly()
    live_data.create_compustat_annual()
    live_data.create_stock_list()
    live_data.create_live_price()
    live_data.create_misc()
    live_data.create_compustat_pension()
    live_data.create_industry()
    live_data.create_macro()
    live_data.create_risk_rate()
    live_data.create_etf()

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total time to get live_trade data: {int(minutes)}:{int(seconds):02}")
    print("-" * 60)