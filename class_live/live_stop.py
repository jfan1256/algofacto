import asyncio

from core.operation import *

from class_order.order_ibkr import OrderIBKR
from class_live.live_callback import OrderCounter

class LiveStop:
    def __init__(self,
                 ibkr_server
                 ):

        '''
        ibkr_server (ib_sync server): IBKR IB Sync server
        '''

        self.ibkr_server = ibkr_server

    # Execute reset orders
    async def exec_stop(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------EXECUTE STOP ORDERS------------------------------------------------------------------------------
        print("-------------------------------------------------------------------------EXECUTE STOP ORDERS------------------------------------------------------------------------------")
        # Create OrderIBKR Class
        order_ibkr = OrderIBKR(ibkr_server=self.ibkr_server)

        # Fetch portfolio
        portfolio = self.ibkr_server.portfolio()

        # Params (Note: IBKR has an Order Limit of 50 per second)
        tasks = []
        batch_size = 50
        order_num = 1
        # Subscribe the class method to the newOrderEvent
        order_counter = OrderCounter()
        self.ibkr_server.orderStatusEvent += order_counter.order_status_event_handler

        # Execute Close Orders for each position in Portfolio
        for item in portfolio:
            symbol = item.contract.symbol
            action = 'SELL' if item.position > 0 else 'BUY'

            # Create close order task
            task = order_ibkr._execute_close(symbol=symbol, action=action, order_num=order_num, instant=True)
            tasks.append(task)
            order_num += 1

            # Execute Batch
            if order_num % batch_size == 0:
                batch_num = int(order_num / batch_size)
                print(f"----------------------------------------------------------------BATCH: {batch_num}----------------------------------------------------------------------------------")
                await asyncio.gather(*tasks)
                tasks = []
                # Avoid Order Hit Rate Limit
                time.sleep(2)

        # Ensure any excess tasks are completed
        if tasks:
            print(f"----------------------------------------------------------------BATCH: EXCESS------------------------------------------------------------------------------------------")
            # Wait for current batch of tasks to complete
            await asyncio.gather(*tasks)
            # Ensure the event handler has time to transmit all orders
            await asyncio.sleep(5)

        # Display Order Counts
        print("----------------------------------------------------------------------ORDER METRIC------------------------------------------------------------------------------------------")
        print(f"Total positions closed: {len(portfolio)}")
        order_counter.display_metric()