class OrderCounter:
    def __init__(self):
        self.new_order_count = 0

    def new_order_event_handler(self, trade):
        self.new_order_count += 1
        print(f"New order for {trade.contract.symbol} with order id {trade.order.orderId} has been placed.")