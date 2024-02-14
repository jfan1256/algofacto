class LiveCallback:
    def __init__(self):
        self.processed_orders = set()
        self.order_count = 0
        self.failed_orders = []
        self.total_count = 0

    def order_status_event_handler(self, trade):
        # Extract the necessary information from the Trade object
        symbol = trade.contract.symbol
        orderId = trade.order.orderId
        status = trade.orderStatus.status

        # Check if the order was successfully transmitted
        if orderId not in self.processed_orders:
            if status in ["Submitted", "PreSubmitted", "Filled"]:
                self.order_count += 1
            else:
                self.failed_orders.append(symbol)
            self.total_count += 1
            self.processed_orders.add(orderId)

    def display_metric(self):
        print(f"Succeeded Orders: {self.order_count} / {self.total_count}")
        print(f"Failed Orders: {len(self.failed_orders)} / {self.total_count}")
        print(f"    Cause: num_share=0, which means there is not enough capital to buy/sell these stocks below")
        print(f"    Symbols: {', '.join(self.failed_orders)}")
