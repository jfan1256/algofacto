class Strategy:
    def __init__(self,
                 allocate=None,
                 current_date=None,
                 threshold=None):
        '''
        allocate (float): Percentage of capital to allocate for this strategy
        current_date (str: YYYY-MM-DD): Current date (this will be used as the end date for model training)
        threshold (int): Market cap threshold to determine whether a stock is buyable/shortable or not
        '''

        self.allocate = allocate
        self.current_date = current_date
        self.threshold = threshold

    # Strategies will inherit this function and override it with their specific functionality
    def exec_backtest(self):
        return None

    # Strategies will inherit this function and override it with their specific functionality
    def exec_live(self):
        return None
