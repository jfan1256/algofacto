from typing import Optional, Union, List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorMacro(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 live: bool = None,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 stock: Optional[Union[List[str], str]] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 join: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(live, file_name, skip, start, end, stock, batch_size, splice_size, group, join, general, window)
        # Read in IF
        IF = pd.read_csv(get_large(self.live) / 'macro' / 'fiveYearIR.csv')
        IF.columns = ['date', '5YIF']
        IF['date'] = pd.to_datetime(IF['date']).dt.to_period('M').dt.to_timestamp('M')
        IF = IF.set_index('date')
        IF = IF[~IF.index.duplicated(keep='first')]

        # Read in medianCPI
        medianCPI = pd.read_csv(get_large(self.live) / 'macro' / 'medianCPI.csv')
        medianCPI.columns = ['date', 'medCPI']
        medianCPI['date'] = pd.to_datetime(medianCPI['date']).dt.to_period('M').dt.to_timestamp('M')
        medianCPI = medianCPI.set_index('date')
        medianCPI = medianCPI[~medianCPI.index.duplicated(keep='first')]

        # Read in rIR
        rIR = pd.read_csv(get_large(self.live) / 'macro' / 'realInterestRate.csv')
        rIR.columns = ['date', 'rIR']
        rIR['date'] = pd.to_datetime(rIR['date']).dt.to_period('M').dt.to_timestamp('M')
        rIR = rIR.set_index('date')
        rIR = rIR[~rIR.index.duplicated(keep='first')]

        # Read in UR
        UR = pd.read_csv(get_large(self.live) / 'macro' / 'unemploymentRate.csv')
        UR.columns = ['date', 'UR']
        UR['date'] = pd.to_datetime(UR['date']).dt.to_period('M').dt.to_timestamp('M')
        UR = UR.set_index('date')
        UR = UR[~UR.index.duplicated(keep='first')]

        # Read in PPI
        PPI = pd.read_csv(get_large(self.live) / 'macro' / 'PPI.csv')
        PPI.columns = ['date', 'PPI']
        PPI['date'] = pd.to_datetime(PPI['date']).dt.to_period('M').dt.to_timestamp('M')
        PPI = PPI.set_index('date')
        PPI = PPI[~PPI.index.duplicated(keep='first')]

        # Read in Industry Production Index
        indProdIndex = pd.read_csv(get_large(self.live) / 'macro' / 'indProdIndex.csv')
        indProdIndex.columns = ['date', 'indProdIndex']
        indProdIndex['date'] = pd.to_datetime(indProdIndex['date']).dt.to_period('M').dt.to_timestamp('M')
        indProdIndex = indProdIndex.set_index('date')
        indProdIndex = indProdIndex[~indProdIndex.index.duplicated(keep='first')]

        # Merge all macro data together
        date = pd.read_parquet(get_parquet(self.live) / 'data_date.parquet.brotli')
        date = date.index.get_level_values('date').unique().to_frame().drop('date', axis=1)
        macro = (pd.merge(date, IF, left_index=True, right_index=True, how='left')
                 .merge(medianCPI, left_index=True, right_index=True, how='left')
                 .merge(rIR, left_index=True, right_index=True, how='left')
                 .merge(UR, left_index=True, right_index=True, how='left')
                 .merge(PPI, left_index=True, right_index=True, how='left')
                 .merge(indProdIndex, left_index=True, right_index=True, how='left'))
        factor_macro = macro.replace([np.inf, -np.inf], np.nan)
        self.factor_data = factor_macro
