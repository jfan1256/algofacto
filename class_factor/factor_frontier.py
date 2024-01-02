from typing import Optional, Union, List

from core.operation import *
from class_factor.factor import Factor


class FactorFrontier(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing frontier', animation=spinner_animation)
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
        columns = ['dlttq', 'xsgaq', 'atq', 'ceqq', 'saleq', 'capxy', 'xrdq', 'ppentq', 'niq', 'xintq', 'txtq', 'dpq']
        frontier = pd.read_parquet(get_parquet(self.live) / 'data_fund_raw_q.parquet.brotli', columns=columns)
        outstanding = ['outstanding']
        price_data = pd.read_parquet(get_parquet(self.live) / 'data_misc.parquet.brotli', columns=outstanding)
        ind_data = pd.read_parquet(get_parquet(self.live) / 'data_ind_fama.parquet.brotli')
        frontier = frontier.sort_index()
        price_data = price_data.sort_index()
        price_data_reindexed = price_data.reindex(frontier.index, method='ffill')
        ind_data_reindexed = ind_data.reindex(frontier.index, method='ffill')
        frontier = frontier.merge(price_data_reindexed, left_index=True, right_index=True)
        frontier = frontier.merge(ind_data_reindexed, left_index=True, right_index=True)
        frontier = get_stocks_data(frontier, self.stock)
        frontier['xsgaq'].fillna(0, inplace=True)
        frontier['YtempBM'] = np.log(frontier['outstanding'])
        frontier['tempBook'] = np.log(frontier['ceqq'])
        frontier['tempLTDebt'] = frontier['dlttq'] / frontier['atq']
        frontier['tempCapx'] = frontier['capxy'] / frontier['saleq']
        frontier['tempRD'] = frontier['xrdq'] / frontier['saleq']
        frontier['tempAdv'] = frontier['xsgaq'] / frontier['saleq']
        frontier['tempPPE'] = frontier['ppentq'] / frontier['atq']
        frontier['ebitda'] = frontier['niq'] + frontier['xintq'] + frontier['txtq'] + frontier['dpq']
        frontier['tempEBIT'] = frontier['ebitda'] / frontier['atq']

        frontier['logmefit_NS'] = np.nan
        all_dates = frontier.index.get_level_values('date').unique()

        for d in all_dates:
            mask_date = frontier.index.get_level_values('date') == d
            temp_data = frontier[mask_date]

            mask_period = (temp_data.index.get_level_values('date') <= d) & (temp_data.index.get_level_values('date') > d - pd.Timedelta(days=60*21))
            temp_data_period = temp_data[mask_period]

            X = temp_data_period[['tempBook', 'tempLTDebt', 'tempCapx', 'tempRD', 'tempAdv', 'tempPPE', 'tempEBIT', 'IndustryFama']]
            X = sm.add_constant(X)
            y = temp_data_period['YtempBM']

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)

            model = sm.OLS(y, X).fit()

            # Predict values for date d
            predictions = model.predict(sm.add_constant(temp_data[['tempBook', 'tempLTDebt', 'tempCapx', 'tempRD', 'tempAdv', 'tempPPE', 'tempEBIT', 'IndustryFama']]))

            frontier.loc[mask_date, 'logmefit_NS'] = predictions

        frontier['frontier'] = frontier['YtempBM'] - frontier['logmefit_NS']
        frontier['frontier'] = -1 * frontier['frontier']

        # Filters
        frontier.drop(frontier[(frontier['ceqq'].isna()) | (frontier['ceqq'] < 0)].index, inplace=True)
        self.factor_data = frontier[['frontier']]