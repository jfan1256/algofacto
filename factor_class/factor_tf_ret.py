from typing import List

from functions.utils.func import *
from factor_class.factor import Factor


class FactorTFRet(Factor):
    @timebudget
    @show_processing_animation(message_func=lambda self, *args, **kwargs: f'Initializing data', animation=spinner_animation)
    def __init__(self,
                 file_name: str = None,
                 skip: bool = None,
                 start: str = None,
                 end: str = None,
                 ticker: List[str] = None,
                 batch_size: int = None,
                 splice_size: int = None,
                 group: str = None,
                 general: bool = False,
                 window: int = None):
        super().__init__(file_name, skip, start, end, ticker, batch_size, splice_size, group, general, window)
        self.factor_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_price.parquet.brotli')
        self.etf_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_etf.parquet.brotli')
        self.fama_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_fama.parquet.brotli')
        self.pca_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_pca_ret.parquet.brotli')
        self.macro_data = pd.read_parquet(get_load_data_parquet_dir() / 'data_macro.parquet.brotli')
        self.dividend_data = pd.read_parquet(get_factor_data_dir() / 'factor_dividend.parquet.brotli')
        self.all_rf = pd.concat([self.etf_data, self.fama_data, self.pca_data, self.macro_data], axis=1)

    @ray.remote
    def function(self, splice_data):
        T = [1, 6, 30]
        splice_data = create_return(splice_data, windows=T)
        splice_data = splice_data.fillna(0)

        for t in T:
            ret = f'RET_{t:02}'
            # if window size is too big it can create an index out of bound error (took me 3 hours to debug this error!!!)
            windows = [60]
            name = f'{t:02}_TF_RET'
            for window in windows:
                # Rolling LR to calculate beta coefficients + predictions + alpha + epilson
                betas = []
                # Get factor columns and create returns
                for ticker, df in splice_data.groupby('ticker', group_keys=False):
                    try:
                        extra = get_ticker_data(self.dividend_data, ticker).reset_index('ticker').drop('ticker', axis=1)
                        all_rf = pd.concat([self.all_rf, extra], axis=1)
                        all_rf = all_rf.loc[self.start:self.end]
                        all_rf.fillna(0)
                    except:
                        all_rf = self.all_rf.loc[self.start:self.end]
                        all_rf.fillna(0)

                    # Execute Rolling PCA
                    window_size = 60
                    num_components = 5
                    pca_rf = rolling_pca(data=all_rf, window_size=window_size, num_components=num_components, name='TF')
                    # Add risk-free rate
                    pca_rf = pca_rf.merge(self.fama_data['RF'], left_index=True, right_index=True, how='left')
                    pca_rf = pca_rf.loc[~pca_rf.index.duplicated(keep='first')]
                    factor_col = pca_rf.columns[:-1]

                    model_data = df[[ret]].merge(pca_rf, on='date').dropna()
                    model_data[ret] -= model_data.RF
                    rolling_ols = RollingOLS(endog=model_data[ret], exog=sm.add_constant(model_data[factor_col]), window=window)
                    factor_model = rolling_ols.fit(params_only=True).params.rename(columns={'const': 'ALPHA'})

                    # Compute predictions of ticker's return
                    alpha = factor_model['ALPHA']
                    beta_coef = factor_model[factor_col]
                    factor_ret = model_data[factor_col]
                    ticker_ret = df.reset_index('ticker').drop(columns='ticker', axis=1)[ret]

                    predictions = []
                    epsilons = []
                    for index, row in factor_ret.iterrows():
                        prediction = row @ beta_coef.loc[index] + alpha.loc[index]
                        epsilon = ticker_ret.loc[index] - prediction
                        predictions.append(prediction)
                        epsilons.append(epsilon)

                    result = factor_model.assign(ticker=ticker).set_index('ticker', append=True).swaplevel()
                    result['PRED'] = predictions
                    result['EPSIL'] = epsilons
                    result['EPSIL'] = result['EPSIL'].rolling(window=window).sum() / result['EPSIL'].rolling(window=window).std()
                    betas.append(result)

                betas = pd.concat(betas).rename(columns=lambda x: f'{x}_{name}_{window:02}')
                splice_data = splice_data.join(betas)

        return splice_data
