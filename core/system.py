from pathlib import Path

# Get root directory
def get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent

# Get config directory
def get_config():
    return get_root_dir() / 'data_config'

# Get historical trade directory
def get_historical_trade():
    return get_root_dir() / 'trade_historical'

# Get live trade directory
def get_live_trade():
    return get_root_dir() / 'trade_live'

# Get parquet directory (historical or live)
def get_parquet(live) -> Path:
    if live:
        return get_live_trade() / 'data_parquet'
    else:
        return get_historical_trade() / 'data_parquet'

# Get factor directory (historical or live)
def get_factor(live) -> Path:
    if live:
        return get_live_trade() / 'data_factor'
    else:
        return get_historical_trade() / 'data_factor'

# Get prep directory (historical or live)
def get_prep(live) -> Path:
    if live:
        return get_live_trade() / 'data_prep'
    else:
        return get_historical_trade() / 'data_prep'

# Get large directory (historical or live)
def get_large(live) -> Path:
    if live:
        return get_live_trade() / 'data_large'
    else:
        return get_historical_trade() / 'data_large'

# Get data live directory
def get_live():
    return get_live_trade() / 'data_live'

# Get data adj directory
def get_adj():
    return get_live_trade() / 'data_adj'

# Get ml result directory (historical or live)
def get_ml_result(live, model_name) -> Path:
    if live and ('lightgbm' in model_name or 'catboost' in model_name):
        return get_live_trade() / 'strat_ml_ret' / 'result'
    elif live and ('randomforest' in model_name):
        return get_live_trade() / 'strat_ml_trend' / 'result'
    else:
        return get_historical_trade() / 'result'

# Get ml report directory (historical or live)
def get_ml_report(live, model_name) -> Path:
    if live and ('lightgbm' in model_name or 'catboost' in model_name):
        return get_live_trade() / 'strat_ml_ret' / 'report'
    elif live and ('randomforest' in model_name):
        return get_live_trade() / 'strat_ml_trend' / 'report'
    else:
        return get_historical_trade() / 'report'

# Get ml model subdirectory (historical or live)
def get_ml_result_model(live, model):
    return get_ml_result(live, model) / f'{model}'

# Get live price directory
def get_live_price():
    return get_live_trade() / 'live_price'

# Get live stock directory
def get_live_stock():
    return get_live_trade() / 'live_stock'

# Get live monitor directory
def get_live_monitor():
    return get_live_trade() / 'live_monitor'

# Get strat ml ret directory
def get_strat_ml_ret():
    return get_live_trade() / 'strat_ml_ret'

# Get strat mrev etf directory
def get_strat_mrev_etf():
    return get_live_trade() / 'strat_mrev_etf'

# Get strat port ims directory
def get_strat_port_ims():
    return get_live_trade() / 'strat_port_ims'

# Get strat port ivm directory
def get_strat_port_ivm():
    return get_live_trade() / 'strat_port_ivm'

# Get strat port iv directory
def get_strat_port_iv():
    return get_live_trade() / 'strat_port_iv'

# Get strat port im directory
def get_strat_port_im():
    return get_live_trade() / 'strat_port_im'

# Get strat port id directory
def get_strat_port_id():
    return get_live_trade() / 'strat_port_id'

# Get strat trend mls directory
def get_strat_trend_mls():
    return get_live_trade() / 'strat_trend_mls'

# Get strat mrev mkt directory
def get_strat_mrev_mkt():
    return get_live_trade() / 'strat_mrev_mkt'
