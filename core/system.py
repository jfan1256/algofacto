from pathlib import Path

def get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def get_historical_trade():
    return get_root_dir() / 'trade_historical'

def get_live_trade():
    return get_root_dir() / 'trade_live'

def get_parquet(live) -> Path:
    if live:
        return get_live_trade() / 'parquet'
    else:
        return get_historical_trade() / 'parquet'

def get_factor(live) -> Path:
    if live:
        return get_live_trade() / 'factor'
    else:
        return get_historical_trade() / 'factor'

def get_prep(live) -> Path:
    if live:
        return get_live_trade() / 'prep'
    else:
        return get_historical_trade() / 'prep'

def get_large(live) -> Path:
    if live:
        return get_live_trade() / 'large'
    else:
        return get_historical_trade() / 'large'

def get_ml_result(live, model_name) -> Path:
    if live and ('lightgbm' in model_name or 'catboost' in model_name):
        return get_live_trade() / 'strat_ml_ret' / 'result'
    elif live and ('randomforest' in model_name):
        return get_live_trade() / 'strat_ml_trend' / 'result'
    else:
        return get_historical_trade() / 'result'

def get_ml_report(live, model_name) -> Path:
    if live and ('lightgbm' in model_name or 'catboost' in model_name):
        return get_live_trade() / 'strat_ml_ret' / 'report'
    elif live and ('randomforest' in model_name):
        return get_live_trade() / 'strat_ml_trend' / 'report'
    else:
        return get_historical_trade() / 'report'

def get_ml_result_model(live, model):
    return get_ml_result(live, model) / f'{model}'

def get_live_price():
    return get_live_trade() / 'live_price'

def get_live_stock():
    return get_live_trade() / 'stock'

def get_strat_ml_ret():
    return get_live_trade() / 'strat_ml_ret'

def get_strat_mrev_etf():
    return get_live_trade() / 'strat_mrev_etf'

def get_strat_port_ims():
    return get_live_trade() / 'strat_port_ims'

def get_strat_port_ivmd():
    return get_live_trade() / 'strat_port_ivmd'

def get_strat_port_iv():
    return get_live_trade() / 'strat_port_iv'

def get_strat_port_im():
    return get_live_trade() / 'strat_port_im'

def get_strat_port_id():
    return get_live_trade() / 'strat_port_id'

def get_strat_trend_mls():
    return get_live_trade() / 'strat_trend_mls'

def get_strat_mrev_mkt():
    return get_live_trade() / 'strat_mrev_mkt'
