import sys
import threading
import time

from pathlib import Path
from functools import wraps


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

def print_data_shape(self, *args, **kwargs):
    print('Shape: ' + str(self.data.shape))

def show_processing_animation(animation: any, message_func=None, post_func=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the message
            if callable(message_func):
                message = message_func(*args, **kwargs)
            else:
                message = "Processing..."

            stop_event = threading.Event()

            # Start the ellipsis animation in a thread
            t = threading.Thread(target=animation, args=(message, stop_event))
            t.start()

            # Now, run the actual function to do the work
            result = func(*args, **kwargs)

            # Stop the animation
            stop_event.set()
            t.join()

            sys.stdout.write('\033[92m✔\033[0m\n')
            sys.stdout.flush()
            if post_func:
                post_func(*args, **kwargs)

            return result

        return wrapper

    return decorator

def spinner_animation(message: str, stop_event: threading.Event):
    line_position = 60
    num_dashes = line_position - len(message) - 1
    sys.stdout.write(f'{message} {"-" * num_dashes} ')

    sys.stdout.flush()
    spinner_chars = ['|', '/', '-', '\\']
    while not stop_event.is_set():
        for char in spinner_chars:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.2)
            if stop_event.is_set():
                break
            sys.stdout.write('\b')
    sys.stdout.write('\b| ')
