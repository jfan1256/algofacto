import sys
import threading
import time

from functools import wraps

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