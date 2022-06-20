import errno
import os
import signal
import functools

class TimeoutError(Exception):
    pass

import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


        
# def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
#     def decorator(func):
#         def _handle_timeout(signum, frame):
#             raise TimeoutError(error_message)

#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             signal.signal(signal.SIGALRM, _handle_timeout)
#             signal.alarm(seconds)
#             try:
#                 result = func(*args, **kwargs)
#             finally:
#                 signal.alarm(0)
#             return result

#         return wrapper

#     return decorator