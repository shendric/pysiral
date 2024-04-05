# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 17:33:45 2016

@author: Stefan Hendricks

This module is dedicatet to convert between different time standards
"""

import time
import inspect
from loguru import logger
from datetime import datetime

from dateutil.relativedelta import relativedelta


def debug_timer(message: str = None):
    """
    Decorator function that logs the time in seconds for a function/method
    if python is run in debug mode with environment variable set PYTHON_DEBUG_MODE=1.
    (only the log level is set to DEBUG)

    :param message: Message for the log

    :return: decorated function
    """
    def decorated_func(func):
        def wrapped_func(*args, **kwargs):
            stack = inspect.stack()
            caller_obj = stack[1][0].f_locals["self"]
            caller = caller_obj if message is None else message
            t0 = time.time()
            return_value = func(*args, **kwargs)
            t1 = time.time()
            elapsed_seconds = t1 - t0
            logger.debug(f"{caller} run in {elapsed_seconds:.3f} seconds ({get_duration(elapsed_seconds)})")
            return return_value
        # Preserve function metadata (especially annotations) of the wrapped function
        wrapped_func.__doc__ = func.__doc__
        wrapped_func.__name__ = func.__name__
        wrapped_func.__annotations__ = func.__annotations__
        return wrapped_func
    return decorated_func


def get_duration(elapsed_seconds, fmt="%H:%M:%S"):
    # Example time
    datum = datetime(1900, 1, 1)
    duration = datum + relativedelta(seconds=elapsed_seconds)
    return duration.strftime(fmt)


class StopWatch(object):

    def __init__(self):
        self.t0 = None
        self.t1 = None
        self.reset()

    def reset(self):
        self.t0 = None
        self.t1 = None

    def start(self):
        self.t0 = time.time()
        return self

    def stop(self):
        self.t1 = time.time()

    def get_seconds(self):
        return self.t1 - self.t0

    def get_duration(self, fmt="%H:%M:%S"):
        # Example time
        datum = datetime(1900, 1, 1)
        elapsed_seconds = self.t1 - self.t0
        duration = datum + relativedelta(seconds=elapsed_seconds)
        return duration.strftime(fmt)
