from functools import wraps
from time import perf_counter_ns
from contextlib import contextmanager
import inspect
import unittest.mock as mock
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic
import os
import matplotlib.style as mplstyle
import logging
from colorama import Fore, Style
import pathlib
from pathlib import Path

from utils.setup_notebook import source_code_path_is_from_notebook, get_notebook_dir


def set_default_log_func(log_func):
    global _default_log_func
    _default_log_func = log_func


def format_time(elapsed_ns: int):
    """
    Helper function to format the elapsed time in the appropriate unit
    (seconds, milliseconds, microseconds, or nanoseconds).
    """
    if elapsed_ns >= 1_000_000_000:  # More than or equal to 1 second
        elapsed = elapsed_ns / 1_000_000_000
        unit = "s"
        color = Fore.RED
    elif elapsed_ns >= 1_000_000:  # More than or equal to 1 millisecond
        elapsed = elapsed_ns / 1_000_000
        unit = "ms"
        color = Fore.BLUE
    elif elapsed_ns >= 1_000:  # More than or equal to 1 microsecond
        elapsed = elapsed_ns / 1_000
        unit = "µs"
        color = Fore.GREEN
    else:
        elapsed = elapsed_ns
        unit = "ns"
        color = Fore.LIGHTBLACK_EX
    return elapsed, unit, color


def _get_debug_info_string(source_file: str, line_number: int):
    debug_info = ""
    if source_code_path_is_from_notebook(source_file):
        debug_info = (
            f"{Fore.LIGHTBLACK_EX}(notebook_cell:{line_number}){Style.RESET_ALL}"
        )
    else:
        if get_notebook_dir() is not None:
            source_file = Path(source_file).relative_to(get_notebook_dir())

        debug_info = (
            f"{Fore.LIGHTBLACK_EX}({source_file}:{line_number}){Style.RESET_ALL}"
        )
    return debug_info


def time_function(log_before: bool = True, log_func=None):
    if log_func is None:
        log_func = _default_log_func

    def decorator(f):
        @wraps(f)
        def wrap(*args, **kw):
            # Use inspect to get the source file and line number
            source_file = inspect.getfile(f)
            _, starting_line = inspect.getsourcelines(f)

            debug_info = _get_debug_info_string(source_file, starting_line)

            if log_before:
                log_func(
                    f"{Fore.BLACK}func: {f.__name__!r} {Style.RESET_ALL}called {debug_info}"
                )

            ts = perf_counter_ns()
            result = f(*args, **kw)
            te = perf_counter_ns()

            elapsed_ns = te - ts
            elapsed, unit, color = format_time(elapsed_ns)

            # Print with optional colored output if colorama is available
            log_func(
                f"{Fore.BLACK}func: {f.__name__!r} {Style.RESET_ALL}took: "
                f"{color}{elapsed:,.4f} {unit}{Style.RESET_ALL} "
                f"{debug_info}"
            )

            return result

        return wrap

    return decorator


@contextmanager
def time_line(description: str = "Code", log_before: bool = True, log_func=None):
    # Get the current stack frame to determine where the with statement was called
    frame_info = inspect.stack()[2]

    source_file = frame_info.filename
    line_number = frame_info.lineno

    debug_info = _get_debug_info_string(source_file, line_number)

    if log_func is None:
        log_func = _default_log_func

    if log_before:
        log_func(f"{Fore.BLACK}{description} {Style.RESET_ALL}started {debug_info}")

    # Start timing
    ts = perf_counter_ns()

    # Execute the block
    yield

    # End timing
    te = perf_counter_ns()

    elapsed_ns = te - ts
    elapsed, unit, color = format_time(elapsed_ns)

    # Log in standard editor format with colors if available: filename and line_number
    log_func(
        f"{Fore.BLACK}{description} {Style.RESET_ALL}took: "
        f"{color}{elapsed:,.4f} {unit}{Style.RESET_ALL} "
        f"{debug_info}"
    )
