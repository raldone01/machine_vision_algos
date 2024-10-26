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


def format_time(elapsed_ns: int) -> tuple:
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


def format_time_str(elapsed_ns: int) -> str:
    elapsed, unit, color = format_time(elapsed_ns)
    return f"{color}{elapsed:,.4f} {unit}{Style.RESET_ALL}"


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

            # Print with optional colored output if colorama is available
            log_func(
                f"{Fore.BLACK}func: {f.__name__!r} {Style.RESET_ALL}took: "
                f"{format_time_str(elapsed_ns)} "
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

    # Log in standard editor format with colors if available: filename and line_number
    log_func(
        f"{Fore.BLACK}{description} {Style.RESET_ALL}took: "
        f"{format_time_str(elapsed_ns)} "
        f"{debug_info}"
    )


class BenchmarkResult:
    def __init__(self):
        self.name = ""
        self.mean = 0
        self.std_dev = 0
        self.min = 0
        self.max = 0
        self.runs = 0
        self.warmup_runs = 0
        self.runtimes = None

    def __str__(self):
        return (
            f"{self.name}: mean={format_time_str(self.mean)}, "
            f"std_dev={format_time_str(self.std_dev)}, "
            f"min={format_time_str(self.min)}, "
            f"max={format_time_str(self.max)}, "
            f"runs={self.runs}, warmup_runs={self.warmup_runs}"
        )

    def __repr__(self):
        return str(self)


def benchmark_fun(
    name, fun, warmup_runs: int = 100, runs: int = 1000, *args, **kwargs
) -> BenchmarkResult:
    runtimes = np.empty(runs, dtype=np.int64)
    # Warm-up runs
    for _ in range(warmup_runs):
        fun(*args, **kwargs)
    # Actual benchmark runs
    for i in range(runs):
        ts = perf_counter_ns()  # Start time
        fun(*args, **kwargs)  # Execute the function with passed args and kwargs
        te = perf_counter_ns()  # End time
        elapsed_ns = te - ts  # Calculate elapsed time
        runtimes[i] = elapsed_ns

    # Calculate statistics
    mean = np.mean(runtimes)
    std_dev = np.std(runtimes)
    min_runtime = np.min(runtimes)
    max_runtime = np.max(runtimes)

    # Return the results
    result = BenchmarkResult()
    result.name = name
    result.mean = mean
    result.std_dev = std_dev
    result.min = min_runtime
    result.max = max_runtime
    result.runs = runs
    result.warmup_runs = warmup_runs
    result.runtimes = runtimes
    return result


def setup_process_for_benchmarking():
    if os.name == "posix":
        # Increase the niceness of the process until it doesn't change anymore
        starting_niceness = os.nice(0)
        last_niceness = starting_niceness
        current_niceness = os.nice(19)
        while last_niceness < current_niceness:
            last_niceness = current_niceness
            current_niceness = os.nice(1)
        print(f"Set process niceness from {starting_niceness} to {last_niceness}")
        # Pin the process to the first core
        os.sched_setaffinity(0, {0})
        print("Pinned process to first core")
    # https://stackoverflow.com/a/1023269/4479969
    if os.name == "nt":
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api
        import win32process

        handle = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
        win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
        win32process.SetProcessAffinityMask(handle, 1)
