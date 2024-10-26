import os
import logging
from functools import wraps
from dataclasses import astuple, dataclass
from time import perf_counter_ns
from datetime import datetime, timezone
from contextlib import contextmanager
import inspect
import unittest.mock as mock
from pathlib import Path

import numpy as np
from icecream import ic
from colorama import Fore, Style

from utils.setup_notebook import source_code_path_is_from_notebook, get_notebook_dir

_default_log_func = logging.info


def set_default_log_func(log_func):
    global _default_log_func
    _default_log_func = log_func


@dataclass
class FormatTimeNsResult:
    elapsed: float
    unit: str
    color: str
    scale: int

    def __iter__(self):
        return iter(astuple(self))

    def __str__(self):
        return f"{self.color}{self.elapsed:,.4f} {self.unit}{Style.RESET_ALL}"


def format_time_ns(elapsed_ns: int) -> FormatTimeNsResult:
    """
    Helper function to format the elapsed time in the appropriate unit
    (seconds, milliseconds, microseconds, or nanoseconds).
    """
    if elapsed_ns >= 1_000_000_000:  # More than or equal to 1 second
        scale = 1_000_000_000
        elapsed = elapsed_ns / scale
        unit = "s"
        color = Fore.RED
    elif elapsed_ns >= 1_000_000:  # More than or equal to 1 millisecond
        scale = 1_000_000
        elapsed = elapsed_ns / scale
        unit = "ms"
        color = Fore.BLUE
    elif elapsed_ns >= 1_000:  # More than or equal to 1 microsecond
        scale = 1_000
        elapsed = elapsed_ns / scale
        unit = "µs"
        color = Fore.GREEN
    else:
        scale = 1
        elapsed = elapsed_ns
        unit = "ns"
        color = Fore.LIGHTBLACK_EX

    return FormatTimeNsResult(elapsed, unit, color, scale)


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
                f"{format_time_ns(elapsed_ns)} "
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
        f"{format_time_ns(elapsed_ns)} "
        f"{debug_info}"
    )


@dataclass
class BenchmarkResult:
    name: str = ""
    mean: float = 0.0
    std_dev: float = 0.0
    min: int = 0
    max: int = 0
    runs: int = 0
    warmup_runs: int = 0
    runtimes: np.ndarray = None
    output: any = None
    start_time: int = 0
    end_time: int = 0

    def __iter__(self):
        return iter(astuple(self))

    def __str__(self):
        return (
            f"{self.name}: mean={format_time_ns(self.mean)}, "
            f"std_dev={format_time_ns(self.std_dev)}, "
            f"min={format_time_ns(self.min)}, "
            f"max={format_time_ns(self.max)}, "
            f"runs={self.runs}, warmup_runs={self.warmup_runs}"
        )


def benchmark_fun(
    name, fun, warmup_runs: int = 100, runs: int = 1000, *args, **kwargs
) -> BenchmarkResult:
    start_time = datetime.now(timezone.utc).timestamp()
    runtimes = np.empty((runs), dtype=np.int64)
    output = fun(*args, **kwargs)
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

    end_time = datetime.now(timezone.utc).timestamp()

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
    result.output = output
    result.start_time = start_time
    result.end_time = end_time
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
