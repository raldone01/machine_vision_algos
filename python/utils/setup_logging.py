import logging
import sys
from pathlib import Path
import datetime as dt

from colorama import Fore, Style, just_fix_windows_console, Back

from utils.benchmarking import set_default_log_func
from utils.setup_notebook import is_notebook


class ColorFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        else:
            record.asctime = ""

        color_time = Fore.LIGHTBLACK_EX
        level_no = record.levelno
        color_level = Fore.BLACK
        color_background = Back.RESET
        debug_info = ""
        if level_no >= logging.CRITICAL:
            color_level = Fore.RED
            color_background = Back.BLACK
        elif level_no >= logging.ERROR:
            color_level = Fore.RED
        elif level_no >= logging.WARNING:
            color_level = Fore.YELLOW
        elif level_no >= logging.INFO:
            color_level = Fore.GREEN
        elif level_no >= logging.DEBUG:
            color_level = Fore.MAGENTA
            is_just_filename = record.filename == Path(record.filename).name
            if not is_notebook() and is_just_filename:
                debug_info = f" {Fore.LIGHTBLACK_EX}{record.filename}:{record.lineno}{Style.RESET_ALL}"

        s = f"{color_time}{record.asctime} {color_level}{color_background}{record.levelname} {record.name} {Style.RESET_ALL}{record.message}{debug_info}"

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)

        return s


# 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
def set_log_level(log_level: str) -> None:
    log_mapping = logging.getLevelNamesMapping()
    log_level = log_mapping[log_level.upper()]
    logging.getLogger().setLevel(log_level)


# 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
def setup_logging(log_level: str = "DEBUG") -> None:
    just_fix_windows_console()
    logging.basicConfig()

    formatter = ColorFormatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers[:] = []
    logger.addHandler(console_handler)

    set_log_level(log_level)
    set_default_log_func(logging.info)
