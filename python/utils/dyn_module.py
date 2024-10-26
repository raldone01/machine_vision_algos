import importlib.util
import sys
from importlib import reload
from pathlib import Path

from utils.benchmarking import time_line


def get_module_name_from_file_path(file_path: str):
    return Path(file_path).stem


def get_module(file_path: str):
    module_name = get_module_name_from_file_path(file_path)
    return sys.modules.get(module_name)


def load_module(file_path: str):
    module_name = get_module_name_from_file_path(file_path)

    already_loaded = module_name in sys.modules
    if already_loaded:
        with time_line(f"Reloading {module_name}"):
            module = sys.modules[module_name]
            reload(module)
        return module

    with time_line(f"Loading {module_name}"):
        folder = str(Path(file_path).parent)
        if folder not in sys.path:
            sys.path.append(folder)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return module
